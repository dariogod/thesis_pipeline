import os
import numpy as np
import scipy.io as sio
import cv2
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from perspective_transform.util.synthetic_util import SyntheticUtil
from perspective_transform.util.iou_util import IouUtil
from perspective_transform.util.projective_camera import ProjectiveCamera
from perspective_transform.models.models import create_model
from perspective_transform.deep.siamese import BranchNetwork, SiameseNetwork
from perspective_transform.deep.camera_dataset import CameraDataset
from arguments import Arguments
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Perspective_Transform:
    def __init__(self):
        self.query_index = 0
        self.current_directory = os.path.join(os.getcwd(), 'weights')

        # Load database features and cameras
        deep_database_path = os.path.join(self.current_directory, "data/deep/feature_camera_91k.mat")
        data = sio.loadmat(deep_database_path)
        self.database_features = data['features']
        self.database_cameras = data['cameras']

        # Initialize deep feature extractor
        self.deep_model_path = os.path.join(self.current_directory, "deep_network.pth")
        self.deep_net, self.data_transform = self.initialize_deep_feature_extractor()

        # Initialize GAN model
        self.model = self.initialize_two_GAN()

        # Initialize nearest neighbor search
        self.neigh = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.database_features)

        print('Perspective Transform initialized successfully!')

    def homography_matrix(self, image):
        image = cv2.resize(image, (1280, 720))
        edge_map, seg_map = self.testing_two_GAN(image)

        test_features = self.generate_deep_feature(edge_map)

        # Retrieve closest camera from database
        _, indices = self.neigh.kneighbors([test_features[self.query_index]])
        retrieved_index = indices[0][0]

        retrieved_camera_data = self.database_cameras[retrieved_index]
        u, v, fl = retrieved_camera_data[0:3]
        rod_rot = retrieved_camera_data[3:6]
        cc = retrieved_camera_data[6:9]

        retrieved_camera = ProjectiveCamera(fl, u, v, cc, rod_rot)
        retrieved_h = IouUtil.template_to_image_homography_uot(retrieved_camera, template_h=74, template_w=115)

        model_points = sio.loadmat(os.path.join(self.current_directory, "data/worldcup2014.mat"))['points']
        model_line_index = sio.loadmat(os.path.join(self.current_directory, "data/worldcup2014.mat"))['line_segment_index']

        retrieved_image = SyntheticUtil.camera_to_edge_image(retrieved_camera_data, model_points, model_line_index, 720, 1280, line_width=2)

        # Refine using Lucas-Kanade
        dist_threshold = 50
        query_dist = np.minimum(SyntheticUtil.distance_transform(edge_map), dist_threshold)
        retrieved_dist = np.minimum(SyntheticUtil.distance_transform(retrieved_image), dist_threshold)

        h_retrieved_to_query = SyntheticUtil.find_transform(retrieved_dist, query_dist)
        refined_h = h_retrieved_to_query @ retrieved_h

        warp_img = cv2.warpPerspective(seg_map, np.linalg.inv(refined_h), (115, 74), borderMode=cv2.BORDER_CONSTANT)

        return np.linalg.inv(refined_h), warp_img

    def initialize_deep_feature_extractor(self):
        branch = BranchNetwork()
        net = SiameseNetwork(branch)
        if os.path.isfile(self.deep_model_path):
            checkpoint = torch.load(self.deep_model_path, map_location=device)
            net.load_state_dict(checkpoint['state_dict'])

        net = net.to(device)
        net.eval()
        cudnn.benchmark = True

        normalize = transforms.Normalize(mean=[0.0188], std=[0.128])
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        return net, data_transform

    def generate_deep_feature(self, edge_map):
        edge_map = cv2.resize(edge_map, (320, 180))
        edge_map = cv2.cvtColor(edge_map, cv2.COLOR_RGB2GRAY)
        edge_map = np.expand_dims(edge_map, axis=0)  # add channel dimension

        dataset = CameraDataset(edge_map, edge_map, batch_size=1, num_batch=-1, data_transform=self.data_transform, is_train=False)

        features = []
        with torch.no_grad():
            for i in range(len(dataset)):
                x, _ = dataset[i]
                x = x.to(device)
                feat = self.deep_net.feature_numpy(x)
                features.append(feat)

        return np.vstack(features)

    def testing_two_GAN(self, image):
        image = Image.fromarray(image)
        osize = [256, 256]
        image = transforms.Compose([
            transforms.Resize(osize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(osize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])(image)
        image = image.unsqueeze(0)

        self.model.set_input(image)
        self.model.test()
        visuals = self.model.get_current_visuals()

        edge_map = cv2.resize(visuals['fake_D'], (1280, 720))
        seg_map = cv2.resize(visuals['fake_C'], (1280, 720))

        return edge_map, seg_map

    def initialize_two_GAN(self):
        # Store original sys.argv and replace it temporarily to avoid conflict
        # with the main script's command line arguments
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]  # Keep only the script name
        
        # Initialize arguments
        opt = Arguments().parse()
        opt.nThreads = 1
        opt.batchSize = 1
        opt.serial_batches = True
        opt.no_flip = True
        opt.continue_train = False
        
        # Restore original sys.argv
        sys.argv = original_argv

        model = create_model(opt)
        return model
