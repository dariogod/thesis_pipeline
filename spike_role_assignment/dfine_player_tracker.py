import json
import numpy as np
import cv2
import os
import torch
from transformers import DFineForObjectDetection, AutoImageProcessor

class DFinePlayerTracker:
    def __init__(self):
        model_name = "ustc-community/dfine_x_coco"
        
        # Initialize the D-FINE model and image processor
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = DFineForObjectDetection.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Tracking parameters
        self.next_id = 0
        self.tracks = {}  # Dictionary to store active tracks
        self.max_age = 30  # Maximum number of frames a track can be inactive before being removed
        self.iou_threshold = 0.3  # Minimum IoU to consider as a match
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate area of each box
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection coordinates
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        # Calculate intersection area
        w = max(0, xi2 - xi1)
        h = max(0, yi2 - yi1)
        intersection = w * h
        
        # Calculate IoU
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        
        return iou
    
    def update_tracks(self, detections, class_name):
        """Associate detections with existing tracks based on IoU"""
        # If no tracks exist yet, create new tracks for all detections
        if len(self.tracks.get(class_name, {})) == 0:
            new_tracks = {}
            for detection in detections:
                track_id = self.next_id
                self.next_id += 1
                new_tracks[track_id] = {
                    'bbox': detection['bbox'],
                    'age': 0,
                    'confidence': detection['confidence'],
                    'last_seen': 0  # Frame counter when last seen
                }
                detection['track_id'] = track_id
            self.tracks[class_name] = new_tracks
            return
        
        # Calculate IoU between each detection and each track
        matched_track_ids = []
        matched_detection_indices = []
        
        for i, detection in enumerate(detections):
            max_iou = -1
            best_track_id = -1
            
            for track_id, track in self.tracks[class_name].items():
                if track_id in matched_track_ids:
                    continue
                
                iou = self.calculate_iou(detection['bbox'], track['bbox'])
                if iou > max_iou and iou >= self.iou_threshold:
                    max_iou = iou
                    best_track_id = track_id
            
            if best_track_id != -1:
                matched_track_ids.append(best_track_id)
                matched_detection_indices.append(i)
                detections[i]['track_id'] = best_track_id
                # Update track information
                self.tracks[class_name][best_track_id]['bbox'] = detection['bbox']
                self.tracks[class_name][best_track_id]['age'] = 0
                self.tracks[class_name][best_track_id]['confidence'] = detection['confidence']
                self.tracks[class_name][best_track_id]['last_seen'] = 0
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[class_name][track_id] = {
                    'bbox': detection['bbox'],
                    'age': 0,
                    'confidence': detection['confidence'],
                    'last_seen': 0
                }
                detection['track_id'] = track_id
        
        # Update age of all tracks and remove old ones
        tracks_to_remove = []
        for track_id in self.tracks[class_name]:
            if track_id not in matched_track_ids:
                self.tracks[class_name][track_id]['age'] += 1
                self.tracks[class_name][track_id]['last_seen'] += 1
                if self.tracks[class_name][track_id]['age'] > self.max_age:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[class_name][track_id]
    
    def track_players(self, input_path, output_dir=None, store_results=True):
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Set up output directory only if we're storing results
        if store_results:
            if output_dir is None:
                base_name = os.path.basename(input_path).split('.')[0]
                output_dir = os.path.join('dfine_player_tracker_results', base_name)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Define output paths
            output_video_path = os.path.join(output_dir, 'player_detections.mp4')
            output_json_path = os.path.join(output_dir, 'player_detections.json')
            frames_dir = os.path.join(output_dir, 'frames')
            
            # Create directories
            os.makedirs(frames_dir, exist_ok=True)
            
            # Set up video writer
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
        
        # List to store all detections
        all_detections = []
        frame_count = 0
        
        # Class mapping for COCO dataset
        # D-FINE is pre-trained on COCO where 'person' is class 0 and 'sports ball' is class 32
        relevant_classes = {0: 'person', 32: 'sports ball'}
        
        # Initialize tracking for each class
        for class_name in relevant_classes.values():
            if class_name not in self.tracks:
                self.tracks[class_name] = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            # Convert BGR to RGB (D-FINE expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_height, img_width = rgb_frame.shape[:2]
            
            # Prepare inputs for D-FINE
            inputs = self.image_processor(images=rgb_frame, return_tensors="pt").to(self.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process results
            results = self.image_processor.post_process_object_detection(
                outputs, 
                threshold=0.5,  # Confidence threshold
                target_sizes=[(img_height, img_width)]
            )[0]  # Get first image in batch
            
            frame_detections = []
            
            # Group detections by class for tracking
            class_detections = {class_name: [] for class_name in relevant_classes.values()}
            
            # Extract detection results
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                cls_id = label.item()
                
                # Only process persons and sports balls
                if cls_id in relevant_classes:
                    class_name = relevant_classes[cls_id]
                    
                    # Convert box coordinates to integers
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    conf = float(score.cpu().numpy())
                    
                    # Create detection object (track_id will be assigned by the tracker)
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'track_id': None,  # Will be assigned by tracker
                        'class': class_name,
                    }
                    
                    # Add to class-specific detection list
                    class_detections[class_name].append(detection)
            
            # Update tracks for each class
            for class_name, detections in class_detections.items():
                self.update_tracks(detections, class_name)
                frame_detections.extend(detections)
            
            # Save detection data for this frame
            all_detections.append({
                'frame_id': int(frame_count),
                'detections': frame_detections
            })
            
            if store_results:
                # Draw detections on frame
                for detection in frame_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    track_id = detection['track_id']
                    class_name = detection['class']
                    
                    # Different colors for different classes
                    color = (0, 0, 0) if class_name == 'person' else (255, 255, 255)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    
                    # Draw track ID
                    label_text = f"{track_id}"
                    font_scale = 1.0  # Increased from 0.5 to 1.0
                    font_thickness = 2  # Increased from 1 to 2
                    font = cv2.FONT_HERSHEY_PLAIN
                    cv2.putText(frame, 
                                label_text, 
                                (x1, y2 + 20),  # Increased y-offset from 15 to 20 to accommodate larger text
                                font, font_scale, 
                                (0, 0, 255), font_thickness, cv2.LINE_AA)
                
                # Write frame to output video
                out.write(frame)
                
                # Save frame image
                frame_path = os.path.join(frames_dir, f'frame_{frame_count:04d}.jpg')
                cv2.imwrite(frame_path, frame)
            
            frame_count += 1
        
        # Release resources
        cap.release()
        if store_results:
            out.release()
        cv2.destroyAllWindows()
        
        # Save detections to JSON if storing results
        if store_results:
            with open(output_json_path, 'w') as f:
                json.dump(all_detections, f, indent=4)
        
        return all_detections 