import json
import numpy as np
import cv2
import os
from ultralytics import YOLO

class PlayerTracker:
    def __init__(self):
        model_path = 'models/yolo11n.pt'
        
        os.makedirs('models', exist_ok=True)
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found. Downloading YOLO model...")
            self.model = YOLO("yolo11n.pt")
            os.rename("yolo11n.pt", model_path)
        else:
            self.model = YOLO(model_path)
        
    
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
                output_dir = os.path.join('player_tracker_results', base_name)
            
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
        
        # Class mapping
        class_mapping = {0: 'person', 32: 'sports ball'}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
                        
            # Use YOLO to detect and track persons and sports balls in the frame
            results = self.model.track(frame, persist=True, verbose=False, classes=[0, 32])  # Class 0 is 'person', 32 is 'sports ball'
            
            frame_detections = []
            
            # Process detection results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = class_mapping.get(cls_id, 'unknown')

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None
                    
                    if store_results:
                        # Different colors for different classes
                        color = (0, 0, 0) if cls_id == 0 else (255, 255, 255)  # Black for person, white for ball
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                        # Draw only track ID text
                        label = f"{track_id}"
                        font_scale = 0.5
                        font_thickness = 1
                        font = cv2.FONT_HERSHEY_PLAIN
                        cv2.putText(frame, 
                                    label, 
                                    (x1, y2 + 7), 
                                    font, font_scale, 
                                    (0, 0, 0), font_thickness, cv2.LINE_AA)
                    
                    # Store detection data
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'track_id': int(track_id) if track_id is not None else None,
                        'class': class_name,
                    }
                    frame_detections.append(detection)
            
            # Save detection data for this frame
            all_detections.append({
                'frame_id': int(frame_count),
                'detections': frame_detections
            })
            
            if store_results:
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
