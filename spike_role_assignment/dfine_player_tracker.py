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
            
            # Extract detection results
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                cls_id = label.item()
                
                # Only process persons and sports balls
                if cls_id in relevant_classes:
                    class_name = relevant_classes[cls_id]
                    
                    # Convert box coordinates to integers
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    conf = float(score.cpu().numpy())
                    
                    # D-FINE doesn't provide tracking IDs, so we'll use detection index as a placeholder
                    # In a real application, you might want to integrate a dedicated tracker
                    track_id = len(frame_detections)
                    
                    if store_results:
                        # Different colors for different classes
                        color = (0, 0, 0) if cls_id == 0 else (255, 255, 255)  # Black for person, white for ball
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                        # Draw label and track ID
                        label_text = f"{track_id}"
                        font_scale = 0.5
                        font_thickness = 1
                        font = cv2.FONT_HERSHEY_PLAIN
                        cv2.putText(frame, 
                                    label_text, 
                                    (x1, y2 + 7), 
                                    font, font_scale, 
                                    (0, 0, 0), font_thickness, cv2.LINE_AA)
                    
                    # Store detection data
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'track_id': track_id,
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