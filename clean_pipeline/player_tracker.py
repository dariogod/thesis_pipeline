import json
import numpy as np
import cv2
import os
import torch
from typing import Dict, List, Optional, Literal
from ultralytics import YOLO
from transformers import DFineForObjectDetection, AutoImageProcessor
import math
from custom_types import BBox, Detection, TrackInfo, FrameDetections

class PlayerTracker:
    def __init__(self, underlying_model: Literal["yolo", "d-fine"] = "yolo"):
        self.underlying_model = underlying_model
        
        if underlying_model == "yolo":
            model_path = 'models/yolo11n.pt'
            
            os.makedirs('models', exist_ok=True)
            if not os.path.exists(model_path):
                print(f"Model file {model_path} not found. Downloading YOLO model...")
                self.model = YOLO("yolo11n.pt")
                os.rename("yolo11n.pt", model_path)
            else:
                self.model = YOLO(model_path)
                
        elif underlying_model == "d-fine":
            model_name = "ustc-community/dfine_x_coco"
            
            # Initialize the D-FINE model and image processor
            self.image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False) #TODO: check use_fast param
            self.model = DFineForObjectDetection.from_pretrained(model_name)
            
            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            # Tracking parameters
            self.next_id: int = 0
            self.tracks: Dict[str, Dict[int, TrackInfo]] = {}  # Dictionary to store active tracks
            self.max_age: int = 30  # Maximum number of frames a track can be inactive before being removed
            self.iou_threshold: float = 0.3  # Minimum IoU to consider as a match
        else:
            raise ValueError(f"Unknown model type: {underlying_model}. Choose 'yolo' or 'd-fine'")
        
        # Class mapping for COCO dataset
        # 'person' is class 0 and 'sports ball' is class 32
        self.class_mapping: Dict[int, str] = {0: 'person', 32: 'sports ball'}
    
    def calculate_iou(self, box1: BBox, box2: BBox) -> float:
        """Calculate IoU between two bounding boxes"""
        # Convert to lists if needed
        box1_list = box1.as_list()
        box2_list = box2.as_list()
        
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1_list
        x1_2, y1_2, x2_2, y2_2 = box2_list
        
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
    
    def update_tracks(self, detections: List[Detection], class_name: str) -> None:
        """Associate detections with existing tracks based on IoU"""
        # If no tracks exist yet, create new tracks for all detections
        if len(self.tracks.get(class_name, {})) == 0:
            new_tracks: Dict[int, TrackInfo] = {}
            for detection in detections:
                track_id = self.next_id
                self.next_id += 1
                new_tracks[track_id] = TrackInfo(
                    bbox=detection.bbox,
                    age=0,
                    confidence=detection.confidence,
                    last_seen=0  # Frame counter when last seen
                )
                detection.track_id = track_id
            self.tracks[class_name] = new_tracks
            return
        
        # Calculate IoU between each detection and each track
        matched_track_ids: List[int] = []
        matched_detection_indices: List[int] = []
        
        for i, detection in enumerate(detections):
            max_iou: float = -1
            best_track_id: int = -1
            
            for track_id, track in self.tracks[class_name].items():
                if track_id in matched_track_ids:
                    continue
                
                iou = self.calculate_iou(detection.bbox, track.bbox)
                if iou > max_iou and iou >= self.iou_threshold:
                    max_iou = iou
                    best_track_id = track_id
            
            if best_track_id != -1:
                matched_track_ids.append(best_track_id)
                matched_detection_indices.append(i)
                detections[i].track_id = best_track_id
                # Update track information
                self.tracks[class_name][best_track_id].bbox = detection.bbox
                self.tracks[class_name][best_track_id].age = 0
                self.tracks[class_name][best_track_id].confidence = detection.confidence
                self.tracks[class_name][best_track_id].last_seen = 0
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[class_name][track_id] = TrackInfo(
                    bbox=detection.bbox,
                    age=0,
                    confidence=detection.confidence,
                    last_seen=0
                )
                detection.track_id = track_id
        
        # Update age of all tracks and remove old ones
        tracks_to_remove: List[int] = []
        for track_id in self.tracks[class_name]:
            if track_id not in matched_track_ids:
                self.tracks[class_name][track_id].age += 1
                self.tracks[class_name][track_id].last_seen += 1
                if self.tracks[class_name][track_id].age > self.max_age:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[class_name][track_id]
    
    def track_players(self, input_path: str, output_dir: Optional[str] = None, store_results: bool = True) -> List[FrameDetections]:
        # Open the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Set up font
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = height / 1000
        font_thickness = math.floor(font_face)
        text_height = cv2.getTextSize("1234567890", font_face, font_scale, font_thickness)[0][1]

        
        # Set up output directory only if we're storing results
        if store_results:
            if output_dir is None:
                base_name = os.path.basename(input_path).split('.')[0]
                model_name = self.underlying_model.replace("-", "_")
                output_dir = os.path.join(f'{model_name}_player_tracker_results', base_name)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Define output paths
            output_video_path = os.path.join(output_dir, 'player_detections.mp4')
            output_json_path = os.path.join(output_dir, 'player_detections.json')
            
            # Set up video writer
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
        
        # List to store all detections
        all_detections: List[FrameDetections] = []
        frame_count: int = 0
        
        # Initialize tracking for each class if using D-FINE
        if self.underlying_model == "d-fine":
            for class_name in self.class_mapping.values():
                if class_name not in self.tracks:
                    self.tracks[class_name] = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            frame_detections: List[Detection] = []
            
            if self.underlying_model == "yolo":
                # Use YOLO to detect and track persons and sports balls in the frame
                results = self.model.track(frame, persist=True, verbose=False, classes=[0, 32])
                
                # Process detection results
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        class_name = self.class_mapping.get(cls_id, 'unknown')

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None
                        
                        # Store detection data
                        bbox = BBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
                        detection = Detection(
                            bbox=bbox,
                            confidence=float(conf),
                            track_id=int(track_id) if track_id is not None else None,
                            class_name=class_name,
                        )
                        frame_detections.append(detection)
                        
            elif self.underlying_model == "d-fine":
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
                
                # Group detections by class for tracking
                class_detections: Dict[str, List[Detection]] = {class_name: [] for class_name in self.class_mapping.values()}
                
                # Extract detection results
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    cls_id = label.item()
                    
                    # Only process persons and sports balls
                    if cls_id in self.class_mapping:
                        class_name = self.class_mapping[cls_id]
                        
                        # Convert box coordinates to integers
                        x1, y1, x2, y2 = map(int, box.cpu().numpy())
                        conf = float(score.cpu().numpy())
                        
                        # Create detection object (track_id will be assigned by the tracker)
                        bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
                        detection = Detection(
                            bbox=bbox,
                            confidence=float(conf),
                            track_id=None,  # Will be assigned by tracker
                            class_name=class_name,
                        )
                        
                        # Add to class-specific detection list
                        class_detections[class_name].append(detection)
                
                # Update tracks for each class
                for class_name, detections in class_detections.items():
                    self.update_tracks(detections, class_name)
                    frame_detections.extend(detections)
            
            # Save detection data for this frame
            frame_data = FrameDetections(
                frame_id=int(frame_count),
                detections=frame_detections
            )
            all_detections.append(frame_data)
            
            if store_results:
                # Draw detections on frame
                for detection in frame_detections:
                    x1, y1, x2, y2 = detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2
                    track_id = detection.track_id
                    class_name = detection.class_name
                    
                    # Different colors for different classes
                    color = (0, 0, 0) if class_name == 'person' else (255, 255, 255)
                    
                    # Draw bounding box
                    cv2.rectangle(img = frame, pt1 = (x1, y1), pt2 = (x2, y2), color = color, thickness = font_thickness)
                    
                    # Draw track ID
                    if track_id is not None:
                        cv2.putText(
                            img = frame, 
                            text = f"ID: {track_id}", 
                            org = (x1, y1 - int(text_height * 0.5)),  # Adjust offset based on text height
                            fontFace = font_face, 
                            fontScale = font_scale, 
                            color = (0, 0, 0), 
                            thickness = font_thickness, 
                            lineType = cv2.LINE_AA)
                
                # Write frame to output video
                out.write(frame)
            
            frame_count += 1
        
        # Release resources
        cap.release()
        if store_results:
            out.release()
        cv2.destroyAllWindows()
        
        # Save detections to JSON if storing results
        if store_results:
            # Convert to dict format for JSON serialization
            detections_json = [frame_data.model_dump() for frame_data in all_detections]
            with open(output_json_path, 'w') as f:
                json.dump(detections_json, f, indent=4)
        
        return all_detections
