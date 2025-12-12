"""
Traffic Sign Detection - Inference Script
Use this to detect traffic signs in images after training

Usage:
    python inference.py --model path/to/best.pt --image path/to/image.jpg
    python inference.py --model path/to/best.pt --image path/to/folder --batch
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO, RTDETR
import json

class TrafficSignDetector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """Initialize detector with trained model"""
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Auto-detect model type
        if 'rtdetr' in str(model_path).lower():
            print(f"Loading RT-DETR model: {model_path}")
            self.model = RTDETR(model_path)
            self.model_type = 'rtdetr'
        else:
            print(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            self.model_type = 'yolo'
        
        self.class_names = self.model.names
        print(f"Model loaded successfully! Classes: {len(self.class_names)}")
    
    def detect_single_image(self, image_path, visualize=True, save_path=None):
        """Detect traffic signs in a single image"""
        print(f"\nProcessing: {image_path}")
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        img = cv2.imread(str(image_path))
        
        for result in results:
            boxes = result.boxes
            
            if len(boxes) == 0:
                print("No traffic signs detected")
                return detections
            
            print(f"Found {len(boxes)} traffic sign(s):")
            
            for idx, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                class_name = self.class_names[cls]
                
                detection = {
                    'id': idx + 1,
                    'class_id': cls,
                    'class_name': class_name,
                    'confidence': round(conf, 3),
                    'bbox': {
                        'x1': int(xyxy[0]),
                        'y1': int(xyxy[1]),
                        'x2': int(xyxy[2]),
                        'y2': int(xyxy[3])
                    }
                }
                
                detections.append(detection)
                
                print(f"  [{idx+1}] {class_name} (confidence: {conf:.2%})")
                print(f"      Location: ({int(xyxy[0])}, {int(xyxy[1])}) to ({int(xyxy[2])}, {int(xyxy[3])})")
                
                # Visualize
                if visualize:
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Draw bbox with color based on confidence
                    color = (0, int(255 * conf), int(255 * (1 - conf)))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label
                    label = f"{class_name} {conf:.0%}"
                    
                    # Calculate label size and draw background
                    (label_w, label_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(img, (x1, y1 - label_h - 10), 
                                (x1 + label_w, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(img, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save visualization
        if visualize and save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), img)
            print(f"\nVisualization saved to: {save_path}")
        
        return detections
    
    def detect_batch(self, image_folder, output_folder='./output'):
        """Detect traffic signs in multiple images"""
        image_folder = Path(image_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = [f for f in image_folder.iterdir() 
                      if f.suffix.lower() in image_exts]
        
        print(f"\nFound {len(image_files)} images in {image_folder}")
        
        all_results = {}
        
        for img_file in image_files:
            output_path = output_folder / f"{img_file.stem}_detected{img_file.suffix}"
            detections = self.detect_single_image(
                img_file, 
                visualize=True,
                save_path=output_path
            )
            
            all_results[str(img_file.name)] = detections
        
        # Save JSON report
        json_path = output_folder / 'detections.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nBatch processing complete!")
        print(f"Results saved to: {output_folder}")
        print(f"JSON report: {json_path}")
        
        return all_results
    
    def detect_video(self, video_path, output_path=None):
        """Detect traffic signs in video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if output_path:
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        
        print(f"\nProcessing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Draw detections
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Draw bbox
                    color = (0, int(255 * conf), int(255 * (1 - conf)))
                    cv2.rectangle(frame, (int(x1), int(y1)), 
                                (int(x2), int(y2)), color, 2)
                    
                    # Draw label
                    label = f"{self.class_names[cls]} {conf:.0%}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Write frame
            if output_path:
                out.write(frame)
            
            # Show progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...", end='\r')
        
        cap.release()
        if output_path:
            out.release()
            print(f"\n\nVideo saved to: {output_path}")
        
        print(f"Total frames processed: {frame_count}")

def main():
    parser = argparse.ArgumentParser(description='Traffic Sign Detection')
    parser.add_argument('--model', required=True, help='Path to trained model weights')
    parser.add_argument('--image', required=True, help='Path to image or folder')
    parser.add_argument('--output', default='./output', help='Output folder')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--batch', action='store_true', help='Process folder of images')
    parser.add_argument('--video', action='store_true', help='Process video file')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = TrafficSignDetector(
        args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run detection
    if args.video:
        output_video = Path(args.output) / 'output_video.mp4'
        detector.detect_video(args.image, output_video)
    elif args.batch:
        detector.detect_batch(args.image, args.output)
    else:
        output_path = Path(args.output) / 'detected_image.jpg'
        detections = detector.detect_single_image(
            args.image,
            visualize=not args.no_viz,
            save_path=output_path if not args.no_viz else None
        )
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Detection Summary:")
        print(f"{'='*50}")
        print(f"Total signs detected: {len(detections)}")
        
        if detections:
            print("\nDetailed Results:")
            for det in detections:
                print(f"  - {det['class_name']}: {det['confidence']:.1%} confidence")

if __name__ == '__main__':
    main()
