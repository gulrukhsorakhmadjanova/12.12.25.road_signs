"""
Traffic Sign Detection and Classification Training Pipeline
Supports YOLOv11 and RT-DETR models with Mapillary Vistas dataset

Requirements:
pip install ultralytics opencv-python pillow numpy pandas matplotlib scikit-learn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
"""

import os
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import yaml
from ultralytics import YOLO, RTDETR
import torch

class MapillaryDatasetConverter:
    """Convert Mapillary Vistas dataset to YOLO/RT-DETR format"""
    
    # Mapillary traffic sign class IDs (you may need to adjust based on dataset version)
    TRAFFIC_SIGN_CLASSES = {
        'information--parking--g1': 0,
        'regulatory--stop--g1': 1,
        'regulatory--no-entry--g1': 2,
        'regulatory--yield--g1': 3,
        'regulatory--maximum-speed-limit-5--g1': 4,
        'regulatory--maximum-speed-limit-10--g1': 5,
        'regulatory--maximum-speed-limit-20--g1': 6,
        'regulatory--maximum-speed-limit-30--g1': 7,
        'regulatory--maximum-speed-limit-40--g1': 8,
        'regulatory--maximum-speed-limit-50--g1': 9,
        'regulatory--maximum-speed-limit-60--g1': 10,
        'regulatory--maximum-speed-limit-70--g1': 11,
        'regulatory--maximum-speed-limit-80--g1': 12,
        'regulatory--maximum-speed-limit-90--g1': 13,
        'regulatory--maximum-speed-limit-100--g1': 14,
        'regulatory--maximum-speed-limit-110--g1': 15,
        'regulatory--maximum-speed-limit-120--g1': 16,
        'regulatory--no-parking--g1': 17,
        'regulatory--no-overtaking--g1': 18,
        'regulatory--priority-road--g1': 19,
        'warning--pedestrian-crossing--g1': 20,
        'warning--children--g1': 21,
        'warning--traffic-signals--g1': 22,
        'warning--other-danger--g1': 23,
    }
    
    def __init__(self, mapillary_root, output_root):
        self.mapillary_root = Path(mapillary_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
    def convert_bbox(self, size, box):
        """Convert bbox to YOLO format (normalized xywh)"""
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        x = (box[0] + box[2]) / 2.0
        y = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        return (x * dw, y * dh, w * dw, h * dh)
    
    def process_annotations(self, split='training'):
        """Process Mapillary annotations and convert to YOLO format"""
        print(f"Processing {split} split...")
        
        # Create output directories
        img_dir = self.output_root / split / 'images'
        label_dir = self.output_root / split / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Read Mapillary annotations
        ann_file = self.mapillary_root / split / 'v2.0' / 'annotations.json'
        
        if not ann_file.exists():
            print(f"Warning: {ann_file} not found!")
            return
            
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Process each image
        processed = 0
        for img_data in data['images']:
            img_id = img_data['id']
            img_name = img_data['file_name']
            width = img_data['width']
            height = img_data['height']
            
            # Find all annotations for this image
            annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
            
            # Filter for traffic signs only
            sign_annotations = []
            for ann in annotations:
                category = next((cat for cat in data['categories'] if cat['id'] == ann['category_id']), None)
                if category and category['name'] in self.TRAFFIC_SIGN_CLASSES:
                    sign_annotations.append((ann, category['name']))
            
            # Skip images without traffic signs
            if not sign_annotations:
                continue
            
            # Copy image
            src_img = self.mapillary_root / split / 'images' / img_name
            dst_img = img_dir / img_name
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            # Create label file
            label_file = label_dir / (Path(img_name).stem + '.txt')
            with open(label_file, 'w') as f:
                for ann, sign_name in sign_annotations:
                    class_id = self.TRAFFIC_SIGN_CLASSES[sign_name]
                    bbox = ann['bbox']  # [x, y, width, height]
                    
                    # Convert to YOLO format
                    x_center, y_center, w, h = self.convert_bbox(
                        (width, height),
                        [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    )
                    
                    f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")
            
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} images...")
        
        print(f"Completed {split} split: {processed} images with traffic signs")
    
    def create_yaml_config(self):
        """Create dataset configuration YAML for YOLO"""
        config = {
            'path': str(self.output_root.absolute()),
            'train': 'training/images',
            'val': 'validation/images',
            'test': 'testing/images',
            'nc': len(self.TRAFFIC_SIGN_CLASSES),
            'names': {v: k for k, v in self.TRAFFIC_SIGN_CLASSES.items()}
        }
        
        yaml_path = self.output_root / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Dataset config saved to {yaml_path}")
        return yaml_path

class TrafficSignTrainer:
    """Train YOLOv11 or RT-DETR models for traffic sign detection"""
    
    def __init__(self, dataset_yaml):
        self.dataset_yaml = dataset_yaml
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
    def train_yolo(self, model_size='x', epochs=100, imgsz=640, batch=16):
        """
        Train YOLOv11 model
        model_size: 'n', 's', 'm', 'l', 'x' (nano to extra-large)
        """
        print(f"\n{'='*50}")
        print(f"Training YOLOv11{model_size} model")
        print(f"{'='*50}\n")
        
        # Load pretrained model
        model = YOLO(f'yolo11{model_size}.pt')
        
        # Train
        results = model.train(
            data=self.dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project='runs/traffic_signs',
            name=f'yolo11{model_size}',
            patience=20,
            save=True,
            plots=True,
            amp=True,  # Automatic Mixed Precision
            verbose=True
        )
        
        # Validate
        metrics = model.val()
        
        print(f"\nTraining completed!")
        print(f"Best weights saved to: {model.trainer.best}")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        
        return model, results
    
    def train_rtdetr(self, model_size='l', epochs=100, imgsz=640, batch=16):
        """
        Train RT-DETR model
        model_size: 'l' or 'x' (large or extra-large)
        """
        print(f"\n{'='*50}")
        print(f"Training RT-DETR-{model_size} model")
        print(f"{'='*50}\n")
        
        # Load pretrained model
        model = RTDETR(f'rtdetr-{model_size}.pt')
        
        # Train
        results = model.train(
            data=self.dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project='runs/traffic_signs',
            name=f'rtdetr-{model_size}',
            patience=20,
            save=True,
            plots=True,
            amp=True,
            verbose=True
        )
        
        # Validate
        metrics = model.val()
        
        print(f"\nTraining completed!")
        print(f"Best weights saved to: {model.trainer.best}")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        
        return model, results

class TrafficSignDetector:
    """Inference class for trained models"""
    
    def __init__(self, model_path, model_type='yolo'):
        """
        model_path: path to trained weights
        model_type: 'yolo' or 'rtdetr'
        """
        if model_type == 'yolo':
            self.model = YOLO(model_path)
        else:
            self.model = RTDETR(model_path)
        
        self.class_names = self.model.names
        
    def predict(self, image_path, conf=0.25, iou=0.45):
        """
        Detect traffic signs in image
        Returns: list of detections with [class, confidence, bbox, label]
        """
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            save=False,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                detections.append({
                    'class_id': cls,
                    'class_name': self.class_names[cls],
                    'confidence': conf,
                    'bbox': xyxy.tolist(),
                })
        
        return detections
    
    def predict_and_visualize(self, image_path, output_path=None, conf=0.25):
        """Predict and save visualization"""
        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=True if output_path is None else False
        )
        
        if output_path:
            img = cv2.imread(image_path)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Draw bbox
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{self.class_names[cls]} {conf:.2f}"
                    cv2.putText(img, label, (int(x1), int(y1)-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(output_path, img)
            print(f"Result saved to {output_path}")
        
        return results

def main():
    """Main training pipeline"""
    
    # Configuration
    MAPILLARY_ROOT = "/path/to/mapillary_vistas"  # Change this
    OUTPUT_ROOT = "./traffic_signs_dataset"
    
    print("="*60)
    print("Traffic Sign Detection Training Pipeline")
    print("="*60)
    
    # Step 1: Convert dataset
    print("\nStep 1: Converting Mapillary dataset to YOLO format...")
    converter = MapillaryDatasetConverter(MAPILLARY_ROOT, OUTPUT_ROOT)
    
    # Process all splits
    for split in ['training', 'validation']:
        converter.process_annotations(split)
    
    # Create config
    dataset_yaml = converter.create_yaml_config()
    
    # Step 2: Train models
    trainer = TrafficSignTrainer(dataset_yaml)
    
    # Train YOLOv11x (best accuracy)
    print("\n" + "="*60)
    print("Training YOLOv11x (Best Version)")
    print("="*60)
    yolo_model, yolo_results = trainer.train_yolo(
        model_size='x',
        epochs=100,
        imgsz=640,
        batch=8  # Adjust based on your GPU
    )
    
    # Train RT-DETR-x
    print("\n" + "="*60)
    print("Training RT-DETR-x")
    print("="*60)
    rtdetr_model, rtdetr_results = trainer.train_rtdetr(
        model_size='x',
        epochs=100,
        imgsz=640,
        batch=8
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModels saved in: runs/traffic_signs/")
    print(f"Dataset config: {dataset_yaml}")

if __name__ == "__main__":
    main()
