# Traffic Sign Detection & Classification

Complete training pipeline for detecting and classifying traffic signs using **YOLOv11** and **RT-DETR** on the Mapillary Vistas dataset.

## 🎯 Features

- ✅ Train with **YOLOv11** (latest YOLO version) - Best accuracy
- ✅ Train with **RT-DETR** (Real-Time Detection Transformer) - Fast & accurate
- ✅ Supports **multiple traffic signs** in single image
- ✅ Automatic dataset conversion from Mapillary Vistas
- ✅ Complete inference pipeline for images, batches, and videos
- ✅ 24+ traffic sign classes (expandable)

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- 50GB+ storage for dataset

### Installation

```bash
# Clone the repository (or download the files)
git clone <your-repo-url>
cd traffic-sign-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python pillow numpy pandas matplotlib scikit-learn pyyaml

# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Dataset Setup

### 1. Download Mapillary Vistas Dataset

Visit: https://www.mapillary.com/dataset/vistas

Download the following:
- Training images
- Validation images
- Training annotations (v2.0)
- Validation annotations (v2.0)

### 2. Dataset Structure

Organize your dataset like this:

```
mapillary_vistas/
├── training/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── v2.0/
│       └── annotations.json
├── validation/
│   ├── images/
│   └── v2.0/
│       └── annotations.json
└── testing/
    └── images/
```

## 🚀 Training

### Quick Start

Edit the paths in `train.py`:

```python
MAPILLARY_ROOT = "/path/to/mapillary_vistas"  # Your dataset path
OUTPUT_ROOT = "./traffic_signs_dataset"        # Converted dataset output
```

Then run:

```bash
python train.py
```

### Training Options

**Train YOLOv11 only:**
```python
# In train.py, comment out RT-DETR training section
trainer = TrafficSignTrainer(dataset_yaml)
yolo_model, results = trainer.train_yolo(
    model_size='x',    # Options: 'n', 's', 'm', 'l', 'x'
    epochs=100,
    imgsz=640,
    batch=8           # Adjust based on GPU memory
)
```

**Train RT-DETR only:**
```python
rtdetr_model, results = trainer.train_rtdetr(
    model_size='x',    # Options: 'l', 'x'
    epochs=100,
    imgsz=640,
    batch=8
)
```

### Model Sizes

| Model | Size | Speed | Accuracy | GPU Memory |
|-------|------|-------|----------|------------|
| YOLOv11n | 2.6MB | Fastest | Good | 4GB |
| YOLOv11s | 9.4MB | Very Fast | Better | 6GB |
| YOLOv11m | 20.1MB | Fast | Very Good | 8GB |
| YOLOv11l | 25.3MB | Medium | Excellent | 10GB |
| YOLOv11x | 56.9MB | Slower | **Best** | 16GB |
| RT-DETR-l | 32MB | Fast | Excellent | 12GB |
| RT-DETR-x | 67MB | Medium | **Best** | 16GB |

### Training Tips

1. **Reduce batch size if out of memory:**
   ```python
   batch=4  # or batch=2 for lower memory
   ```

2. **Use mixed precision training (automatic):**
   - Already enabled with `amp=True`
   - Saves ~40% GPU memory

3. **Monitor training:**
   ```bash
   # Training logs and plots saved in:
   runs/traffic_signs/yolo11x/
   runs/traffic_signs/rtdetr-x/
   ```

## 🔍 Inference

### Single Image

```bash
python inference.py --model runs/traffic_signs/yolo11x/weights/best.pt --image test_image.jpg
```

### Batch Processing

```bash
python inference.py --model runs/traffic_signs/yolo11x/weights/best.pt --image ./test_images/ --batch --output ./results/
```

### Video Processing

```bash
python inference.py --model runs/traffic_signs/yolo11x/weights/best.pt --image video.mp4 --video --output ./output/
```

### Advanced Options

```bash
python inference.py \
  --model path/to/best.pt \
  --image test.jpg \
  --conf 0.5 \           # Confidence threshold (0-1)
  --iou 0.45 \           # IoU threshold for NMS
  --output ./results/ \  # Output directory
  --no-viz               # Disable visualization
```

## 🎨 Using the Model in Your Code

```python
from inference import TrafficSignDetector

# Initialize detector
detector = TrafficSignDetector('path/to/best.pt', conf_threshold=0.3)

# Detect in single image
detections = detector.detect_single_image(
    'test.jpg',
    visualize=True,
    save_path='output.jpg'
)

# Process results
for det in detections:
    print(f"Found: {det['class_name']}")
    print(f"Confidence: {det['confidence']:.2%}")
    print(f"Location: {det['bbox']}")
```

## 📁 Project Structure

```
traffic-sign-detection/
├── train.py                    # Main training script
├── inference.py                # Inference script
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── mapillary_vistas/          # Your dataset (download separately)
├── traffic_signs_dataset/     # Converted dataset (auto-generated)
│   ├── training/
│   ├── validation/
│   └── dataset.yaml
└── runs/                      # Training outputs
    └── traffic_signs/
        ├── yolo11x/
        │   └── weights/
        │       └── best.pt    # Best YOLO model
        └── rtdetr-x/
            └── weights/
                └── best.pt    # Best RT-DETR model
```

## 🎯 Supported Traffic Signs

The system supports 24+ traffic sign classes including:

- **Regulatory Signs:** Stop, Yield, No Entry, Speed Limits (5-120 km/h), No Parking, etc.
- **Warning Signs:** Pedestrian Crossing, Children, Traffic Signals, Other Danger
- **Information Signs:** Parking, Priority Road

You can easily add more classes by editing `TRAFFIC_SIGN_CLASSES` in `train.py`.

## 📊 Performance Metrics

After training, check these files for metrics:

- `runs/traffic_signs/yolo11x/results.png` - Training curves
- `runs/traffic_signs/yolo11x/confusion_matrix.png` - Class predictions
- `runs/traffic_signs/yolo11x/val_batch0_pred.jpg` - Sample predictions

Key metrics:
- **mAP50:** Mean Average Precision at IoU=0.50
- **mAP50-95:** Mean Average Precision at IoU=0.50:0.95
- **Precision:** Ratio of correct positive predictions
- **Recall:** Ratio of detected actual objects

## 🔧 Troubleshooting

### Out of Memory Error
```python
# Reduce batch size in train.py
batch=4  # or batch=2
```

### CUDA Not Available
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Slow Training
- Use smaller model: `model_size='m'` or `'s'`
- Reduce image size: `imgsz=416`
- Enable multi-GPU: The code automatically uses all available GPUs

### No Traffic Signs Detected
- Lower confidence threshold: `--conf 0.2`
- Check if image quality is sufficient
- Ensure model is trained properly

## 🚀 Advanced Usage

### Fine-tune on Custom Dataset

1. Add your images to `traffic_signs_dataset/training/images/`
2. Create corresponding labels in `traffic_signs_dataset/training/labels/`
3. Update `dataset.yaml` with new classes
4. Resume training:

```python
model = YOLO('runs/traffic_signs/yolo11x/weights/last.pt')
model.train(data='traffic_signs_dataset/dataset.yaml', epochs=50)
```

### Export Model

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/traffic_signs/yolo11x/weights/best.pt')

# Export to different formats
model.export(format='onnx')      # ONNX
model.export(format='torchscript')  # TorchScript
model.export(format='tflite')    # TensorFlow Lite
model.export(format='coreml')    # CoreML (iOS)
```

### Use on Edge Devices

For Raspberry Pi, Jetson Nano, or mobile:

1. Export to TFLite or ONNX
2. Use smaller model (yolo11n or yolo11s)
3. Reduce image size to 320 or 416

## 📝 License

This project uses:
- **Ultralytics YOLOv11:** AGPL-3.0 License
- **Mapillary Vistas Dataset:** CC BY-NC-SA 4.0

## 🤝 Contributing

Feel free to:
- Add more traffic sign classes
- Improve data augmentation
- Optimize inference speed
- Add new features

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Ultralytics docs: https://docs.ultralytics.com
3. Open an issue on GitHub

## 🎉 Results Example

After training, you can detect traffic signs like this:

```
Processing: test_image.jpg
Found 3 traffic sign(s):
  [1] regulatory--stop--g1 (confidence: 94.2%)
      Location: (245, 156) to (389, 298)
  [2] regulatory--maximum-speed-limit-50--g1 (confidence: 87.6%)
      Location: (523, 201) to (612, 287)
  [3] warning--pedestrian-crossing--g1 (confidence: 91.3%)
      Location: (789, 145) to (891, 245)
```

Happy training! 🚦🤖
