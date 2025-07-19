# Captcha Recognition System

A comprehensive system for training CNN models to recognize captcha images. The system loads images, extracts text labels from filenames, and prepares data for CNN training.

## Features

- **Image Loading**: Automatically loads images from train/test directories
- **Text Extraction**: Extracts text labels from image filenames
- **Data Preprocessing**: Resizes images, normalizes pixel values, and applies data augmentation
- **CNN Model**: Multi-output CNN that predicts each character position separately
- **Training Pipeline**: Complete training script with callbacks and evaluation
- **Visualization**: Tools to visualize samples and training progress

## System Architecture

The system consists of several key components:

### 1. `load_images.py` - Image Loader
The `CaptchaImageLoader` class handles:
- Loading images from train/test directories
- Extracting text labels from filenames (format: `{label}_{text}_{index}.jpg`)
- Preprocessing images (resize, normalize)
- Creating TensorFlow data generators
- Data augmentation for training

### 2. `captcha_model.py` - CNN Model
The `CaptchaCNN` class provides:
- Multi-output CNN architecture
- Separate output for each character position
- Training and evaluation methods
- Text prediction functionality

### 3. `train_captcha.py` - Training Script
Complete training pipeline with:
- Command-line argument parsing
- Training with callbacks (early stopping, learning rate reduction)
- Model evaluation
- Training history visualization

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data is organized as follows:
```
captcha_data/
└── output/
    ├── train/
    │   ├── real_abc123_00001.jpg
    │   ├── syn_xyz789_00002.jpg
    │   └── ...
    └── test/
        ├── real_def456_00003.jpg
        ├── syn_uvw012_00004.jpg
        └── ...
```

## Usage

### Quick Start

1. **Test the system**:
```bash
python example_usage.py
```

2. **Train a model**:
```bash
python train_captcha.py --epochs 50 --batch_size 32
```

3. **Train with custom parameters**:
```bash
python train_captcha.py \
    --data_dir captcha_data/output \
    --epochs 100 \
    --batch_size 64 \
    --img_size 200 80 \
    --max_length 8 \
    --model_path my_captcha_model.h5 \
    --evaluate
```

### Advanced Usage

#### Using the Image Loader

```python
from load_images import CaptchaImageLoader

# Initialize loader
loader = CaptchaImageLoader(
    data_dir="captcha_data/output",
    img_size=(200, 80),
    chars="0123456789abcdefghijklmnopqrstuvwxyz"
)

# Get dataset information
loader.get_dataset_info()

# Visualize samples
loader.visualize_samples(num_samples=5, split='train')

# Create data generators
train_gen = loader.create_data_generator(split='train', batch_size=32, augment=True)
test_gen = loader.create_data_generator(split='test', batch_size=32, augment=False)
```

#### Using the CNN Model

```python
from captcha_model import CaptchaCNN

# Create model
model = CaptchaCNN(img_size=(200, 80), num_classes=36, max_length=8)
keras_model = model.build_model()

# Train the model
history = model.train(train_gen, test_gen, epochs=50)

# Make predictions
predicted_text = model.predict_text(image_array)
```

## Data Format

The system expects image filenames in the format:
```
{label}_{text}_{index}.jpg
```

Examples:
- `real_abc123_00001.jpg` - Real image with text "abc123"
- `syn_xyz789_00002.jpg` - Synthetic image with text "xyz789"

The text extraction automatically:
1. Splits the filename by underscores
2. Extracts the text part (second component)
3. Handles fallback extraction for different naming patterns

## Model Architecture

The CNN model uses:
- **Input**: RGB images (200×80×3)
- **Convolutional layers**: 4 conv layers with batch normalization and max pooling
- **Dense layers**: 2 dense layers with dropout
- **Output**: 8 separate outputs (one for each character position)
- **Loss**: Categorical crossentropy for each position
- **Optimizer**: Adam with learning rate scheduling

## Training Features

- **Data Augmentation**: Random brightness, contrast, rotation
- **Early Stopping**: Prevents overfitting
- **Learning Rate Reduction**: Automatically reduces LR when validation loss plateaus
- **Model Checkpointing**: Saves best model during training
- **Visualization**: Plots training history

## Evaluation

The system provides comprehensive evaluation:
- Per-character accuracy
- Full text accuracy
- Sample predictions with visualization
- Training history plots

## File Structure

```
captcha-solver/
├── load_images.py          # Image loader and preprocessing
├── captcha_model.py        # CNN model definition
├── train_captcha.py        # Training script
├── example_usage.py        # Usage examples
├── requirements.txt        # Dependencies
├── README.md              # This file
└── captcha_data/
    └── output/
        ├── train/          # Training images
        └── test/           # Test images
```

## Customization

### Changing Image Size
```python
loader = CaptchaImageLoader(img_size=(300, 100))
```

### Adding New Characters
```python
loader = CaptchaImageLoader(chars="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
```

### Modifying Model Architecture
Edit the `build_model()` method in `CaptchaCNN` class.

### Custom Data Augmentation
Modify the `_augment_data()` method in `CaptchaImageLoader`.

## Troubleshooting

### Common Issues

1. **No images found**: Check that your data directory structure is correct
2. **Memory errors**: Reduce batch size or image size
3. **Poor accuracy**: 
   - Increase training data
   - Adjust model architecture
   - Tune hyperparameters
   - Check data quality

### Debugging

- Use `loader.get_dataset_info()` to check data loading
- Use `loader.visualize_samples()` to inspect images
- Check filename format matches expected pattern

## Performance Tips

1. **Use GPU**: Install TensorFlow with GPU support
2. **Data preprocessing**: Use `tf.data` for efficient data loading
3. **Batch size**: Adjust based on available memory
4. **Image size**: Balance between accuracy and speed

## License

This project is licensed under the MIT License - see the LICENSE file for details.