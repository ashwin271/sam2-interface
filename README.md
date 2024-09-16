# SAM2 Interface

This repository provides a Python interface for interactive image segmentation using the Segment Anything Model 2 (SAM2). Users can perform real-time segmentation by clicking on images to specify points of interest or by providing hardcoded input points.

## Features

- **Real-time Segmentation**: Click on the image to specify points and see the segmentation mask in real-time.
- **Support for Single Point Input**: Easily specify a single point for segmentation.
- **Visualization**: Visualize the original image and the segmentation mask using OpenCV.

## Requirements

- Python 3.6+
- PyTorch
- OpenCV
- NumPy
- Pillow

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/sam2-interface.git
   cd sam2-interface
   ```

2. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Hardcoded Input Segmentation

The `hc-input.py` script allows users to specify a hardcoded point for segmentation.

1. Update the paths to your image, checkpoint, and model configuration file in the script:

   ```python
   image_path = "./dog-pic.png"  # Replace with the path to your image
   checkpoint_path = "./checkpoints/sam2_hiera_small.pt"
   model_cfg_path = "sam2_hiera_s.yaml"  # Replace with the correct config file path
   ```

2. Update the hardcoded point coordinates:

   ```python
   point = [206, 143]  # Replace with the actual point coordinates
   ```

3. Run the script:

   ```sh
   python hc-input.py
   ```

4. The segmentation mask will be displayed based on the hardcoded point.

### Real-time Single Point Segmentation

The `realtime-single-pt.py` script allows users to click on an image to specify a single point for segmentation.

1. Update the paths to your image, checkpoint, and model configuration file in the script:

   ```python
   image_path = "./dog-pic.png"  # Replace with the path to your image
   checkpoint_path = "./checkpoints/sam2_hiera_small.pt"
   model_cfg_path = "sam2_hiera_s.yaml"  # Replace with the correct config file path
   ```

2. Run the script:

   ```sh
   python realtime-single-pt.py
   ```

3. Click on the image to specify a point. The segmentation mask will be displayed in real-time.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

