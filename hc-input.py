import torch
import numpy as np
from PIL import Image
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def load_image(image_path):
    """
    Load an image from the specified path and convert it to a PIL Image in RGB format.
    """
    image = Image.open(image_path).convert("RGB")
    return image

def preprocess_image(image):
    """
    Preprocess the PIL Image:
    - Convert to NumPy array.
    - Normalize to [0, 1].
    - Convert to PyTorch tensor in [C, H, W] format.
    """
    image_np = np.array(image)
    
    # Check if the image has the correct number of channels
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("Image must have 3 channels (RGB).")
    
    # Normalize the image to [0, 1] and convert to float32
    image_np = image_np.astype(np.float32) / 255.0
    
    # Convert to CHW format
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
    
    return image_tensor

def run_image_prediction(image_path, checkpoint_path, model_cfg_path, point, point_label=1):
    """
    Run image prediction using SAM 2:
    - Load and preprocess the image.
    - Load the SAM 2 model.
    - Set the image for the predictor.
    - Predict the mask based on the input point.
    - Visualize the resulting mask.
    
    Args:
        image_path (str): Path to the input image.
        checkpoint_path (str): Path to the SAM 2 checkpoint file.
        model_cfg_path (str): Path to the SAM 2 model configuration file.
        point (list or tuple): (x, y) coordinates for the mask generation.
        point_label (int, optional): Label for the point (1 for foreground, 0 for background). Defaults to 1.
    """
    # Load the model
    model = build_sam2(model_cfg_path, checkpoint_path)
    predictor = SAM2ImagePredictor(model)

    # Load and preprocess the image
    image = load_image(image_path)
    image_tensor = preprocess_image(image)

    # Debugging: Print the shape and type of the image tensor
    print(f"Image tensor shape (C, H, W): {image_tensor.shape}")
    print(f"Image tensor dtype: {image_tensor.dtype}")

    # Convert the image tensor to CPU and NumPy array in [H, W, C] format
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    # Debugging: Print the shape and dtype of the NumPy image
    print(f"Image NumPy shape (H, W, C): {image_np.shape}")
    print(f"Image NumPy dtype: {image_np.dtype}")

    # Prepare prompts
    prompts = {
        "point_coords": [point],      # List of points
        "point_labels": [point_label] # Corresponding labels
    }

    # Perform prediction
    with torch.inference_mode():
        try:
            predictor.set_image(image_np)
        except NotImplementedError as e:
            print(f"Error in set_image: {e}")
            return

        try:
            masks, _, _ = predictor.predict(
                point_coords=prompts["point_coords"],
                point_labels=prompts["point_labels"]
            )
        except AssertionError as e:
            print(f"AssertionError during predict: {e}")
            return
        except Exception as e:
            print(f"Unexpected error during predict: {e}")
            return

    # Print or save the results
    print(f"Predicted masks: {masks}")

    # Visualize the mask
    visualize_mask(image_np, masks[0])

def visualize_mask(image, mask):
    """
    Visualize the original image and the predicted mask using OpenCV.

    Args:
        image (np.ndarray): Original image in [H, W, C] format, dtype float32.
        mask (np.ndarray): Predicted mask array.
    """
    # Ensure mask is a binary mask and convert to uint8
    mask = mask.astype(np.uint8) * 255

    # Convert the image to uint8
    image_uint8 = (image * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

    # Create a color mask (red channel)
    color_mask = np.zeros_like(image_bgr)
    color_mask[:, :, 2] = mask  # Red channel

    # Overlay the mask on the image
    overlay = cv2.addWeighted(image_bgr, 0.7, color_mask, 0.3, 0)

    # Display the original image and the overlay
    cv2.imshow("Original Image", image_bgr)
    cv2.imshow("Image with Mask", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "./dog-pic.png"  # Replace with the path to your image
    checkpoint_path = "./checkpoints/sam2_hiera_small.pt"
    model_cfg_path = "sam2_hiera_s.yaml"  # Replace with the correct config file path

    # User input point (x, y) in image coordinates
    point = [206, 143]  # Replace with the actual point coordinates
    point_label = 1      # 1 for foreground, 0 for background

    run_image_prediction(image_path, checkpoint_path, model_cfg_path, point, point_label)