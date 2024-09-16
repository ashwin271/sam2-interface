import os
import torch
import numpy as np
from PIL import Image, ImageTk
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import tkinter as tk
from tkinter import filedialog, messagebox

# Initialize global variables
points = []
masks = []
image_bgr = None
image_np = None
mask_overlay = None
panel = None  # Define panel globally
image_path = ""  # Will be set via UI

# Use absolute paths for checkpoint and model configuration
checkpoint_path = os.path.abspath("./checkpoints/sam2_hiera_small.pt")
model_cfg_path = os.path.abspath("D:/code/segment-anything/sam2_configs/sam2_hiera_s.yaml")  # Replace with the correct config file path

def load_image(image_path):
    """
    Load an image from the specified path and convert it to a PIL Image in RGB format.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image:\n{e}")
        return None

def preprocess_image(image):
    """
    Preprocess the PIL Image:
    - Convert to NumPy array.
    - Normalize to [0, 1].
    - Convert to PyTorch tensor in [C, H, W] format.
    """
    image_np_local = np.array(image)
    
    # Check if the image has the correct number of channels
    if image_np_local.ndim != 3 or image_np_local.shape[2] != 3:
        raise ValueError("Image must have 3 channels (RGB).")
    
    # Normalize the image to [0, 1] and convert to float32
    image_np_local = image_np_local.astype(np.float32) / 255.0
    
    # Convert to CHW format
    image_tensor = torch.from_numpy(image_np_local).permute(2, 0, 1).float()
    
    return image_tensor

def run_image_prediction(image_path, checkpoint_path, model_cfg_path, points, point_labels=[1]):
    """
    Run image prediction using SAM 2:
    - Load and preprocess the image.
    - Load the SAM 2 model.
    - Set the image for the predictor.
    - Predict the mask based on the input points.
    - Update the mask overlay.
    
    Args:
        image_path (str): Path to the input image.
        checkpoint_path (str): Path to the SAM 2 checkpoint file.
        model_cfg_path (str): Path to the SAM 2 model configuration file.
        points (list): List of (x, y) coordinates for the mask generation.
        point_labels (list, optional): Labels for the points (1 for foreground, 0 for background). Defaults to [1].
    """
    global masks, mask_overlay
    
    if not os.path.isfile(image_path):
        messagebox.showerror("Error", f"Invalid image path: {image_path}")
        return
    
    if not os.path.isfile(checkpoint_path):
        messagebox.showerror("Error", f"Checkpoint file not found: {checkpoint_path}")
        return
    
    if not os.path.isfile(model_cfg_path):
        messagebox.showerror("Error", f"Model configuration file not found: {model_cfg_path}")
        return
    
    # Load the model
    try:
        model = build_sam2(model_cfg_path, checkpoint_path)
        predictor = SAM2ImagePredictor(model)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to build SAM2 model:\n{e}")
        return

    # Load and preprocess the image
    image = load_image(image_path)
    if image is None:
        return
    image_tensor = preprocess_image(image)

    # Convert the image tensor to CPU and NumPy array in [H, W, C] format
    image_np_local = image_tensor.permute(1, 2, 0).cpu().numpy()

    # Prepare prompts
    prompts = {
        "point_coords": points,          # List of points
        "point_labels": point_labels     # Corresponding labels
    }

    # Perform prediction
    with torch.inference_mode():
        try:
            predictor.set_image(image_np_local)
        except NotImplementedError as e:
            messagebox.showerror("Error", f"Error in set_image: {e}")
            return

        try:
            masks_generated, _, _ = predictor.predict(
                point_coords=prompts["point_coords"],
                point_labels=prompts["point_labels"]
            )
        except AssertionError as e:
            messagebox.showerror("Error", f"AssertionError during predict: {e}")
            return
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error during predict: {e}")
            return

    # Store the masks for visualization
    masks = masks_generated

    # Update the mask overlay
    update_mask_overlay()

def update_mask_overlay():
    """
    Update the mask overlay with all predicted masks.
    """
    global mask_overlay, image_bgr, masks

    if masks.size > 0:
        combined_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, (mask.astype(np.uint8) * 255))
        
        # Create a color mask (red channel)
        color_mask = np.zeros_like(image_bgr)
        color_mask[:, :, 2] = combined_mask  # Red channel

        # Overlay the mask on the image
        mask_overlay = cv2.addWeighted(image_bgr, 0.7, color_mask, 0.3, 0)
    else:
        mask_overlay = image_bgr.copy()

def visualize_image():
    """
    Display the image with the mask overlay in the Tkinter window.
    """
    global mask_overlay, panel

    if mask_overlay is not None:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk

def select_image():
    """
    Open a file dialog to select an image and initialize the display.
    """
    global image_bgr, image_np, mask_overlay, points, masks, image_path

    file_path = filedialog.askopenfilename(title="Select Image",
                                           filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if file_path:
        if not os.path.isfile(file_path):
            messagebox.showerror("Error", f"The selected path is not a file: {file_path}")
            return

        # Reset points and masks
        points = []
        masks = []
        mask_overlay = None

        # Load and preprocess the image
        image = load_image(file_path)
        if image is None:
            return
        image_tensor = preprocess_image(image)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

        # Convert the image to uint8 for display
        image_uint8 = (image_np * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

        # Initialize the mask overlay
        mask_overlay = image_bgr.copy()

        # Update image_path
        image_path = file_path

        # Update the display
        visualize_image()

def clear_mask():
    """
    Clear all segmentation masks.
    """
    global masks, mask_overlay, points
    masks = []
    points = []  # Clear the points as well
    if image_bgr is not None:
        mask_overlay = image_bgr.copy()
        visualize_image()

def on_click(event):
    """
    Handle mouse click events on the image.
    """
    global points, mask_overlay, image_path

    if not os.path.isfile(image_path):
        messagebox.showwarning("Warning", "Please select a valid image first.")
        return

    # Get the coordinates of the click relative to the panel
    x = event.x
    y = event.y

    # Calculate the scaling factor between the displayed image and the actual image
    panel_width = panel.winfo_width()
    panel_height = panel.winfo_height()
    image_height, image_width = mask_overlay.shape[:2]
    
    scale_x = image_width / panel_width
    scale_y = image_height / panel_height
    
    # Adjust coordinates based on scaling
    actual_x = int(x * scale_x)
    actual_y = int(y * scale_y)

    # Ensure coordinates are within image bounds
    if actual_x < 0 or actual_y < 0 or actual_x >= image_width or actual_y >= image_height:
        messagebox.showwarning("Warning", "Click is outside the image boundaries.")
        return

    # Add the point
    points.append([actual_x, actual_y])
    print(f"Point selected: [{actual_x}, {actual_y}]")

    # Run the prediction with all points
    run_image_prediction(image_path, checkpoint_path, model_cfg_path, points, point_labels=[1]*len(points))

    # Update the visualized image with the new mask
    visualize_image()

def main():
    """
    Main function to set up the GUI and run the application.
    """
    global panel

    # Set up the Tkinter window
    root = tk.Tk()
    root.title("SAM2 Interactive Segmentation")

    # Set window size
    root.geometry("800x600")

    # Create buttons frame
    btn_frame = tk.Frame(root)
    btn_frame.pack(side="top", fill="x", padx=10, pady=10)

    # Create buttons
    btn_select = tk.Button(btn_frame, text="Select Image", command=select_image)
    btn_select.pack(side="left", padx=5)

    btn_clear = tk.Button(btn_frame, text="Clear Mask", command=clear_mask)
    btn_clear.pack(side="left", padx=5)

    # Create a panel to display the image
    panel = tk.Label(root)
    panel.pack(side="top", fill="both", expand=True)

    # Bind the click event to the panel
    panel.bind("<Button-1>", on_click)

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()