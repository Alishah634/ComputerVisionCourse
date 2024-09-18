import numpy as np
import cv2
import matplotlib.pyplot as plt
from termcolor import cprint 
import os

'''
This was fun!!!
Thank you to the GTA for solving my problem!
'''
ALPHA = "\u03B1"  # GREEK SMALL LETTER ALPHA

# Function to generate a 3x3 rotation homography matrix for a given angle alpha (in degrees)
def rotational_homography(alpha: float) -> np.ndarray:
    try:
        cprint(f"Generating rotation homography for {ALPHA} = {alpha} degrees.", "cyan")
        theta = np.radians(alpha)
        H = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        cprint(f"Rotation homography matrix:\n{H}", "green")
        return H
    except Exception as e:
        cprint(f"Error in rotational_homography: {e}", "red")
        raise

def vertical_tilt_skew_homography(alpha: float) -> np.ndarray:
    try:
        cprint(f"Generating vertical tilting homography for {ALPHA} = {alpha}.", "cyan")
        H = np.array([[1, 0, 0],
                      [alpha, 1, 0],
                      [0, 0, 1]])
        cprint(f"Vertical tilting homography matrix:\n{H}", "green")
        return H
    except Exception as e:
        cprint(f"Error in vertical_tilt_skew_homography: {e}", "red")
        raise

def horizontal_tilt_skew_homography(alpha: float) -> np.ndarray:
    try:
        cprint(f"Generating horizontal tilting homography for {ALPHA} = {alpha}.", "cyan")
        H = np.array([[1, alpha, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        cprint(f"Horizontal tilting homography matrix:\n{H}", "green")
        return H
    except Exception as e:
        cprint(f"Error in horizontal_tilt_skew_homography: {e}", "red")
        raise
    
# Thank you to the GTA for telling me to do the noramlize and denormalization this fixed all my values
def normalization_homography(width: int, height: int) -> np.ndarray:
    try:
        cprint("Generating normalization homography.", "cyan")
        H_norm = np.array([[2 / width, 0, -1],
                           [0, 2 / height, -1],
                           [0, 0, 1]])
        cprint(f"Normalization homography:\n{H_norm}", "green")
        return H_norm
    except Exception as e:
        cprint(f"Error in normalization_homography: {e}", "red")
        raise

def denormalization_homography(width: int, height: int) -> np.ndarray:
    try:
        cprint("Generating de-normalization homography.", "cyan")
        H_denorm = np.array([[width / 2, 0, width / 2],
                             [0, height / 2, height / 2],
                             [0, 0, 1]])
        cprint(f"De-normalization homography:\n{H_denorm}", "green")
        return H_denorm
    except Exception as e:
        cprint(f"Error in denormalization_homography: {e}", "red")
        raise

def composite_homography(H_transform: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        cprint("Generating composite homography.", "cyan")
        H_norm = normalization_homography(width, height)
        H_denorm = denormalization_homography(width, height)
        H_composite = H_denorm @ H_transform @ H_norm
        cprint(f"Composite homography:\n{H_composite}", "green")
        return H_composite
    except Exception as e:
        cprint(f"Error in composite_homography: {e}", "red")
        raise

def apply_homography(H: np.ndarray, image: np.ndarray) -> np.ndarray:
    try:
        cprint("Applying homography using cv2.warpPerspective.", "cyan")
        height, width = image.shape[:2]
        transformed_image = cv2.warpPerspective(image, H, (width, height))
        cprint("Homography applied successfully.", "green")
        return transformed_image
    except Exception as e:
        cprint(f"Error in apply_homography: {e}", "red")
        raise

def create_grid_image(width: int, height: int, grid_spacing: int = 50) -> np.ndarray:
    try:
        cprint(f"Creating grid image of size {width}x{height}.", "cyan")
        image = np.ones((height, width), dtype=np.uint8) * 255  # White background
        for x in range(0, width, grid_spacing):
            cv2.line(image, (x, 0), (x, height), 0, 1)  # Draw vertical lines
        for y in range(0, height, grid_spacing):
            cv2.line(image, (0, y), (width, y), 0, 1)   # Draw horizontal lines
        cprint("Grid image created successfully.", "green")
        return image
    except Exception as e:
        cprint(f"Error in create_grid_image: {e}", "red")
        raise

def display_transformations(image: np.ndarray, transformation_func, alpha_values: list, operation_name: str, save_path: str = None):
    try:
        cprint(f"Displaying transformations for {operation_name}.", "cyan")
        fig, axes = plt.subplots(1, len(alpha_values), figsize=(15, 5))
        for idx, alpha in enumerate(alpha_values):
            H = transformation_func(alpha)
            H_composite = composite_homography(H, image.shape[1], image.shape[0])
            transformed_image = apply_homography(H_composite, image)
            axes[idx].imshow(transformed_image, cmap='gray')
            axes[idx].set_title(f'{operation_name}\n{ALPHA} = {alpha}')
            axes[idx].axis('off')
        
        # Save the figure if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            cprint(f"Figure saved as {save_path}", "green")
        
        plt.show()
        cprint(f"Transformations for {operation_name} displayed successfully.", "green")
    except Exception as e:
        cprint(f"Error in display_transformations: {e}", "red")
        raise

if __name__ == "__main__":
    try:
        cprint("Starting main function.", "cyan")
        width, height = 640, 480
        grid_image = create_grid_image(width, height)

        # Display results for rotation, vertical tilt, and horizontal tilt
        alpha_values_rotation = np.arange(0, 180, 45).tolist() # [0, 45, 90, 135]
        alpha_values_tilt = np.arange(-1, 1, 0.25).tolist() 

        # Rotation
        cprint("Displaying rotation transformations.", "cyan")
        display_transformations(grid_image, rotational_homography, alpha_values_rotation, 'Rotation', save_path='figures/rotation_transformations.png')

        # Vertical tilting
        cprint("Displaying vertical tilting transformations.", "cyan")
        display_transformations(grid_image, vertical_tilt_skew_homography, alpha_values_tilt, 'Vertical Tilting', save_path='figures/vertical_tilting_transformations.png')

        # Horizontal tilting
        cprint("Displaying horizontal tilting transformations.", "cyan")
        display_transformations(grid_image, horizontal_tilt_skew_homography, alpha_values_tilt, 'Horizontal Tilting', save_path='figures/horizontal_tilting_transformations.png')

        cprint("Main function completed.", "green")
    except Exception as e:
        cprint(f"Error in main function: {e}", "red")
        raise
