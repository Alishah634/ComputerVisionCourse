import numpy as np
import cv2
from typing import List
from skimage.draw import polygon
import numpy as np
import cv2
from typing import List, Tuple
from termcolor import cprint

'''Util function for converting to numpy array'''
def conv_to_numpy(conv: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    for points in conv:
        points = np.array(points).flatten()         
        cprint(f"The points: {points} have a type {type(points)}", "white") # DEBUG STATMENT!!!
        cprint(f"\n\n\nPoints After ->  P: {points[0]} ,Q:{points[1]} ,R:{points[2]} ,S:{points[3]}\n", "white") # DEBUG STATEMENT!!!
    return conv

'''Util function for printing out a formatted numpy array'''
def print_matrix(matrix: np.ndarray):
    for row in matrix:
        cprint(f" ".join(f"{value: .3f}" for value in row),"white")
        
def verify_homography(src_pts: np.ndarray, dest_pts: np.ndarray)-> np.ndarray:
    return np.array((cv2.find_homography(src_pts, dest_pts))).flatten()

'''Compute the Homography H using the concept Ax = b -> x = A^-1*b :'''    
def compute_homography(src_pts: np.ndarray, dest_pts: np.ndarray) -> np.ndarray:
    def construct_matrix_A(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        A = np.zeros((8, 8))
        for i in range(4):
            x, y = src_pts[i]
            x_prime, y_prime = dst_pts[i]
            A[2*i] = [x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime]
            A[2*i + 1] = [0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime]
        return A
    
    A = construct_matrix_A(src_pts, dest_pts)
    b = np.array([coord for point in dest_pts for coord in point])
    print_matrix(A)  # DEBUG STATEMENT!!!
    # cprint(f"{b}","blue")  # DEBUG STATEMENT!!!
    try:
        H_temp = np.dot(np.linalg.inv(A), b)
        H = np.append(H_temp, 1).reshape((3, 3))
    except np.linalg.LinAlgError:
        cprint("Error: Matrix inversion failed. The matrix A might be singular.","red")
        H = np.eye(3)  # Return identity matrix as fallback just to be safe, however this should never be reached
    return H

'''Create an mask, the region of interest(roi) so we only change that region when applying the transform:'''    
def create_roi_mask(points: List[List[int]], target_image: np.ndarray, target_img_name: str, source_img_name: str) -> np.ndarray:
    """Create an empty mask for the ROI of the given image dimensions."""
    def generate_mask(image_shape: Tuple[int, int]) -> np.ndarray:
        height, width = image_shape
        return np.zeros((height, width), dtype=np.uint8)

    """Extract row and column coordinates from the provided points."""
    def extract_polygon_coordinates(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        columns = points[:, 0]
        rows = points[:, 1]
        return rows, columns

    """Draw the overlay of provided mask using row and column coordinates."""
    def visualize_mask(mask: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> None:
        rr, cc = polygon(rows, columns)
        mask[rr, cc] = 255
        return

    """Save the generated mask as an image file."""
    def save_mask_image(mask: np.ndarray, filename: str) -> None:
        cv2.imwrite(filename, mask)
        return

    mask = generate_mask(target_image.shape[:2])
    rows, cols = extract_polygon_coordinates(np.array(points))
    visualize_mask(mask, rows, cols)
    mask_filename = f'roi_{source_img_name}_to_{target_img_name}.jpg'
    save_mask_image(mask, mask_filename)

    return mask


def apply_homography_transform(roi_mask: (None | np.ndarray), src_image: np.ndarray, dst_image: np.ndarray, homography_matrix: np.ndarray) -> np.ndarray:
    """Applies the homography transformation to a region of interest (ROI) in the destination image."""
    
    """Return the width and height of the image."""
    def get_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
        return image.shape[1], image.shape[0]

    """Apply inverse homography to the pixel location (x, y)."""
    def transform_pixel_location(h_matrix: np.ndarray, x: int, y: int) -> np.ndarray:
        pixel_coords = [x, y, 1]
        transformed_coords = np.dot(np.linalg.inv(h_matrix), pixel_coords)
        return (transformed_coords / transformed_coords[2]).astype(int)

    """Copy the transformed pixel from the source image to the destination image if within bounds."""
    def copy_transformed_pixel_to_dest(source_img: np.ndarray, dest_img: np.ndarray, src_x: int, src_y: int, dst_x: int, dst_y: int) -> None:
        if 0 <= src_x < source_img.shape[1] and 0 <= src_y < source_img.shape[0]:
            dest_img[dst_y, dst_x] = source_img[src_y, src_x]
        return

    """Apply the transformation to pixels only within the ROI."""
    def apply_homography_transform_to_mask(roi: np.ndarray, src_img: np.ndarray, dest_img: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
        width, height = get_image_dimensions(dest_img)
        for y in range(height):
            for x in range(width):
                if roi[y, x] == 255: 
                    transformed_coords = transform_pixel_location(h_matrix, x, y)
                    copy_transformed_pixel_to_dest(src_img, dest_img, transformed_coords[0], transformed_coords[1], x, y)
        return dest_img

    """Apply homography transformation to the entire destination image when there is no ROI."""
    def apply_homography_transform(src_img: np.ndarray, dest_img: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
        width, height = get_image_dimensions(dest_img)
        for y in range(height):
            for x in range(width):
                transformed_coords = transform_pixel_location(h_matrix, x, y)
                copy_transformed_pixel_to_dest(src_img, dest_img, transformed_coords[0], transformed_coords[1], x, y)
        return dest_img
    
    result_image = np.copy(dst_image)
    if roi_mask is not None:
        result_image = apply_homography_transform_to_mask(roi_mask, src_image, result_image, homography_matrix)
    else:
        result_image = apply_homography_transform(src_image, result_image, homography_matrix)

    return result_image

def compute_affine_transform(src_pts: np.ndarray, dest_pts: np.ndarray) -> np.ndarray:
    src_pts, dest_pts = np.array(src_pts).flatten() , np.array(dest_pts).flatten()  
    cprint(f"{type(src_pts), type(dest_pts)}", "blue")  # DEBUG STATEMENT!!!
    # Initialize matrix A and vector b
    A = np.zeros((8, 6))
    b = dest_pts  # Destination points are vector b
    # Construct the A matrix
    for i in range(4):
        src_x, src_y = src_pts[2 * i], src_pts[2 * i + 1]
        A[2 * i:2 * i + 2, :] = np.array([
            [src_x, src_y, 1, 0, 0, 0],
            [0, 0, 0, src_x, src_y, 1]
        ])
    # Compute H = (A^-1) b and reshape to form affine matrix
    try:
        H_temp = np.dot(np.linalg.pinv(A), b)
        H = np.reshape(np.append(H_temp, [0, 0, 1]), (3, 3))
    except np.linalg.LinAlgError:
        cprint("Error: Matrix inversion failed. The matrix A might be singular.","red")
        H = np.eye(3)  # Return identity matrix as fallback just to be safe, however this should never be reached
    return H


'''Specific function for breaking down the logic of Task 1'''
def Task1(image_points: List[np.ndarray]):
    '''_____________________________________________For part 1 of Task 1:_____________________________________________'''
    image_points = conv_to_numpy(image_points.copy())
    cprint(f"{type(image_points)}", "blue") # DEBUG STATEMENT!!!
    cprint(f"{type(image_points[0])}", "blue") # DEBUG STATEMENT!!!
    cprint(f"{type(image_points[0][0])}", "blue") # DEBUG STATEMENT!!!
    
    img1 = cv2.imread('img1.jpg')
    img2 = cv2.imread('img2.jpg')
    img3 = cv2.imread('img3.jpg')
    alex_image = cv2.imread('alex_honnold.jpg')
    
    # Find homography b/w image d) and a), i.e the 1st(0 index, first image) and the 4th image(3 index, last image):
    H_da = compute_homography(image_points[-1], image_points[0])
    # Find homography b/w image d) and b), i.e the 2nd(1 index, second image) and the 4th image(3 index, last image):
    H_db = compute_homography(image_points[-1], image_points[1]) 
    # Find homography b/w image d) and c), i.e the 3rd(2 index, third image) and the 4th image(3 index, last image):
    H_dc = compute_homography(image_points[-1], image_points[2])
    # H_dc = verify_homography(image_points[1], image_points[2]) # DEBUG STATMENT!!!
    cprint(f"Calculated Homography\n","green")


    # Find the region of interest:
    img_1_roi = create_roi_mask(image_points[0], img1, 'img_1_a', 'alex_image')
    img_2_roi = create_roi_mask(image_points[1], img2, 'img_2_b', 'alex_image')
    img_3_roi = create_roi_mask(image_points[2], img3, 'img_3_c', 'alex_image')
    cprint(f"Found ROI\n","green")
    # Apply the transforms:
    res_da = apply_homography_transform(img_1_roi, alex_image, img1, H_da)
    res_db = apply_homography_transform(img_2_roi, alex_image, img2, H_db)
    res_dc = apply_homography_transform(img_3_roi, alex_image, img3, H_dc)
    cprint(f"Applied Transform","green")
    
    #  Write the result to the images:
    cv2.imwrite('proj_d_to_a.jpg', res_da)
    cv2.imwrite('proj_d_to_b.jpg', res_db)
    cv2.imwrite('proj_d_to_c.jpg', res_dc)
    cprint(f"Wrote to the file","green")

    '''_____________________________________________For part 2 of Task 1:_____________________________________________'''
    cprint(f"Starting Task 1 part 2:","green")
    H_ab = compute_homography(image_points[0], image_points[1])
    H_bc = compute_homography(image_points[1], image_points[2])
    # H_bc = verify_homography(image_points[1], image_points[2]) # DEBUG STATMENT!!!
    # roi_combination = create_roi_mask(image_points[0], img1, 'img_1_a', 'img_2_b')  # Adjust the ROI if needed
    H_combination = np.matmul(H_bc, H_ab) 
    res_combination = apply_homography_transform(None, img1, np.zeros_like(img1), H_combination)
    cv2.imwrite('task1_part2.jpg' , res_combination) 
    cprint(f"Finishing Task 1 part 2:","green")
    
    '''_____________________________________________For part 3 of Task 1:_____________________________________________'''
    cprint(f"Starting Affine Task 1 part 3 portion:\n","green")
    H_affine_da = compute_affine_transform(image_points[-1], image_points[0])
    H_affine_db = compute_affine_transform(image_points[-1], image_points[1])
    H_affine_dc = compute_affine_transform(image_points[-1], image_points[2])
    
    res_affine_da = apply_homography_transform(img_1_roi, alex_image, img1, H_affine_da)
    res_affine_db = apply_homography_transform(img_2_roi, alex_image, img2, H_affine_db)
    res_affine_dc = apply_homography_transform(img_3_roi, alex_image, img3, H_affine_dc)
    
    cv2.imwrite('proj_affine_d_to_a.jpg', res_affine_da)
    cv2.imwrite('proj_affine_d_to_b.jpg', res_affine_db)
    cv2.imwrite('proj_affine_d_to_c.jpg', res_affine_dc)
    cprint(f"Finished Affine Task 1 part 3 portion:\n","green")

def Task2(image_points):
    cprint(f"Running Task2:","green")
    '''_____________________________________________For part 1 of Task 2:_____________________________________________'''
    image_points = conv_to_numpy(image_points.copy())
    cprint(f"{type(image_points)}", "green") # DEBUG STATEMENT!!!
    cprint(f"{type(image_points[0])}", "green") # DEBUG STATEMENT!!!
    cprint(f"{type(image_points[0][0])}", "green") # DEBUG STATEMENT!!!
    
    img1 = cv2.imread('MyImages/board_a.jpeg')
    img2 = cv2.imread('MyImages/board_b.jpeg')
    img3 = cv2.imread('MyImages/board_c.jpeg')
    liam_image = cv2.imread('MyImages/liam_image.jpg')
    
    # Find homography b/w image d) and a), i.e the 1st(0 index, first image) and the 4th image(3 index, last image):
    H_da = compute_homography(image_points[-1], image_points[0])
    # Find homography b/w image d) and b), i.e the 2nd(1 index, second image) and the 4th image(3 index, last image):
    H_db = compute_homography(image_points[-1], image_points[1]) 
    # Find homography b/w image d) and c), i.e the 3rd(2 index, third image) and the 4th image(3 index, last image):
    H_dc = compute_homography(image_points[-1], image_points[2])
    cprint(f"Calculated Homography\n","green")


    # Find the region of interest:
    img_1_roi = create_roi_mask(image_points[0], img1, 'board_a', 'liam_image')
    img_2_roi = create_roi_mask(image_points[1], img2, 'board_b', 'liam_image')
    img_3_roi = create_roi_mask(image_points[2], img3, 'board_c', 'liam_image')
    cprint(f"Found ROI\n","green")
    
    # Apply the transforms:
    res_da = apply_homography_transform(img_1_roi, liam_image, img1, H_da)
    res_db = apply_homography_transform(img_2_roi, liam_image, img2, H_db)
    res_dc = apply_homography_transform(img_3_roi, liam_image, img3, H_dc)
    cprint(f"Applied Transform","green")
    
    # Write the result to the images:
    cv2.imwrite('proj_liam_to_a.jpg', res_da)
    cv2.imwrite('proj_liam_to_b.jpg', res_db)
    cv2.imwrite('proj_liam_to_c.jpg', res_dc)
    cprint(f"Wrote to the file","green")

    '''_____________________________________________For part 2 of Task 2:_____________________________________________'''
    cprint(f"Starting Task 2 part 2:","green")
    H_ab = compute_homography(image_points[0], image_points[1])
    H_bc = compute_homography(image_points[1], image_points[2])
    # roi_combination = create_roi_mask(image_points[0], img1, 'img_1_a', 'img_2_b')  # Adjust the ROI if needed
    H_combination = np.matmul(H_bc, H_ab) 
    res_combination = apply_homography_transform(None, img1, np.zeros_like(img1),  H_combination)
    cv2.imwrite('task2_part2.jpg' , res_combination) 
    cprint(f"Finishing Task 2 part 2:","green")
    
    '''_____________________________________________For part 3 of Task 2:_____________________________________________'''
    cprint(f"Starting Affine Task 2 part 3 portion:\n","green")
    H_affine_da = compute_affine_transform(image_points[-1], image_points[0])
    H_affine_db = compute_affine_transform(image_points[-1], image_points[1])
    H_affine_dc = compute_affine_transform(image_points[-1], image_points[2])
    
    res_affine_da = apply_homography_transform(img_1_roi, liam_image, img1, H_affine_da)
    res_affine_db = apply_homography_transform(img_2_roi, liam_image, img2, H_affine_db)
    res_affine_dc = apply_homography_transform(img_3_roi, liam_image, img3, H_affine_dc)
    
    cv2.imwrite('proj_affine_laim_to_a.jpg', res_affine_da)
    cv2.imwrite('proj_affine_laim_to_b.jpg', res_affine_db)
    cv2.imwrite('proj_affine_laim_to_c.jpg', res_affine_dc)
    cprint(f"Finished Affine Task 1 part 3 portion:\n","green")

if __name__ == '__main__':
    # P Q R S
    img_1_a = [    [437,862],  [698,3150], [2391, 2268], [2530, 901] ]   
    img_2_b = [    [526,1448], [490,2681], [1900,2760],  [1831,833]  ]   
    img_3_c = [    [1190,575],  [278,1794], [1791,3126],  [2876,2282] ]   
    img_alex_d = [ [7,6],   [6,652],    [777,659],    [770,6]     ]   
    image_points = [img_1_a, img_2_b, img_3_c, img_alex_d]
    Task1(image_points)
    
    # Task 2 image points: 
    # P Q R S
    board_1_a = [    [172,1214 ],  [1364,1285], [1228, 139],  [442, 127] ]   
    board_2_b = [    [1113,81 ],  [317,90 ], [189,1240],  [1380,1181],   ]   
    board_3_c = [    [1104,211],  [368,460], [748,1433 ],  [1361,1143] ]   
    img_liam = [   [5,6     ],  [2,579   ], [798,578   ],  [798,2   ]     ]   
    image_points = [board_1_a, board_2_b, board_3_c, img_liam]
    
    Task2(image_points)
    