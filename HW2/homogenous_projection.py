import numpy as np
import cv2
from typing import List
from skimage.draw import polygon
import numpy as np
import cv2
from typing import List

def conv_to_numpy(conv: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    for points in conv:
        points = np.array(points).flatten()         
        print(f"The points: {points} have a type {type(points)}") # DEBUG STATMENT!!!
        print(f"\n\n\nPoints After ->  P: {points[0]} ,Q:{points[1]} ,R:{points[2]} ,S:{points[3]}\n") # DEBUG STATEMENT!!!
    return conv

def print_matrix(matrix: np.ndarray):
    for row in matrix:
        print(" ".join(f"{value: .3f}" for value in row))
        
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
    # print_matrix(A)  # DEBUG STATEMENT!!!
    # print(b)  # DEBUG STATEMENT!!!
    try:
        H_temp = np.dot(np.linalg.inv(A), b)
        H = np.append(H_temp, 1).reshape((3, 3))
    except np.linalg.LinAlgError:
        print("Error: Matrix inversion failed. The matrix A might be singular.")
        H = np.eye(3)  # Return identity matrix as fallback
    return H

def get_roi_map(pts, dest_img, dest_img_name, src_img_name):
    roi_map = np.zeros((dest_img.shape[0], dest_img.shape[1]))
    pts = np.array(pts)  # Convert to numpy array for easier slicing
    col = pts[:, 0]
    row = pts[:, 1]
    rows, cols = polygon(row, col)
    roi_map[rows, cols] = 255
    filename = f'roi_{src_img_name}_to_{dest_img_name}.jpg'
    cv2.imwrite(filename, roi_map)
    return roi_map

def apply_transform(dest_img, source_img, H, roi_map= None):
    dest = np.copy(dest_img)
    (W_s, H_s) = source_img.shape[0:2]
    (W_d, H_d) = dest_img.shape[0:2]
    if roi_map is None:
        for i in range(H_d):
            for j in range(W_d):
                x_hc = [i, j, 1]
                xp_hc = np.dot(np.linalg.inv(H), x_hc)  # inverting H for inverse homography
                xp = (xp_hc / xp_hc[2]).astype(int)
                if 0 <= xp[1] < W_s and 0 <= xp[0] < H_s:
                    dest[j, i] = source_img[xp[1], xp[0]]
    else:
        for i in range(H_d):
            for j in range(W_d):
                if roi_map[j, i] == 255:
                    x_hc = [i, j, 1]
                    xp_hc = np.dot(np.linalg.inv(H), x_hc)  # inverting H for inverse homography
                    xp = (xp_hc / xp_hc[2]).astype(int)
                    if 0 <= xp[1] < W_s and 0 <= xp[0] < H_s:
                        dest[j, i] = source_img[xp[1], xp[0]]
    return dest

def compute_affine_transform(src_pts: np.ndarray, dest_pts: np.ndarray) -> np.ndarray:
    # Convert lists to numpy arrays 
    src_pts = np.array(src_pts).flatten()  
    dest_pts = np.array(dest_pts).flatten()  
    print(type(src_pts), type(dest_pts))  # DEBUG STATEMENT!!!
    # Initialize matrix A and vector b
    A = np.zeros((8, 6))
    b = dest_pts  # Destination points as vector b
    
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
        print("Error: Matrix inversion failed. The matrix A might be singular.")
        H = np.eye(3)  # Return identity matrix as fallback
    
    return H

def Task1(image_points: List[np.ndarray]):
    '''_____________________________________________For part 1 of Task 1:_____________________________________________'''
    image_points = conv_to_numpy(image_points.copy())
    print(type(image_points)) # DEBUG STATEMENT!!!
    print(type(image_points[0])) # DEBUG STATEMENT!!!
    print(type(image_points[0][0])) # DEBUG STATEMENT!!!
    
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
    print(f"Calculated Homography\n")

    # Find the region of interest:
    img_1_roi = get_roi_map(image_points[0], img1, 'img_1_a', 'alex_image')
    img_2_roi = get_roi_map(image_points[1], img2, 'img_2_b', 'alex_image')
    img_3_roi = get_roi_map(image_points[2], img3, 'img_3_c', 'alex_image')
    print(f"Found ROI\n")
    
    # Apply the transforms:
    res_da = apply_transform(img1, alex_image, H_da)
    res_db = apply_transform(img2, alex_image, H_db)
    res_dc = apply_transform(img3, alex_image, H_dc)
    print(f"Applied Transform")
    
    #  Write the result to the images:
    cv2.imwrite('proj_d_to_a.jpg', res_da)
    cv2.imwrite('proj_d_to_b.jpg', res_db)
    cv2.imwrite('proj_d_to_c.jpg', res_dc)
    print(f"Wrote to the file")

    '''_____________________________________________For part 2 of Task 1:_____________________________________________'''
    print("Starting Task 1 part 2:")
    H_ab = compute_homography(image_points[0], image_points[1])
    H_bc = compute_homography(image_points[1], image_points[2])
    H_combination = np.matmul(H_bc, H_ab) 
    res_combination = apply_transform(np.zeros_like(img1), img1, H_combination)
    cv2.imwrite('task1_part2.jpg' , res_combination) 
    print("Finishing Task 1 part 2:")
    
    '''_____________________________________________For part 3 of Task 1:_____________________________________________'''
    print("Starting Affine Task 1 part 3 portion:\n")
    H_affine_da = compute_affine_transform(image_points[-1], image_points[0])
    H_affine_db = compute_affine_transform(image_points[-1], image_points[1])
    H_affine_dc = compute_affine_transform(image_points[-1], image_points[2])
    
    res_affine_da = apply_transform(img1, alex_image, H_affine_da, img_1_roi)
    res_affine_db = apply_transform(img2, alex_image, H_affine_db, img_2_roi)
    res_affine_dc = apply_transform(img3, alex_image, H_affine_dc, img_3_roi)
    
    cv2.imwrite('proj_affine_d_to_a.jpg', res_affine_da)
    cv2.imwrite('proj_affine_d_to_b.jpg', res_affine_db)
    cv2.imwrite('proj_affine_d_to_c.jpg', res_affine_dc)
    print("Finished Affine Task 1 part 3 portion:\n")

def Task2(image_points):
    print("Running Task2")
    '''_____________________________________________For part 1 of Task 2:_____________________________________________'''
    image_points = conv_to_numpy(image_points.copy())
    print(type(image_points)) # DEBUG STATEMENT!!!
    print(type(image_points[0])) # DEBUG STATEMENT!!!
    print(type(image_points[0][0])) # DEBUG STATEMENT!!!
    
    img1 = cv2.imread('MyImages/board_a.jpg')
    img2 = cv2.imread('MyImages/board_b.jpg')
    img3 = cv2.imread('MyImages/board_c.jpg')
    liam_image = cv2.imread('liam_image.jpg')
    
    # Find homography b/w image d) and a), i.e the 1st(0 index, first image) and the 4th image(3 index, last image):
    H_da = compute_homography(image_points[-1], image_points[0])
    # Find homography b/w image d) and b), i.e the 2nd(1 index, second image) and the 4th image(3 index, last image):
    H_db = compute_homography(image_points[-1], image_points[1]) 
    # Find homography b/w image d) and c), i.e the 3rd(2 index, third image) and the 4th image(3 index, last image):
    H_dc = compute_homography(image_points[-1], image_points[2])
    print(f"Calculated Homography\n")

    # Find the region of interest:
    img_1_roi = get_roi_map(image_points[0], img1, 'board_a', 'liam_image')
    img_2_roi = get_roi_map(image_points[1], img2, 'board_b', 'liam_image')
    img_3_roi = get_roi_map(image_points[2], img3, 'board_c', 'liam_image')
    print(f"Found ROI\n")
    
    # Apply the transforms:
    res_da = apply_transform(img1, liam_image, H_da)
    res_db = apply_transform(img2, liam_image, H_db)
    res_dc = apply_transform(img3, liam_image, H_dc)
    print(f"Applied Transform")
    
    # Write the result to the images:
    cv2.imwrite('proj_liam_to_a.jpg', res_da)
    cv2.imwrite('proj_liam_to_b.jpg', res_db)
    cv2.imwrite('proj_liam_to_c.jpg', res_dc)
    print(f"Wrote to the file")

    '''_____________________________________________For part 2 of Task 2:_____________________________________________'''
    print("Starting Task 2 part 2:")
    H_ab = compute_homography(image_points[0], image_points[1])
    H_bc = compute_homography(image_points[1], image_points[2])
    H_combination = np.matmul(H_bc, H_ab) 
    res_combination = apply_transform(np.zeros_like(img1), img1, H_combination)
    cv2.imwrite('task1_part2.jpg' , res_combination) 
    print("Finishing Task 2 part 2:")
    
    '''_____________________________________________For part 3 of Task 2:_____________________________________________'''
    print("Starting Affine Task 2 part 3 portion:\n")
    H_affine_da = compute_affine_transform(image_points[-1], image_points[0])
    H_affine_db = compute_affine_transform(image_points[-1], image_points[1])
    H_affine_dc = compute_affine_transform(image_points[-1], image_points[2])
    
    res_affine_da = apply_transform(img1, liam_image, H_affine_da, img_1_roi)
    res_affine_db = apply_transform(img2, liam_image, H_affine_db, img_2_roi)
    res_affine_dc = apply_transform(img3, liam_image, H_affine_dc, img_3_roi)
    
    cv2.imwrite('proj_affine_laim_to_a.jpg', res_affine_da)
    cv2.imwrite('proj_affine_laim_to_b.jpg', res_affine_db)
    cv2.imwrite('proj_affine_laim_to_c.jpg', res_affine_dc)
    print("Finished Affine Task 1 part 3 portion:\n")



if __name__ == '__main__':
    img_1_a = [    [437,862],  [698,3150], [2391, 2268], [2530, 901] ]   
    img_2_b = [    [526,1448], [490,2681], [1900,2760],  [1831,833]  ]   
    img_3_c = [    [1190,575],  [278,1794], [1791,3126],  [2876,2282] ]   
    img_alex_d = [ [7,6],   [6,652],    [777,659],    [770,6]     ]   
    image_points = [img_1_a, img_2_b, img_3_c, img_alex_d]
    
    # Task1(image_points)
    
    # Task 2 image points: 
    img_1_a = [    [437,862 ],  [698,3150], [2391, 2268],  [2530, 901] ]   
    img_2_b = [    [526,1448],  [490,2681], [1900,2760 ],  [1831,833 ]  ]   
    img_3_c = [    [1190,575],  [278,1794], [1791,3126 ],  [2876,2282] ]   
    img_liam = [   [7,6     ],  [6,652   ], [777,659   ],  [770,6    ]     ]   
    image_points = [img_1_a, img_2_b, img_3_c, img_liam]
    
    Task2(image_points)
