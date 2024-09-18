import numpy as np
import cv2
from termcolor import cprint
from skimage import io
import matplotlib.pyplot as plt 
from typing import List, Tuple

test = 0
H_STAR = None
def point_display(img, points):
    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1], color='red')
    plt.show() 

def get_range_points(domain_pts: np.ndarray, ratio: int) -> np.ndarray:
    P, Q, R, S = domain_pts
    w= Q[0] - P[0]
    h = int(ratio*w)
    P_range = P
    Q_range = [ Q[0], P[1]   , 1]
    R_range = [ Q[0], h+P[1] , 1]
    S_range = [ P[0], h+P[1] , 1]       
    return np.array([ P_range, Q_range, R_range, S_range ] )

def calc_bounds(H: np.ndarray, points: np.ndarray) -> Tuple[int, int, int, int]:
    x_min, x_max,  y_min, y_max = 0,0,0,0
    for point in points:
        temp = np.dot(H, np.array([point]).T)
        temp = temp.flatten()  # Flatten to make sure it's 1D for indexing
        x_min, x_max = min(x_min, temp[0]), max(x_max, temp[0])
        y_min, y_max = min(y_min, temp[1]), max(y_max, temp[1])
    return int(x_min), int(x_max), int(y_min), int(y_max)

# Util function for printing out a formatted numpy array
def print_matrix(matrix: np.ndarray):
    for row in matrix:
        cprint(" ".join(f"{value:.3f}" for value in row), "white")

'''-----------------------------------------------------Task 1 Point to Point-----------------------------------------------------'''
def calc_point_to_point_homography(src_pts: np.ndarray, dest_pts: np.ndarray) -> np.ndarray:
    A = np.zeros((8, 8))
    b = np.zeros((8,))

    for i in range(4):
        x, y, _= src_pts[i]
        x_prime, y_prime, _ = dest_pts[i]
        A[2 * i] = [x, y, 1, 0, 0, 0, -x * x_prime, -y * x_prime]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * y_prime, -y * y_prime]
        b[2 * i] = x_prime
        b[2 * i + 1] = y_prime

    print_matrix(A)  # DEBUG STATEMENT!!!

    try:
        H_temp = np.dot(np.linalg.inv(A), b)
        H = np.append(H_temp, 1).reshape((3, 3))
    except np.linalg.LinAlgError:
        cprint("Error: Matrix inversion failed. The matrix A might be singular.", "red")
        H = np.eye(3)
    return H

def create_empty_canvas(width: int, height: int) -> np.ndarray:
    limit = 50000  # Limit dimensions to prevent large sizes
    empty_img = np.zeros((min(height, limit), min(width, limit), 3), dtype=np.uint8)
    cprint(f"Created a blank canvas with dimensions: {empty_img.shape}", "green")
    return empty_img

# Function to warp an image back using a homography matrix
def apply_p2p_warping(img: np.ndarray, homography_matrix: np.ndarray) -> np.ndarray:
    # Calculate the corner points of the original image
    img_corners = np.array([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]])
    img_corners_homo = np.append(img_corners, np.ones((4, 1)), axis=1)
    # Apply the inverse homography matrix to the corners to find new coordinates
    transformed_corners = np.linalg.inv(homography_matrix).dot(img_corners_homo.T)
    transformed_corners /= transformed_corners[-1]  # Normalize
    transformed_corners = transformed_corners.astype(int)
    # Find the offset and new dimensions for the resulting image
    x_offset = min(transformed_corners[0])
    y_offset = min(transformed_corners[1])
    new_img_width = max(transformed_corners[0]) - min(transformed_corners[0])
    new_img_height = max(transformed_corners[1]) - min(transformed_corners[1])
    cprint(f"New image dimensions: Width={new_img_width}, Height={new_img_height}", "yellow")
    # Create a blank image with the calculated size
    output_img = create_empty_canvas(new_img_width, new_img_height)
    # Iterate through each pixel in the new image and map it back to the original
    for col in range(output_img.shape[1]):
        for row in range(output_img.shape[0]):
            x_coord = col + x_offset
            y_coord = row + y_offset

            # Apply the homography to map the pixel back to the original image coordinates
            projected_point = homography_matrix.dot([x_coord, y_coord, 1])
            original_x = round(projected_point[0] / projected_point[2])
            original_y = round(projected_point[1] / projected_point[2])

            # Check if the calculated coordinates are within bounds of the original image
            if 0 <= original_x < img.shape[1] and 0 <= original_y < img.shape[0]:
                output_img[row, col] = img[original_y, original_x]

    cprint("Image transformation completed using inverse homography.", "green")
    return output_img

def Task1_p2p(domain_pts: np.ndarray, ratio: int, img, image_name: str, task_number: str):
    print(image_name)
    range_points = get_range_points(domain_pts, ratio)
    distorted_image = np.copy(img)
    H_inverse = np.linalg.inv(calc_point_to_point_homography(domain_pts, range_points))
    result_img = apply_p2p_warping(distorted_image, H_inverse)
    cv2.imwrite(f'Final_Results/Task{task_number}_P2P_{image_name}.jpg', result_img)
'''-----------------------------------------------------END of Task 1 Point to Point-----------------------------------------------------'''

'''-----------------------------------------------------Task 1 Two Step-----------------------------------------------------'''
def calculate_projective_homography(points: np.ndarray) -> np.ndarray:
    # Extract points P, Q, R, S
    P, Q, R, S = points
    # Compute lines between points
    line_pq, line_rs, line_pr, line_qs, = np.cross(P, Q)  , np.cross(R, S)  , np.cross(P, R)  , np.cross(Q, S)  
    # Find the intersection points of the lines and cast to float64
    intersection_pq_rs = np.cross(line_pq, line_rs).astype(np.float64)  
    intersection_pr_qs = np.cross(line_pr, line_qs).astype(np.float64)  
    # Normalize the intersection points (convert from homogeneous to Euclidean coordinates)
    if intersection_pq_rs[2] != 0:  
        intersection_pq_rs /= intersection_pq_rs[2]
    if intersection_pr_qs[2] != 0:  
        intersection_pr_qs /= intersection_pr_qs[2]
    # Compute the vanishing line by taking the cross product of the two intersection points
    vanishing_line = np.cross(intersection_pq_rs, intersection_pr_qs).astype(np.float64)
    # Normalize the vanishing line
    if vanishing_line[2] != 0:  
        vanishing_line /= vanishing_line[2]
    # Construct the projective homography matrix H
    projective_homography = np.eye(3, dtype=np.float64) 
    projective_homography[2] = vanishing_line  # Set the vanishing line in the last row
    return projective_homography

def apply_two_step_warping(H: np.ndarray, x_min: int, y_min: int, img1: np.ndarray, distorted_image: np.ndarray) -> np.ndarray:
    undistorted_image = np.copy(distorted_image)
    for i in range(distorted_image.shape[1]):
        for j in range(distorted_image.shape[0]):
            x, y, w = np.dot(np.linalg.inv(H), np.array([i+x_min, j+y_min, 1]) )
            x_coord = int(np.floor(x/w))
            y_coord = int(np.floor(y/w))
            if y_coord > 0 and y_coord < img1.shape[0] and x_coord > 0 and x_coord < img1.shape[1]: 
                undistorted_image[j,i] = img1[y_coord, x_coord]
    return undistorted_image

def calculate_affine_homography(points: np.ndarray) -> np.ndarray:
    # Extract points for calculation of lines
    P, Q, R, S = points
    # Calculate the lines connecting points
    line_pq = np.cross(P, Q)
    line_pr = np.cross(P, R)
    line_qs = np.cross(Q, S)
    line_rs = np.cross(R, S)
    # Normalize the lines
    line_pq_normalized = line_pq / line_pq[2]
    line_pr_normalized = line_pr / line_pr[2]
    line_qs_normalized = line_qs / line_qs[2]
    line_rs_normalized = line_rs / line_rs[2]
    # Create matrix A using the line equations
    A = np.array([
        [line_pq_normalized[0] * line_pr_normalized[0],
         line_pq_normalized[0] * line_pr_normalized[1] + line_pq_normalized[1] * line_pr_normalized[0]],
        [line_qs_normalized[0] * line_rs_normalized[0],
         line_qs_normalized[0] * line_rs_normalized[1] + line_qs_normalized[1] * line_rs_normalized[0]]
    ])
    # Create vector b using the normalized lines
    b = np.array([
        -line_pq_normalized[1] * line_pr_normalized[1],
        -line_qs_normalized[1] * line_rs_normalized[1]
    ])
    # Solve for s
    try:
        s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        cprint("Matrix A is singular, using pseudo-inverse.", "yellow")
        s = np.linalg.pinv(A) @ b
    # Construct the symmetric matrix S
    S = np.eye(2)  # Initialize a 2x2 identity matrix
    S[0, 0] = s[0]
    S[0, 1] = s[1]
    S[1, 0] = s[1]
    # Decompose S to obtain the affine transformation matrix
    U, D, V = np.linalg.svd(S)
    sqrt_D = np.sqrt(np.diag(D))
    transformation_matrix = U @ sqrt_D @ U.T
    # Construct the 3x3 homography matrix
    affine_homography = np.eye(3)
    affine_homography[0:2, 0:2] = transformation_matrix
    return affine_homography

def Task1_TwoStep(domain_pts: np.ndarray, ratio: int, img, image_name: str, task_number: str):
    range_pts = get_range_points(domain_pts, ratio)
    
    # REMOVING Projective:
    distortion_removed = "projection"
    H_vp_inverse = np.linalg.inv(calculate_projective_homography(domain_pts))
    # H_vp_inverse /= H_vp_inverse[2,2]
    x_min, x_max,  y_min, y_max = calc_bounds(H_vp_inverse, domain_pts)
    distorted_image = np.zeros((y_max-y_min+1, x_max-x_min+1,3),np.uint8)
    result_img = apply_two_step_warping(H_vp_inverse, x_min, y_min, np.copy(img), distorted_image)
    cv2.imwrite(f"Final_Results/Task{task_number}_TwoStep_{distortion_removed}_{image_name}.jpg",result_img)
    cprint("Removed Projectie Distortion", "green")
    
    # REMOVING AFFINE:
    distortion_removed = "affine"
    H_affine =  np.linalg.inv(calculate_affine_homography(domain_pts ))
    H_affine /= H_affine[2,2]
    x_min, x_max,  y_min, y_max = calc_bounds(H_affine, domain_pts)
    distorted_image = np.zeros((y_max-y_min+1, x_max-x_min+1,3),np.uint8)
    result_img = apply_two_step_warping(H_affine, x_min, y_min, np.copy(img), distorted_image)
    cv2.imwrite(f"Final_Results/Task{task_number}_TwoStep_{distortion_removed}_{image_name}.jpg",result_img)
    cprint("Removed Affine Distortion", "green")
    
    # # REMOVING Both Distortions:
    distortion_removed = "both"
    H_overall = H_vp_inverse @ H_affine
    H_overall /= H_overall[2,2]
    x_min, x_max,  y_min, y_max = calc_bounds(H_overall, domain_pts)
    distorted_image = np.zeros((y_max-y_min+1, x_max-x_min+1,3),np.uint8)
    result_img = apply_two_step_warping(H_overall, x_min, y_min, np.copy(img), distorted_image)
    cv2.imwrite(f"Final_Results/Task{task_number}_TwoStep_{distortion_removed}_{image_name}.jpg",result_img)
    cprint("Removed Both Projectie & Affine Distortion", "green")

'''-----------------------------------------------------END of Task 1 Two Step-----------------------------------------------------'''

'''-----------------------------------------------------Task 1 One Step-----------------------------------------------------'''
# Breaks sometimes, need to account for float int type of points
def calc_lines(domain_pts: np.ndarray) -> List[np.ndarray]:
    def line_bw_points(p1, p2):
        p1_coord = np.array([p1[0],p1[1],1])
        p2_coord = np.array([p2[0],p2[1],1])
        l = np.cross(p1_coord, p2_coord).astype(float)
        l /= np.linalg.norm(l)
        return l
    if len(domain_pts) > 4:
        P, Q, R, S, _, _, _ = domain_pts
    else:
        P, Q, R, S = domain_pts
    return  [line_bw_points(P,Q) , line_bw_points(P,S) ,line_bw_points(R,S), line_bw_points(Q,R), line_bw_points(Q,S), line_bw_points(P,R)]

def compute_homography_onestep_affine(pts: np.ndarray) -> np.ndarray:
    # Calculate 5 pairs of orthogonal lines
    l1 = np.cross(pts[0], pts[1])  # PQ
    m1 = np.cross(pts[1], pts[3])  # QS
    
    l2 = np.cross(pts[1], pts[3])  # QS
    m2 = np.cross(pts[3], pts[2])  # SR
    
    l3 = np.cross(pts[3], pts[2])  # SR
    m3 = np.cross(pts[2], pts[0])  # RP
    
    l4 = np.cross(pts[2], pts[0])  # RP
    m4 = np.cross(pts[0], pts[1])  # PQ
    
    l5 = np.cross(pts[0], pts[3])  # PS
    m5 = np.cross(pts[2], pts[1])  # RQ

    # Build the X matrix and y vector for least square estimation
    def build_x(l: np.ndarray, m: np.ndarray) -> np.ndarray:
        return np.array([
            l[0] * m[0],
            (l[1] * m[0] + l[0] * m[1]) / 2,
            l[1] * m[1],
            (l[2] * m[0] + l[0] * m[2]) / 2,
            (l[2] * m[1] + l[1] * m[2]) / 2
        ])

    X1 = build_x(l1, m1)
    X2 = build_x(l2, m2)
    X3 = build_x(l3, m3)
    X4 = build_x(l4, m4)
    X5 = build_x(l5, m5)

    X = np.vstack((X1, X2, X3, X4, X5))
    y = np.array([-m1[2]*l1[2], -m2[2]*l2[2], -m3[2]*l3[2], -m4[2]*l4[2], -m5[2]*l5[2]], dtype=float)

    # Solve for the dual degenerate conic parameters
    param = np.linalg.lstsq(X, y, rcond=None)[0]

    # Construct the image of the dual degenerate conic (dual_degenerate_conic)
    dual_degenerate_conic = np.zeros((3, 3))
    dual_degenerate_conic[0, :] = [param[0], param[1] / 2, param[3] / 2]
    dual_degenerate_conic[1, :] = [param[1] / 2, param[2], param[4] / 2]
    dual_degenerate_conic[2, :] = [param[3] / 2, param[4] / 2, 1]
    dual_degenerate_conic /= np.max(dual_degenerate_conic)

    # Decompose the conic to find A and v for the homography matrix
    S = dual_degenerate_conic[:2, :2]
    U, D, Vt = np.linalg.svd(S)
    A = U @ np.sqrt(np.diag(D)) @ Vt
    v = np.linalg.inv(A.T) @ dual_degenerate_conic[2, :2]

    # Construct the homography matrix H
    H = np.eye(3)
    H[:2, :2] = A
    H[2, :2] = v

    return H

# Function to warp an image back using a homography matrix and apply correct scaling
def apply_p2p_warping_with_scaling(img: np.ndarray, homography_matrix: np.ndarray) -> np.ndarray:
    # Calculate the corner points of the original image
    img_corners = np.array([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]])
    img_corners_homo = np.append(img_corners, np.ones((4, 1)), axis=1)
    # Apply the inverse homography matrix to the corners to find new coordinates
    transformed_corners = np.linalg.inv(homography_matrix).dot(img_corners_homo.T)
    transformed_corners /= transformed_corners[-1]  # Normalize to homogeneous coordinates
    transformed_corners = transformed_corners.astype(int)
    # Find the offset and new dimensions for the resulting image
    x_offset = min(transformed_corners[0])
    y_offset = min(transformed_corners[1])
    new_img_width = max(transformed_corners[0]) - min(transformed_corners[0])
    new_img_height = max(transformed_corners[1]) - min(transformed_corners[1])
    # Ensure scaling is correct by keeping the aspect ratio
    scaling_factor_x = img.shape[1] / new_img_width
    scaling_factor_y = img.shape[0] / new_img_height
    scaling_factor = min(scaling_factor_x, scaling_factor_y)
    new_img_width = int(new_img_width * scaling_factor)
    new_img_height = int(new_img_height * scaling_factor)
    # Create a blank image with the calculated size
    output_img = np.zeros((new_img_height, new_img_width, 3), dtype=np.uint8)
    # Iterate through each pixel in the new image and map it back to the original
    for col in range(output_img.shape[1]):
        for row in range(output_img.shape[0]):
            # Apply scaling factor to keep the aspect ratio
            x_coord = (col + x_offset) / scaling_factor
            y_coord = (row + y_offset) / scaling_factor
            # Apply the homography to map the pixel back to the original image coordinates
            projected_point = homography_matrix.dot([x_coord, y_coord, 1])
            original_x = round(projected_point[0] / projected_point[2])
            original_y = round(projected_point[1] / projected_point[2])
            # Check if the calculated coordinates are within bounds of the original image
            if 0 <= original_x < img.shape[1] and 0 <= original_y < img.shape[0]:
                output_img[row, col] = img[original_y, original_x]
    return output_img

def Task1_OneStep(domain_pts: np.ndarray, ratio: int, img, image_name: str, task_number: str):
    H_one = compute_homography_onestep_affine(domain_pts)
    H_one_inv = np.linalg.inv(H_one)
    distorted = np.copy(img)
    # if image_name == "corridor_1":
    #     H_one_inv = np.linalg.inv(calculate_affine_homography(domain_pts[0:4]))
    result_img = apply_p2p_warping_with_scaling(np.copy(img), H_one_inv)
    cv2.imwrite(f"Final_Results/TEST_Task{task_number}_OneStep_{image_name}.jpg",result_img)
'''-----------------------------------------------------END of Task 1 One Step-----------------------------------------------------'''

# Main execution block
if __name__ == "__main__":
    ############################################## TASK 1 P2P ##############################################
    
    ############################################## Declaration Variables ##############################################
    # Load the distorted board image
    img_path = 'HW3_images/board_1.jpeg'
    img = io.imread(img_path)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    board_image_points = np.array([[72, 411, 1], [1220, 141, 1], [1350, 1955, 1], [419, 1755, 1]])
    ratio = 12/8
    # Task1_p2p(board_image_points, ratio, img_bgr, "board_1", 1) 

    img_path = 'HW3_images/corridor.jpeg'
    img = io.imread(img_path)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    corridor_image_points = np.array([[1087, 534, 1 ], [1307, 493, 1 ], [1298, 1344, 1 ], [1081, 1223, 1 ]])

    ratio = 6/3
    # Task1_p2p(corridor_image_points, ratio, img_bgr, "corridor_1", 1)    
    cprint("Transformation complete and saved as task1_p2p.jpg.","green")
    ############################################## End of TASK 1 P2P ##############################################

    ############################################## TASK 1 Two - Step ##############################################
    # Load the distorted board image
    img_path = 'HW3_images/board_1.jpeg'
    img = io.imread(img_path)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    board_image_points = np.array([[72, 411, 1], [1220, 141, 1], [1350, 1955, 1], [419, 1755, 1]])
    ratio = 12/8
    # Task1_TwoStep(board_image_points, ratio, img_bgr, "board_1", 1) 

    img_path = 'HW3_images/corridor.jpeg'
    img = io.imread(img_path)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    corridor_image_points = np.array([[1087, 534, 1 ], [1307, 493, 1 ], [1298, 1344, 1 ], [1081, 1223, 1 ]])

    ratio = 6/3
    # Task1_TwoStep(corridor_image_points, ratio, img_bgr, "corridor_1", 1)    
    cprint("Transformation complete and saved as task1_TwoStep.jpg.","green")
    ############################################## End of TASK 1 Two - Step ##############################################
    
    ############################################# TASK 1 One - Step ##############################################
    # Load the distorted board image
    img_path = 'HW3_images/board_1.jpeg'
    img = io.imread(img_path)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    board_image_points = np.array([[72, 411, 1], [1220, 141, 1], [1350, 1955, 1], [419, 1755, 1], \
        [878, 539, 1], [897, 653, 1], [601, 693, 1], \
        [852, 305, 1],  [865, 423, 1], [761, 444, 1]
        ])
    ratio = 12/8
    # Task1_OneStep(board_image_points, ratio, img_bgr, "board_1", 1) 

    img_path = 'HW3_images/corridor.jpeg'
    img = io.imread(img_path)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    corridor_image_points = np.array([[1087, 534, 1 ], [1307, 493, 1 ], [1298, 1344, 1 ], [1081, 1223, 1 ], \
        [1122, 856, 1], [1119, 1207, 1], [1205, 1252, 1], \
            [1117, 518, 1], [1183, 1268, 1], [1093, 1204, 1]
        ])
    ratio = 6/3

    point_display(img, corridor_image_points)

    Task1_OneStep(corridor_image_points, ratio, img_bgr, "corridor_1", 1)    
    cprint("Transformation complete and saved as task1_OneStep.jpg.","green")
    ############################################# End of TASK 1 One - Step ##############################################
    
    ############################################# TASK 2 MY Images: ##############################################
    cprint("Testing My Images:", "white")
    # My first image info:
    img_path1 = 'MyImages/keyboard.jpeg'
    img1 = io.imread(img_path1)
    img_bgr1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    ratio1 = 0.08333333*(13/6) # These are in inches being converted to feet
    
    keyboard_points =  np.array([[1029, 67, 1], [1090, 1394, 1], [520, 1422, 1], [424, 132, 1]])
    point_display(img1, keyboard_points)
    
    # Run for each task:
    # Task1_p2p(keyboard_points, ratio1, img_bgr1, "keyboard", 2)    
    # Task1_TwoStep(keyboard_points, ratio1, img_bgr1, "keyboard", 2) 
    # Task1_OneStep(keyboard_points, ratio1, img_bgr1, "keyboard", 2)    

    # keyboard_points =  np.array([1029, 67, 1], [1090, 1394, 1], [520, 1422, 1], [424, 132, 1])
    cprint("Successfuly completed run for keyboard image!", "green")


    
    # _=_=__=__=_=_=__=_=_=_=_=_=_=_=_=_=_=_=_=_=__=_=_
    # My second image
    img_path2 = 'MyImages/door.jpeg'
    img2 = io.imread(img_path2)
    img_bgr2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    ratio2 = 0.08333333*(25/9)
    
    door_points = np.array([[813, 465, 1], [1019, 466, 1], [1001, 987, 1], [838, 992, 1]])    
    point_display(img2, door_points)
    
    
    
    
    Task1_p2p(door_points, ratio2, img_bgr2, "door", 2)    
    Task1_TwoStep(door_points, ratio2, img_bgr2, "door", 2) 
    Task1_OneStep(door_points, ratio2, img_bgr2, "door", 2)
    
    cprint("Successfuly completed run for door image!", "green")
    
    # cprint("Transformation complete and saved as Task2_OneStep.jpg.","green")
    
    ############################################# TASK 2: ##############################################