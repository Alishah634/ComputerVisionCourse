import numpy as np
import cv2
from typing import List
from skimage.draw import polygon

def conv_to_numpy(conv: List[List[np.ndarray]])-> List[List[np.ndarray]]:
    for points in conv:
        P, Q, R, S = points
        points[0] = np.array(P).flatten()
        points[1] = np.array(Q).flatten()
        points[2] = np.array(R).flatten()
        points[3] = np.array(S).flatten()
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
    
'''THIS NEEDS TO BE CHANGED NOT MINE!!!'''
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
    
'''THIS NEEDS TO BE CHANGED NOT MINE!!!'''
def apply_transform(dest_img, source_img, H, roi_map=None):
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

def Task1(image_points: List[List[np.ndarray]]):
    image_points = conv_to_numpy(image_points.copy())

    img1 = cv2.imread('img1.jpg')
    img2 = cv2.imread('img2.jpg')
    img3 = cv2.imread('img3.jpg')
    alex_image = cv2.imread('alex_honnold.jpg')

    H_da = compute_homography(image_points[-1], image_points[0])
    H_db = compute_homography(image_points[-1], image_points[1]) 
    H_dc = compute_homography(image_points[-1], image_points[2])
    print(f"Calculated Homography\n")

    img_1_roi = get_roi_map(image_points[0], img1, 'img_1_a', 'alex_image')
    img_2_roi = get_roi_map(image_points[1], img2, 'img_2_b', 'alex_image')
    img_3_roi = get_roi_map(image_points[2], img3, 'img_3_c', 'alex_image')
    print(f"Found ROI\n")

    res_da = apply_transform(img1, alex_image, H_da, img_1_roi)
    res_db = apply_transform(img2, alex_image, H_db, img_2_roi)
    res_dc = apply_transform(img3, alex_image, H_dc, img_3_roi)
    print(f"Applied Transform")
    
    cv2.imwrite('proj_d_to_a.jpg', res_da)
    cv2.imwrite('proj_d_to_b.jpg', res_db)
    cv2.imwrite('proj_d_to_c.jpg', res_dc)
    print(f"Wrote to the file")

if __name__ == '__main__':
    img_1_a = [    (437,862),  (698,3150), (2391, 2268), (2530, 901) ]   
    img_2_b = [    (526,1448), (490,2681), (1900,2760),  (1831,833)  ]   
    img_3_c = [    (1190,575),  (278,1794), (1791,3126),  (2876,2282) ]   
    img_alex_d = [ (7,6),   (6,652),    (777,659),    (770,6)     ]   
    image_points = [img_1_a, img_2_b, img_3_c, img_alex_d]
    
    Task1(image_points)
    print(image_points)
