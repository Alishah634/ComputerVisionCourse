import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import h5py # for reading depth maps

"""
A few notes on the scene_info dictionary:
- depth maps are stored as h5 files. Depth is the distance of the object from the camera (ie Z coordinate in camera coordinates). The depth map can contain invalid points (depth = 0) which correspond to points where the depth could not be estimated.
- The intrinsics are stored as a 3x3 matrix.
- The poses [R,t] are stored as a 4x4 matrix to allow for easy transformation of points from one camera to the other. The resulting transformation matrix is a 4x4 matrix is of the form:
    T = [[R, t]
        [0, 1]] where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
"""

DEPTH_THR = 0.1

def plot_image_and_depth(img0, depth0, img1, depth1, plot_name):
    # Enable constrained layout for uniform subplot sizes
    fig, ax = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

    # Image 0
    ax[0].imshow(img0, aspect='auto')
    ax[0].set_title('Image 0')
    ax[0].axis('off')

    # Depth 0
    im1 = ax[1].imshow(depth0, cmap='jet', aspect='auto')
    ax[1].set_title('Depth 0')
    ax[1].axis('off')
    cbar1 = fig.colorbar(im1, ax=ax[1], shrink=0.8, aspect=20)
    cbar1.ax.yaxis.set_ticks_position('left')
    cbar1.ax.yaxis.set_label_position('left')
    cbar1.ax.tick_params(labelsize=15)

    # Image 1
    ax[2].imshow(img1, aspect='auto')
    ax[2].set_title('Image 1')
    ax[2].axis('off')

    # Depth 1
    im2 = ax[3].imshow(depth1, cmap='jet', aspect='auto')
    ax[3].set_title('Depth 1')
    ax[3].axis('off')
    cbar2 = fig.colorbar(im2, ax=ax[3], shrink=0.8, aspect=20)
    cbar2.ax.yaxis.set_ticks_position('left')
    cbar2.ax.yaxis.set_label_position('left')
    cbar2.ax.tick_params(labelsize=15)

    plt.savefig(plot_name, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    scene_info = pkl.load(open('./data/scene_info/1589_subset.pkl', 'rb'))

    for i_pair in range(len(scene_info)):
        # print(scene_info[i_pair].keys())
        # ['image0','image1','depth0', 'depth1', 'K0', 'K1', 'T0', 'T1', 'overlap_score']
        # print(scene_info[i_pair]['image0']) # path to image0
        # print(scene_info[i_pair]['image1']) # path to image1
        # print(scene_info[i_pair]['depth0'])  # path to depth0
        # print(scene_info[i_pair]['depth1'])  # path to depth1
        # print(scene_info[i_pair]['K0'])  # intrinsic matrix of camera 0 [3,3]
        # print(scene_info[i_pair]['K1'])  # intrinsic matrix of camera 1 [3,3]
        # print(scene_info[i_pair]['T0'])  # pose matrix of camera 0 [4,4]
        # print(scene_info[i_pair]['T1'])  # pose matrix of camera 1 [4,4]
        # print('-------------------')

        # read images
        img0 = plt.imread(scene_info[i_pair]['image0'])
        img1 = plt.imread(scene_info[i_pair]['image1'])

        # read depth
        with h5py.File(scene_info[i_pair]['depth0'], 'r') as f:
            depth0 = f['depth'][:]
        with h5py.File(scene_info[i_pair]['depth1'], 'r') as f:
            depth1 = f['depth'][:]

        # check shapes
        h0, w0 = img0.shape[:-1]
        h1, w1 = img1.shape[:-1]
        assert img0.shape[:-1] ==  depth0.shape, f"depth and image shapes do not match: {img0}, {depth0}"
        assert img1.shape[:-1] ==  depth1.shape, f"depth and image shapes do not match: {img1}, {depth1}"

        # plot image and depth            
        plot_name = f'./pics/image_and_depth_pair_{i_pair}.png'
        plot_image_and_depth(img0, depth0, img1, depth1, plot_name)      

        #(1) make meshgrid of points in image 0
        x = np.linspace(10, img0.shape[1]-10, 100) # ignore a border of 10 pxls    
        y = np.linspace(10, img0.shape[0]-10, 100)
        """
        <Student code>
        # meshgrid of x and y coordinates
        xx, yy = ...
        """
        xx, yy = np.meshgrid(x, y)
        # make homogeneous coordinates for points0 #[3, N]
        """
        <Student code>
        points0 = ...
        """
        points0 = np.vstack((xx.flatten(), yy.flatten(), np.ones_like(xx.flatten())))

        #(2) get depth values at points0
        """
        <Student code>
        depth_values0 = ...
        """
        depth_values0 = depth0[yy.flatten().astype(int), xx.flatten().astype(int)]
        print(points0.shape)
        print(depth0.shape)
        print(depth_values0.shape)
        # remove points with depth 0 (invalid points)
        """
        <Student code>
        valid_points = ...
        # mask points0 and depth_values0
        """
        
        # Where is the mask is False?:
        indices = np.where( (np.abs(depth_values0) > 0.1) )[0]
        # Filter out invalid points and corresponding depth values:
        points0 = points0[:, indices]
        depth_values0 = depth_values0[indices]


        # (3) Find the 3D coordinates of these points in camera 0 frame
        K0 = scene_info[i_pair]['K0'] # [3,3]
        T0 = scene_info[i_pair]['T0'] # [4,4]
        """
        <Student code>
        # inverse of K0
        K0_inv = ...
        # convert points0 to camera coordinates
        xyz_cam0 = ...
        # normalize xyz_cam0 to set z = 1 (sanity check)
        xyz_cam0 = ...
        # get the point at depth
        xyz_cam0 = ...
        # make homogeneous coordinates [4,N]
        xyz_cam0_hc = ...
        # convert to world frame [4,N]
        xyz_world_hc = ...
        """

        K0_inv = np.linalg.inv(K0)
        xyz_cam0 = np.dot(K0_inv, points0)
        xyz_cam0 = xyz_cam0 / xyz_cam0[2, :]
        xyz_cam0 = xyz_cam0 * depth_values0
        xyz_cam0_hc = np.vstack((xyz_cam0, np.ones((1, xyz_cam0.shape[1]))))
        xyz_world_hc = np.dot(np.linalg.inv(T0), xyz_cam0_hc)
        
        # (4) Transform these points to camera 1 frame
        T1 = scene_info[i_pair]['T1']
        """
        <Student code>
        # transform points to camera 1 frame
        xyz_cam1_hc = ...
        # convert to camera 1 coordinates
        xyz_cam1 = ...
        # get z coordinates for depth check
        estimated_depth_values1 = ...
        """
        xyz_cam1_hc = T1 @ xyz_world_hc
        xyz_cam1 = xyz_cam1_hc[:3,:] / xyz_cam1_hc[3,:]
        estimated_depth_values1 = xyz_cam1[2,:]

        # project to image 1
        """
        <Student code>
        points1 = ... # [3, N]
        # normalize by dividing by last row
        points1 = ... # [3, N]
        # check if points1 are within image bounds
        ...
        # get the depth values at these points using the depth map
        true_depth_values1 = ...
        """
        
        K1 = scene_info[i_pair]['K1']
        points1 = np.dot(K1, xyz_cam1)
        points1 = points1 / points1[2, :]
        within_image_bounds_mask = [(x >= 0 and x < w1 and y >= 0 and y < h1) for x, y in zip(points1[0, :], points1[1, :])]
        points1 = points1[:, within_image_bounds_mask]
        estimated_depth_values1 = estimated_depth_values1[within_image_bounds_mask]
        true_depth_values1 = [depth1[int(y), int(x)] for x, y in zip(points1[0, :], points1[1, :])]

        
        
        # (5) plot matching points in image 0 and image 1 with depth check such that the depth values match
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # Horizontally stack the images
        combined_img = np.ones((max(img0.shape[0], img1.shape[0]), img0.shape[1] + img1.shape[1], 3), dtype=np.uint8) * 255
        combined_img[:img0.shape[0], :img0.shape[1]] = img0
        combined_img[:img1.shape[0], img0.shape[1]:] = img1

        ax.imshow(combined_img, aspect='auto')
        ax.scatter(xx, yy, c='r', s=5)
        ax.set_title('Matching points in Image 0 and Image 1')
        ax.axis('off')

        # draw lines between matching points
        for i in range(points1.shape[1]):
            # if depth values match
            if np.abs(estimated_depth_values1[i] - true_depth_values1[i]) < DEPTH_THR and true_depth_values1[i] != 0:
                ax.plot([points0[0,i], points1[0, i] + img0.shape[1]], [points0[1,i], points1[1, i]], 'g')        

        plt.savefig(f'./pics/DEPTH_CORRES/DEPTH_CORRES_{i_pair}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # (6) Plot all 3D points for the pair
        """
        <Student code>
        ...
        """
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(xyz_world_hc[0, :], xyz_world_hc[1, :], xyz_world_hc[2, :], c='r')
        ax.set_title('3D points in world frame')
        ax.set_xlabel('X (World)')
        ax.set_ylabel('Y (World)')
        ax.set_zlabel('Z (World)')
        ax.grid(True)
        plt.savefig(f'./pics/3D_Points/3D_Points_{i_pair}.jpg', bbox_inches='tight', pad_inches=0)
        
        
# My driver code for task 4:
def task4_main():
    print("Cannot start Task 4 from root, need to be in depth map directory!!!")
    pass
    
    
