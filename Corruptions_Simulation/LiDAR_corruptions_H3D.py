from operator import index
# import open3d as o3d
# import os
# import h5py
import json
import numpy as np
# import struct
# import math
# from numpy import random
# import distortion


# Weather Corruptions

'''
Rain
'''

def rain_sim(pointcloud,severity):
    from utils import lisa
    rain_sim = lisa.LISA(show_progressbar=True)
    c = [0.20, 0.73, 1.5625, 3.125, 7.29, 10.42][severity-1]
    points = rain_sim.augment(pointcloud, c)
    return points

'''
Snow
'''
def snow_sim(pointcloud,severity):
    from utils import lisa
    # from utils.wet_ground.augmentation import ground_water_augmentation
    snow_sim = lisa.LISA(mode='gunn', show_progressbar=True) # snow sim from lisa
    c = [0.20, 0.73, 1.5625, 3.125, 7.29, 10.42][severity-1]
    points = snow_sim.augment(pointcloud, c)
    return points

'''
Snow for nus
'''
def snow_sim_nus(pointcloud,severity):
    '''
    first: git clone git@github.com:SysCV/LiDAR_snow_sim.git --recursive
    second: wget https://www.trace.ethz.ch/publications/2022/lidar_snow_simulation/snowflakes.zip
    then: change the utils in wet ground
    '''
    import sys
    sys.path.append('./LiDAR_snow_sim/')
    from utils.snowfall.simulation import augment
    from utils.wet_ground.augmentation import ground_water_augmentation
    # from tools.snowfall.sampling import snowfall_rate_to_rainfall_rate, compute_occupancy
    # snow sim from https://github.com/SysCV/LiDAR_snow_sim
    c = [0.20, 0.73, 1.5625, 3.125, 7.29, 10.42][severity-1]
    snowflake_file_prefix = ['gunn_2.621627143512277_7.716049382716048e-07','gunn_42.730958596843955_4.9603174603174595e-06','gunn_200.20719573938692_1.3888888888888886e-05','gunn_367.80410429625914_2.083333333333333e-05','gunn_791.3884281145265_3.4722222222222215e-05'][severity-1]
    _, pc = augment(pc=pointcloud, only_camera_fov=False,
                                particle_file_prefix=snowflake_file_prefix, noise_floor=0.7,
                                beam_divergence=float(np.degrees(0.003)),
                                shuffle=True, show_progressbar=False)
    points = ground_water_augmentation(pc)
    return points


'''
Fog
'''
def fog_sim(pointcloud,severity):
    from utils.fog_sim import simulate_fog
    from utils.fog_sim import ParameterSet
    c = [0.005, 0.01, 0.02, 0.03, 0.06][severity-1] # form original paper
    parameter_set = ParameterSet(alpha=c, gamma=0.000001)
    points, _, _ = simulate_fog(parameter_set, pointcloud, 1)
    return points

'''
Sunlight
'''
def scene_glare_noise(pointcloud,label, ignorelabel=11,severity=5):
    N, C = pointcloud.shape
    N1=N*0.3
    c = [int(0.010*N1), int(0.020*N1),int(0.030*N1),int(0.040*N1), int(0.050*N1)][severity-1]
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.normal(size=(c, C))
    label[index] = ignorelabel
    return pointcloud,label



# Sensor Corruptions


'''
Crosstalk
'''
def lidar_crosstalk_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [int(0.004*N), int(0.008*N),int(0.012*N),int(0.016*N), int(0.020*N)][severity-1]
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.normal(size=(c, C)) * 3.0
    return pointcloud


'''
Density
'''
def density_dec_global(pointcloud,label, severity):
    N, C = pointcloud.shape
    num = int(N * 0.91) #0.3
    c = [int(0.2*num), int(0.4*num), int(0.6*num), int(0.8*num), num][severity - 1]
    idx = np.random.choice(N, c, replace=False)
    pointcloud = np.delete(pointcloud, idx, axis=0)
    label=np.delete(label, idx, axis=0)
    return pointcloud,label

# '''
# density_dec
# '''
# def density_dec_local(pointcloud, severity):
#     N, C = pointcloud.shape
#     num = int(N * 0.10)
#     c = [(1, num), (2, num), (3, num), (4, num), (5, num)][severity - 1]
#     for _ in range(c[0]):
#         i = np.random.choice(pointcloud.shape[0],1)
#         picked = pointcloud[i]
#         dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
#         idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
#         idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
#         idx = idx[idx_2]
#         pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
#     return pointcloud

'''
Cutout
'''
def cutout_local(pointcloud,label, severity):
    N, C = pointcloud.shape
    num = int(N*0.01) #0.02
    # c = [(2,num), (3,num), (5,num), (7,num), (10,num)][severity-1]
    c = [(10,num), (15,num), (25,num), (35,num), (50,num)][severity-1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
        label=np.delete(label, idx.squeeze(), axis=0)
    return pointcloud,label


'''
Gaussian (L)
'''
def gaussian_noise(pointcloud,label, severity):
    N, C = pointcloud.shape # N*3
    c = [0.02, 0.04, 0.06, 0.08, 0.10][severity-1]
    jitter = np.random.normal(size=(N, C)) * c*0.6
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc,label

'''
Uniform (L)
'''
def uniform_noise(pointcloud, label,severity):
    # TODO
    N, C = pointcloud.shape
    c = [0.02, 0.04, 0.06, 0.08, 0.10][severity - 1]
    jitter = np.random.uniform(-c, c, (N, C))*1.4
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc,label

'''
wet_ground
'''

def wet_ground(pointcloud,label,severity,datesetName):
    from utils.wet_ground_Robo3D.augmentation import ground_water_augmentation

    augmented_pointcloud, augmented_label, drop_ratio = ground_water_augmentation(pointcloud, label,
                                                                                  water_height=0.0002,
                                                                                  noise_floor=0.3,datesetName=datesetName)

    return augmented_pointcloud,augmented_label



'''
3D noise
'''
def Spacenoise(point_cloud,label, dataclassname,levels=5,ignorelabel=11):
    #噪声点数，用于控制噪声水平
    if dataclassname=='isprs':
        noisepoint=50 #100
        ignorelabel = 9
    if dataclassname=='h3d':
        noisepoint=1000 #100
        ignorelabel = 11

    grid_size=10
    # Extract x, y, z, intensity, label from point cloud
    x, y, z, intensity = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], point_cloud[:,
                                                                                         3]

    # Define the grid boundaries based on the input point cloud
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    # Calculate the number of grid cells along each dimension
    num_x_grids = int((x_max - x_min) / grid_size) + 1
    num_y_grids = int((y_max - y_min) / grid_size) + 1
    num_z_grids = int((z_max - z_min) / grid_size) + 1

    # Create a dictionary to hold points within each grid
    grid_dict = {}

    # Populate the dictionary with points belonging to each grid
    for i in range(len(point_cloud)):
        x_idx = int((x[i] - x_min) / grid_size)
        y_idx = int((y[i] - y_min) / grid_size)
        z_idx = int((z[i] - z_min) / grid_size)

        grid_key = (x_idx, y_idx, z_idx)

        if grid_key not in grid_dict:
            grid_dict[grid_key] = []

        grid_dict[grid_key].append(point_cloud[i])

    # List to store the noisy points
    noisy_points = []

    # Add noise to the point cloud
    for grid_key in grid_dict:
        # Get points in the current grid
        points_in_grid = np.array(grid_dict[grid_key])

        # Determine the number of noise points to add based on noise severity
        noise_severity = np.random.randint(1, levels + 1)
        # num_noise_points = len(points_in_grid) * noise_severity // 10
        num_noise_points = noisepoint * noise_severity // 10

        # Randomly generate noise points within the grid's 3D boundaries
        x_noise = np.random.uniform(grid_key[0] * grid_size + x_min, (grid_key[0] + 1) * grid_size + x_min,
                                    num_noise_points)
        y_noise = np.random.uniform(grid_key[1] * grid_size + y_min, (grid_key[1] + 1) * grid_size + y_min,
                                    num_noise_points)
        z_noise = np.random.uniform(grid_key[2] * grid_size + z_min, (grid_key[2] + 1) * grid_size + z_min,
                                    num_noise_points)

        # Randomly select intensity values from the original intensity range
        intensity_noise = np.random.choice(intensity, num_noise_points)

        # Create noise points with label -1
        for j in range(num_noise_points):
            noisy_point = [x_noise[j], y_noise[j], z_noise[j], intensity_noise[j], ignorelabel]
            noisy_points.append(noisy_point)

    # Combine original point cloud with the noisy points
    noisy_points = np.array(noisy_points)

    augmented_point_cloud = np.vstack((point_cloud, noisy_points[:,:4]))
    augmente_label=np.concatenate((label,noisy_points[:,4]))
    augmente_label=augmente_label.astype(np.int32)

    return augmented_point_cloud,augmente_label



'''
Impulse (L)
'''

def impulse_noise(pointcloud,label, severity):
    N, C = pointcloud.shape
    N1=int(N*0.7)
    c = [N1 // 30, N1 // 25, N1 // 20, N1 // 15, N1 // 10][severity - 1]
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.choice([-1, 1], size=(c, C)) * 0.5
    return pointcloud,label

'''
Fov lost
'''

def fov_filter(points, severity):

    angle1 = [-105, -90, -75, -60, -45][severity-1]
    angle2 = [105, 90, 75, 60, 45][severity-1]
    if isinstance(points, np.ndarray):
        pts_npy = points
    elif isinstance(points, BasePoints):
        pts_npy = points.tensor.numpy()
    else:
        raise NotImplementedError
    pts_p = (np.arctan(pts_npy[:, 0] / pts_npy[:, 1]) + (
                pts_npy[:, 1] < 0) * np.pi + np.pi * 2) % (np.pi * 2)
    pts_p[pts_p > np.pi] -= np.pi * 2
    pts_p = pts_p / np.pi * 180
    assert np.all(-180 <= pts_p) and np.all(pts_p <= 180)
    filt = np.logical_and(pts_p >= angle1, pts_p <= angle2)

    return points[filt]



# Motion corruptions

'''
Moving Obj. 
'''
def moving_noise_bbox(pointcloud,severity,bbox):
    cor = 'move_bbox'
    from utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Motion Compensation
'''
def fulltrajectory_noise(pointcloud, pc_pose, severity):
    from utils.lidar_split import lidar_split, reconstruct_pc
    ct = [0.02, 0.04, 0.06, 0.08, 0.10][severity-1]
    cr = [0.002, 0.004, 0.006, 0.008, 0.010][severity-1]
    new_pose_list, new_lidar_list = lidar_split(pointcloud, pc_pose)
    r_noise = np.random.normal(size=(100, 3, 3)) * cr
    t_noise = np.random.normal(size=(100, 3)) * ct
    new_pose_list[:, :3, :3] += r_noise
    new_pose_list[:, :3, 3] += t_noise
    f_pc = reconstruct_pc(new_lidar_list, new_pose_list)
    return f_pc




# Object corruptions

'''
Local Density
'''

def density_dec_bbox(pointcloud,severity,bbox):
    cor = 'density_dec_bbox'
    from utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Local Cutout
'''
def cutout_bbox(pointcloud,severity,bbox):
    cor = 'cutout_bbox'
    from utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud


'''
Local Gaussian
'''
def gaussian_noise_bbox(pointcloud,severity,bbox):
    cor = 'gaussian_noise_bbox'
    from utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Local Uniform
'''

def uniform_noise_bbox(pointcloud,severity,bbox):
    cor = 'uniform_noise_bbox'
    from utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Local Impulse
'''

def impulse_noise_bbox(pointcloud,severity,bbox):
    cor = 'impulse_noise_bbox'
    from utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Scale
'''
def scale_bbox(pointcloud,severity,bbox):
    cor = 'scale_bbox'
    from utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Shear
'''
def shear_bbox(pointcloud,severity,bbox):
    cor = 'shear_bbox'
    from utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud

'''
Rotation
'''
def rotation_bbox(pointcloud,severity,bbox):
    cor = 'rotation_bbox'
    from utils import bbox_util
    pointcloud = bbox_util.pick_bbox(cor,severity,bbox,pointcloud)
    return pointcloud


# Alignment

'''
Spatial
'''

def spatial_alignment_noise(ori_pose, severity):
    '''
    input: ori_pose 4*4
    output: noise_pose 4*4
    '''
    ct = [0.02, 0.04, 0.06, 0.08, 0.10][severity-1]*2
    cr = [0.002, 0.004, 0.006, 0.008, 0.010][severity-1]*2
    r_noise = np.random.normal(size=(3, 3)) * cr
    t_noise = np.random.normal(size=(3)) * ct
    ori_pose[:3, :3] += r_noise
    ori_pose[:3, 3] += t_noise
    return ori_pose


'''
Temporal
'''
def temporal_alignment_noise(severity):
    frame = [2, 4, 6, 8, 10][severity-1]
    return frame









