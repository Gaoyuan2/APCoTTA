import sys,os
from os import mkdir
from os.path import exists

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper_ply import *
import numpy as np



if __name__=='__main__':
    cloud=read_ply('data/Mar18_val.ply')

    # first, make sure the point cloud in a numpy array format, like N*4 or N*5
    cloud = np.vstack((cloud['x'], cloud['y'], cloud['z'], cloud['scalar_Reflectance'], cloud['scalar_Classification'])).T

    limitMin = np.amin(cloud[:, 0:3], axis=0)
    cloud[:, 0:3] -= limitMin
    # cloud[:, 0:3] -= np.array([110,233,15])
    # cloud = self.changeSemLabels(cloud)

    xyz = cloud[:, :3].astype(np.float32)
    colors = cloud[:, 3].astype(np.float32)
    labels = cloud[:, 4].astype(np.int32)

    if not exists('originalOOOOsh3d.ply'):
        write_ply('originalOOOOsh3d.ply', (xyz, colors, labels), ['x', 'y', 'z', 'scalar_Reflectance', 'class'])


    print(cloud.shape)


    # weather-level
    from LiDAR_corruptions_H3D import (gaussian_noise, scene_glare_noise, Spacenoise, uniform_noise,
                                       density_dec_global, cutout_local, impulse_noise)
    severity=5

   

    # noisetype='Spacenoise'
    # lidar_cor, labelsff = Spacenoise(cloud[:, :4], cloud[:, 4], ignorelabel=11,dataclassname='h3d',levels=severity)

    # noisetype='scene_glare_noise'
    # lidar_cor, labelsff = scene_glare_noise(cloud[:, :4], cloud[:, 4], ignorelabel=11,severity=severity)


    #other
    noisetype='gaussian_noise'
    lidar_cor, labelsff = gaussian_noise(cloud[:, :4], cloud[:, 4], severity)
    labelsff = labelsff.astype(np.int32)

    print(noisetype)
    print('运行结果:',lidar_cor.shape)
    print('运行成功!')

    points=lidar_cor[:, :3].astype(np.float32)
    feat=lidar_cor[:, 3].astype(np.float32)
    # labels = lidar_cor[:, 4].astype(np.int32)
    # intensitydiff = lidar_cor[:, 5].astype(np.int32)

    # labels = np.array(labels, dtype=np.int32).reshape((-1,))

    savepath=r'/data/ALS_CTTA_KUA/NewH3DCTTA'
    if not exists(savepath):
        mkdir(savepath)
    savefold=os.path.join(savepath,noisetype)
    if not exists(savefold):
        mkdir(savefold)
    # pathfile = cloud_file.replace('.las', '.ply')
    write_ply(os.path.join(savefold,noisetype+'_'+str(severity)+'.ply'), (points, feat,labelsff), ['x', 'y', 'z', 'Reflectance','class'])

    # write_ply('impulse_noise.ply', (points, feat), ['x', 'y', 'z', 'intensity'])

    # xyz = lidar_cor[:, :4].astype(np.float32)
    # colors = cloud[:, 4:6].astype(np.float32)
    # labels = cloud[:, 6].astype(np.int32)
    # write_ply('Vaihingen3D_EVAL_WITH_REFTEst.ply', (xyz, colors, labels), ['x', 'y', 'z', 'Intensity', 'return_number',
    #                                               'number_of_returns', 'class'])