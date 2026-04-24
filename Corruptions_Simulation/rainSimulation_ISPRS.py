import sys,os
from os import mkdir
from os.path import exists

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper_ply import *
import numpy as np



if __name__=='__main__':
    cloud=read_ply('data/Vaihingen3D_EVAL_WITH_REF.ply')

    # first, make sure the point cloud in a numpy array format, like N*4 or N*5
    cloud = np.vstack((cloud['x'], cloud['y'], cloud['z'], cloud['Intensity'], cloud['return_number'],
                                        cloud['number_of_returns'], cloud['class'])).T

    limitMin = np.amin(cloud[:, 0:3], axis=0)
    cloud[:, 0:3] -= limitMin
    cloud[:, 0:3] -= np.array([110,233,15])
    # cloud = self.changeSemLabels(cloud)

    xyz = cloud[:, :3].astype(np.float32)
    colors = cloud[:, 3:6].astype(np.float32)
    labels = cloud[:, 6].astype(np.int32)

    if not exists('originalOOOO_isprs.ply'):
        write_ply('originalOOOO_isprs.ply', (xyz, colors, labels), ['x', 'y', 'z', 'Intensity', 'return_number',
                                                    'number_of_returns', 'class'])


    print(cloud.shape)


    # weather-level
    from LiDAR_corruptions_ISPRS import (gaussian_noise, scene_glare_noise, Spacenoise, uniform_noise,
                                         density_dec_global, cutout_local, impulse_noise)
    severity=5

    # noisetype='Spacenoise'
    # lidar_cor, labelsff = Spacenoise(cloud[:, :4], cloud[:, 6], ignorelabel=9,dataclassname='isprs',levels=severity)

    noisetype='scene_glare_noise'
    lidar_cor, labelsff = scene_glare_noise(cloud[:, :4], cloud[:, 6], ignorelabel=9,severity=severity)

    #other
    # noisetype='impulse_noise'
    # lidar_cor, labelsff = impulse_noise(cloud[:, :4], cloud[:, 6], severity)
    # labelsff = labelsff.astype(np.int32)

    print(noisetype)
    print('运行结果:',lidar_cor.shape)
    print('运行成功!')

    points=lidar_cor[:, :3].astype(np.float32)
    feat=lidar_cor[:, 3].astype(np.float32)
    # labels = lidar_cor[:, 4].astype(np.int32)
    # intensitydiff = lidar_cor[:, 5].astype(np.int32)

    # labels = np.array(labels, dtype=np.int32).reshape((-1,))

    savepath=r'/data/ALS_CTTA_KUA/NewISPRSCTTA_Severity_5'
    if not exists(savepath):
        mkdir(savepath)
    savefold=os.path.join(savepath,noisetype)
    if not exists(savefold):
        mkdir(savefold)
    # pathfile = cloud_file.replace('.las', '.ply')
    write_ply(os.path.join(savefold,noisetype+'_'+str(severity)+'.ply'), (points, feat,labelsff), ['x', 'y', 'z', 'Intensity','class'])

    # write_ply('impulse_noise.ply', (points, feat), ['x', 'y', 'z', 'intensity'])

    # xyz = lidar_cor[:, :4].astype(np.float32)
    # colors = cloud[:, 4:6].astype(np.float32)
    # labels = cloud[:, 6].astype(np.int32)
    # write_ply('Vaihingen3D_EVAL_WITH_REFTEst.ply', (xyz, colors, labels), ['x', 'y', 'z', 'Intensity', 'return_number',
    #                                               'number_of_returns', 'class'])