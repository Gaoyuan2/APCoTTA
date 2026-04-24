#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020 modified by Meida Chen - 04/25/2022
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

import logging
from os.path import exists, join, basename

from tqdm import tqdm

# Dataset
from datasets.Commondataset_CTTA_FuXian import *
from torch.utils.data import DataLoader

from utils.config import Config
from models.architectures_Semi import KPCNN, KPFCNN
from utils.metrics import IoU_from_confusions, fast_confusion, OA, F1_score


import APCoTTA

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def SelectADAPTATION(base_model,optimizer,config,modelName):

    if modelName == "our":
        logger.info("test-time adaptation: CoTTA")
        model = setup_our(base_model,optimizer,config)

   
    return model

def setup_our(model,optimizer,config):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = APCoTTA.configure_model(model)
    params, param_names = APCoTTA.collect_params(model)
    # optimizer = setup_optimizer(params)
    cotta_model = APCoTTA.OUR(model, optimizer, config,
                                          steps=1,
                                          episodic=False)
    # cotta_model.bn_params=bn_params
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model


def Prepare_parameter():
    chosen_log='results//H3D_checkpoint'
    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    # chkp_idx = -1
    chkp_idx = None
    # Choose to test on validation or test split
    on_val = True

    ############################
    # Initialize the environment
    ############################

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    return config,chosen_chkp

def Prepare_model(config,chkp_path,label_values,ignored_labels):

    ##################################
    # Change model parameters for test
    ##################################

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    if config.dataset_task == 'classification':
        net = KPCNN(config)
    elif config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        net = KPFCNN(config, label_values, ignored_labels)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # Choose to train on CPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:%s" %cudaDevice)
    else:
        device = torch.device("cpu")
    net.to(device)
    ##########################
    # Load previous checkpoint
    ##########################

    checkpoint = torch.load(chkp_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    # Optimizer with specific learning rate for deformable KPConv
    deform_params = [v for k, v in net.named_parameters() if 'offset' in k]
    other_params = [v for k, v in net.named_parameters() if 'offset' not in k]


    deform_lr = config.learning_rate * config.deform_lr_factor
    optimizer = torch.optim.SGD([{'params': other_params},
                                      {'params': deform_params, 'lr': deform_lr}],
                                     lr=config.learning_rate,
                                     momentum=config.momentum,
                                     weight_decay=config.weight_decay)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    # self.epoch = checkpoint['epoch']
    net.eval()
    # net2.eval()
    print("Model and training state restored.")
    return net,device,optimizer

def list_subfolders(dir_path):
    subfolders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
    return subfolders

def cloud_segmentation_test(net, test_loader, config,device ,ioupath,testpath, num_votes=10,debug=False):
        """
        Test method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth =0   #0.95
        test_radius_ratio = 0 #0.7
        softmax = torch.nn.Softmax(1)

        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        test_probs = [np.zeros((l.shape[0], nc_model))-1 for l in test_loader.dataset.input_labels]

        # Test saving path
        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1],testpath)
            if not exists(test_path):
                makedirs(test_path)
        else:
            test_path = None

        aYu = np.sum([np.sum(subarray == -1) for subarray in test_probs]) / nc_model / 100
        previous = None  # 记录前一个值
        count = 0  # 计数器

        with torch.no_grad():
            # for i, batch in enumerate(tqdm(test_loader,desc='Testing')):
            for i, batch in enumerate(test_loader):


                if 'cuda' in device.type:
                    batch.to(device)

                # Forward pass
                outputs,*_ = net(batch, config,test_loader)

                # Get probs and labels
                stacked_probs = softmax(outputs).cpu().detach().numpy()
                # s_points = batch.weakbatch['points'][0].cpu().numpy() #输入点坐标(减去中心点)
                lengths =batch.weakbatch['lengths'][0].cpu().numpy()
                in_inds = batch.weakbatch['input_inds'].cpu().numpy() #输入点的索引
                cloud_inds = batch.weakbatch['cloud_inds'].cpu().numpy() #点云文件索引
                torch.cuda.synchronize(device)

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    # points = s_points[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    test_probs[c_i][inds] = test_smooth * test_probs[c_i][inds] + (1 - test_smooth) * probs
                    i0 += length

                a = np.sum([np.sum(subarray == -1) for subarray in test_probs]) / nc_model
                # if i % 5 == 0:
                print('****************', a)
                #

                if previous == a:  # 如果当前值等于前一个值
                    count += 1
                else:
                    count = 1  # 重置计数器
                    previous = a
                if count == 10:  # 连续10次相同
                    print("连续10次相同，退出循环")
                    break
                if aYu > a >= 0:
                    break
        if True:
            Confs = []
            for i, file_path in enumerate(test_loader.dataset.files):

                probs = test_probs[i]

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                    if label_value in test_loader.dataset.ignored_labels:
                        probs = np.insert(probs, l_ind, 0, axis=1)


                points=np.array(test_loader.dataset.input_trees[i].data)
                colors= test_loader.dataset.input_colors[i]
                gtLabels = test_loader.dataset.input_labels[i]
                # Get the predicted labels
                preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                # Save plys
                cloud_name = basename(file_path)
                test_name = join(test_path, cloud_name)
                gtLabels = np.squeeze(gtLabels).astype(np.int32)

                colors=np.squeeze(colors)
                if 'H3D' in config.datasetClass:
                    write_ply(test_name,
                              [points, colors, preds, gtLabels],
                              ['x', 'y', 'z', 'Reflectance', 'class', 'oclass'])
                elif 'ISPRS' in config.datasetClass:
                    write_ply(test_name,
                              [points, colors, preds, gtLabels],
                              ['x', 'y', 'z', 'Intensity', 'class', 'oclass'])

                Confs += [fast_confusion(gtLabels, preds, test_loader.dataset.label_values)]

            C = np.sum(np.stack(Confs), axis=0)

            # Remove ignored labels from confusions
            for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                if label_value in test_loader.dataset.ignored_labels:
                    C = np.delete(C, l_ind, axis=0)
                    C = np.delete(C, l_ind, axis=1)
            # print('***********在原始数据集上的精度**********')
            IoUs = IoU_from_confusions(C)
            mIoU = np.mean(IoUs)
            OAs = OA(C)
            print('OA: ', OAs)
            F1s = np.array(F1_score(C))
            mF1=np.mean(F1s)
            print('F1:  ')
            s = '{:5.2f} | '.format(100 * mF1)
            for F1 in F1s[:-1]:
                s += '{:5.2f} '.format(100 * F1)
            print('-' * len(s))
            print(s)
            print('-' * len(s))

            print('IoUs:')
            s = '{:5.2f} | '.format(100 * mIoU)
            for IoU in IoUs[:-1]:
                s += '{:5.2f} '.format(100 * IoU)
            print('-' * len(s))
            print(s)
            print('-' * len(s) + '\n')

        if not os.path.exists(ioupath):
            with open(ioupath, 'w') as f:
                f.write('Data_Name: '+test_loader.dataset.path+'\n')
                f.write('OA: '+str(OAs)+'\n')
                f.write('mIoU: ' + str(mIoU)+'\n')
                np.savetxt(f, IoUs[np.newaxis], fmt='%.4f')
                f.write('mF1: ' + str(np.mean(F1s))+'\n')
                np.savetxt(f, F1s[np.newaxis], fmt='%.4f')
                f.write('***********************************'+ '\n')
        else:
            with open(ioupath, 'a') as f:
                f.write('Data_Name: ' + test_loader.dataset.path + '\n')
                f.write('OA: ' + str(OAs) + '\n')
                f.write('mIoU: ' + str(mIoU) + '\n')
                np.savetxt(f, IoUs[np.newaxis], fmt='%.4f')
                f.write('mF1: ' + str(np.mean(F1s)) + '\n')
                np.savetxt(f, F1s[np.newaxis], fmt='%.4f')
                f.write('***********************************' + '\n')

        return mIoU,OAs,os.path.basename(test_loader.dataset.path)

if __name__ == '__main__':

    # Set which gpu is going to be used
    # GPU_ID = '0,1,2,3'
    GPU_ID = '0,1'
    cudaDevice = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # path=r'/data/ALS_CTTA_KUA/H3DCTTA_Final2'
    path=r'data/ISPRSC'

    foldlist=list_subfolders(path)

    config,chosen_chkp=Prepare_parameter()

    label_to_names = {
        0: 'Powerline',
        1: 'Low vegetation',
        2: 'Impervious surfaces',
        3: 'Car',
        4: 'Fence',
        5: 'Roof',
        6: 'Facade',
        7: 'Shrub',
        8: 'Tree',
        9: 'other',
        10: 'other1',
        11: 'other2'
    }  # 针对ISPRS建立标签字典
    label_values = np.sort([k for k, v in label_to_names.items()])
    ignored_labels = np.array([11])

    net,device,optimizer=Prepare_model(config,chosen_chkp,label_values,ignored_labels)



    mode='our'
    message='fwefe'
    config.validation_size = 2000
    config.input_threads = 10 #4 #10
    config.datasetClass = 'H3D'

    net=SelectADAPTATION(net,optimizer,config,modelName=mode)

    ioufile_path = os.path.join(path, 'ALLtest_'+mode+'_.txt')
    if os.path.exists(ioufile_path):
        os.remove(ioufile_path)



    batchsize = 2

    All_dataset_IoU=[]
    All_dataset_OA = []
    All_dataset_name = []
    '数据集固定顺序'
    print(foldlist)

    datasetIndex = np.array(
          [5, 4, 6, 2, 3, 1, 0])
    # for filepath in foldlist:
    for ij in tqdm(range(len(datasetIndex)),desc='*JinDuing********'):
        i=datasetIndex[ij]
        filepath=foldlist[i]
        print('数据集的名称:',filepath)
        start_time=time.time()
        test_dataset = CommonDataset(config, filepath, set='validation', use_potentials=True)
        test_sampler = CommonSampler(test_dataset)
        collate_fn = CommonCollate

            # Data loader
        test_loader = DataLoader(test_dataset,
                                 batch_size=batchsize,
                                 sampler=test_sampler,
                                 collate_fn=collate_fn,
                                 num_workers=config.input_threads,
                                 pin_memory=False)

        # Calibrate samplers
        test_sampler.calibrationcommon(test_loader, verbose=True)


        print('Done in {:.1f}s\n'.format(time.time() - start_time))

        print('\nStart test')
        print('**********\n')

        # Training
        '''针对Tent,online,遇见新的域时要重置'''
        # net.reset()
        if config.dataset_task == 'cloud_segmentation':
            miou,oa,dname=cloud_segmentation_test(net, test_loader, config,device,ioufile_path,mode+message)
            All_dataset_IoU.append(miou)
            All_dataset_OA.append(oa)
            All_dataset_name.append(dname)
        else:
            raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

        del test_loader
    All_dataset_IoU=np.array(All_dataset_IoU)
    All_dataset_OA=np.array(All_dataset_OA)
    print('所有数据集的mOA: ',np.mean(All_dataset_OA))
    print('所有数据集的mMIoU: ',np.mean(All_dataset_IoU))

    if os.path.exists(ioufile_path):
        with open(ioufile_path, 'a') as f:
            f.write('------------------------------------------------------------' + '\n')
            f.write('***********************************'+ '\n')
            f.write('所有数据集的测试结果平均值'+ '\n')
            f.write('Dataset_Name   OA    mIoU'+'\n')
            for item1, item2, item3 in zip(All_dataset_name, All_dataset_OA, All_dataset_IoU):
                f.write(f"{item1} {item2} {item3}\n")

            f.write('\n')
            f.write('***********************************' + '\n')
            f.write('Mean OA: ' + str(np.mean(All_dataset_OA)) + '\n')
            f.write('\n')
            f.write('Mean mIoU: ' + str(np.mean(All_dataset_IoU)) + '\n')
            f.write('***********************************'+ '\n')



