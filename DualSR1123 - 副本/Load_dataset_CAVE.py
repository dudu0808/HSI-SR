import torch
import numpy as np
import torch.utils.data as data
import scipy.io as sio
# from scipy.misc import imresize
import copy
# from Spa_downs import *
# from Spa_downs_gauss12 import *


names_CAVE = [
  'balloons_ms.mat','beads_ms.mat','cd_ms.mat','chart_and_stuffed_toy_ms.mat','clay_ms.mat','cloth_ms.mat','egyptian_statue_ms.mat','face_ms.mat',
    'fake_and_real_beers_ms.mat','fake_and_real_food_ms.mat','fake_and_real_lemon_slices_ms.mat','fake_and_real_lemons_ms.mat','fake_and_real_peppers_ms.mat','fake_and_real_strawberries_ms.mat',
'fake_and_real_sushi_ms.mat','fake_and_real_tomatoes_ms.mat','feathers_ms.mat','flowers_ms.mat','glass_tiles_ms.mat','hairs_ms.mat','jelly_beans_ms.mat','oil_painting_ms.mat',
'paints_ms.mat','photo_and_face_ms.mat','real_and_fake_apples_ms.mat','real_and_fake_peppers_ms.mat','sponges_ms.mat']

names_CAVE_validate=['stuffed_toys_ms.mat']
# ,'watercolors_ms.mat'
names_CAVE_test=['superballs_ms.mat','thread_spools_ms.mat']
# 'stuffed_toys_ms.mat',
names_Harvard = [
    'imge6', 'imgc4', 'imgf8', 'imgb7', 'imgd4', 'imgb1', 'imge1', 'imga6', 'imgh2', 'imgb3', 'imgf3',
    'imgf4', 'imge0', 'imgd3', 'img2', 'imgf2', 'imge5', 'imgc8', 'imge2', 'imgc7', 'imgb9', 'imgh3',
    'imgc5', 'imga7', 'imgb4', 'imgh0', 'imgd7', 'imge7', 'imgb6', 'imga5', 'imgf7', 'imgc2', 'imgf5',
    'imgb2', 'imge3', 'imgc1', 'imga1', 'imgc9', 'imgb5', 'img1', 'imgb0', 'imgd8', 'imgb8'
]

names_ICLV = [
    'BGU_HS_00001','BGU_HS_00030','BGU_HS_00060','BGU_HS_00090','BGU_HS_00120','BGU_HS_00150','BGU_HS_00180',
    'BGU_HS_00020', 'BGU_HS_00040', 'BGU_HS_00050', 'BGU_HS_00070', 'BGU_HS_00080', 'BGU_HS_00100', 'BGU_HS_00110',
    'BGU_HS_00130', 'BGU_HS_00140', 'BGU_HS_00160', 'BGU_HS_00170', 'BGU_HS_00190', 'BGU_HS_00200','BGU_HS_00010',
]

class LoadDataset(data.Dataset):
    def __init__(self, Path, datasets='CAVE', patch_size=128, stride=64, Data_Aug=False, up_mode='bicubic', Train_image_num=26):
        super(LoadDataset, self).__init__()

        if datasets == 'CAVE':
            self.names = names_CAVE
        elif datasets == 'CAVE_validate':
            self.names = names_CAVE_validate
        elif datasets == 'CAVE_test':
            self.names = names_CAVE_test
        elif datasets == 'Harvard':
            self.names = names_Harvard
        elif datasets == 'ICLV':
            self.names = names_ICLV
        else:
            assert 'wrong dataset name'

        self.path = Path                                #The path of HR HSI
        self.Image_size = 512                           #The size of original HR HSI
        self.P_S = patch_size                           #We devide the HR HSI into patches at first and this indicate the size of patch:128
        self.stride = stride                            #The stride of each patch:64.
        self.DA = Data_Aug                              #Use the data augmentation or not.
        self.P_N = int((self.Image_size-self.stride)/self.stride)    #The number of patches.
        self.up_mode = up_mode                          #The upsample mode.
        self.img_num = Train_image_num                  #The number of images in training set.


    def __getitem__(self, Index):


        P_S = self.P_S#patch_size:64
        S = self.stride#32
        P_N = self.P_N#15

        if self.DA:
            Aug = 2
        else:
            Aug = 1


        Image_size = self.Image_size#512
        Patches = P_N**2 #225
        Image_I = int(Index/Aug/Patches) #第Image_I张图片
        Patch_I = int(Index/Aug%Patches) #某一图片的第Patch_I个patch

        # Data = sio.loadmat(self.path+self.names[Image_I]+'/'+self.names[Image_I]+'/'+self.names[Image_I]+'_'+'.mat')
        Data=sio.loadmat(self.path+self.names[Image_I])

        HSI = Data['Z'] #512*512*31
        LR=Data['X']  #128*128*31
        MSI=Data['Y'] #512*512*3
        #HSI = HSI / 65535.0
        # HSI = HSI/(np.max(HSI)-np.min(HSI))
        HSI=HSI.transpose(2,0,1)#31*512*512
        LR = LR.transpose(2, 0, 1)  # 31*128*128
        MSI = MSI.transpose(2, 0, 1)  # 3*512*512

        #for i in range(HSI.shape[0]):
            # nearest,lanczos,bilinear,bicubic,cubic
        #    HSI_Up[i,:,:] = imresize(HSI[i,:,:], (MSI.shape[1], MSI.shape[2]), self.up_mode, mode='F' )

        X = int(Patch_I/P_N) #X,Y is patch index in an image,
        Y = int(Patch_I%P_N)

        #s = int(S/8)       ### set the scal factor as 8
        #p_s = int(P_S/8)
        P_S_LR=int(self.P_S/4)
        S_LR=int(self.stride/4)
        if X*S+P_S > Image_size and Y*S+P_S <= Image_size:
            HSI = HSI[:, -P_S:, Y * S: Y * S + P_S]
            MSI = MSI[:, -P_S:, Y * S: Y * S + P_S]
            LR=LR[:, -P_S_LR:, Y * S_LR: Y * S_LR + P_S_LR]

        elif X*S+P_S <= Image_size and Y*S+P_S > Image_size:
            HSI = HSI[:, X * S:X * S + P_S, -P_S:]
            MSI = MSI[:, X * S:X * S + P_S, -P_S:]
            LR=LR[:, X * S_LR:X * S_LR + P_S_LR, -P_S_LR:]
        elif X*S+P_S > Image_size and Y*S+P_S > Image_size:
            HSI = HSI[:, -P_S: , -P_S: ]
            MSI = MSI[:, -P_S: , -P_S: ]
            LR = LR[:, -P_S_LR: , -P_S_LR: ]
        else:
            HSI = HSI[:, X * S:X * S + P_S, Y * S:Y * S + P_S]
            MSI = MSI[:, X * S:X * S + P_S, Y * S:Y * S + P_S]
            LR = LR[:, X * S_LR:X * S_LR + P_S_LR, Y * S_LR:Y * S_LR + P_S_LR]

        # Data augmantation
        # if self.DA :
        #     if Index%2 == 1:
        #         a = np.random.randint(0,6,1)
        #         if a[0] == 0:
        #             GT = copy.deepcopy(np.flip(GT, 1))  # flip the array upside down
        #         elif a[0] == 1:
        #             GT = copy.deepcopy(np.flip(GT, 2))  # flip the array left to right
        #         elif a[0] == 2:
        #             GT = copy.deepcopy(np.rot90(GT, 1, [1, 2]))  # Rotate 90 degrees clockwise
        #         elif a[0] == 3:
        #             GT = copy.deepcopy(np.rot90(GT, -1, [1, 2]))  # Rotate 90 degrees counterclockwise
        #         elif a[0] == 4:
        #             GT = copy.deepcopy(np.roll(GT, int(GT.shape[1] / 2), 1))  # Roll the array up
        #         elif a[0] == 5:
        #             GT = np.roll(GT, int(GT.shape[1] / 2), 1)  # Roll the array up & left
        #             GT = copy.deepcopy(np.roll(GT, int(GT.shape[2] / 2), 2))
        HSI = torch.from_numpy(HSI)
        MSI = torch.from_numpy(MSI)
        LR = torch.from_numpy(LR)



        return HSI,MSI,LR





    def __len__(self):

        if self.DA:
            Aug = 2
        else:
            Aug = 1

        return int(self.P_N**2*self.img_num*Aug)
