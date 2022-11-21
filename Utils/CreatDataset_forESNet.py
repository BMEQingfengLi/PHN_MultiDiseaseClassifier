from torch.utils.data import Dataset
import SimpleITK as sitk
import os
import numpy as np
import torch
import random

from Utils.CenterRefine import center_refine
from Utils.CropPatchAcrossCenter import croppatchacrosscenter

def downsampleforbalance(allname_list, alllabel_list):
    '''

    :param allname_list:
    :param alllabel_list:
    :return:
    '''
    label_unique_list_np = np.unique(np.array(alllabel_list))
    label_amount_list = []
    for uniquelabel in label_unique_list_np:
        label_amount_list.append(len(np.argwhere(np.array(alllabel_list) == uniquelabel)))
    label_amount_list_np = np.array(label_amount_list)
    min_amount = label_amount_list_np.min()

    # shuffle original daaset
    origin_idxlist = list(range(len(allname_list)))
    random.shuffle(origin_idxlist)
    new_allname_list = []
    new_alllabel_list = []
    for idx in origin_idxlist:
        new_allname_list.append(allname_list[idx])
        new_alllabel_list.append(alllabel_list[idx])

    # downsample
    new_namelist = []
    new_labellist = []
    for label in label_unique_list_np:
        counter = 0
        for idx in range(len(new_allname_list)):
            current_label = new_alllabel_list[idx]
            if counter < min_amount:
                if current_label == label:
                    new_namelist.append(new_allname_list[idx])
                    new_labellist.append(current_label)
                    counter += 1
            else:
                break

    return new_namelist, new_labellist


def ZeroBubble(patch_center, attenm_img_np, radius):
    '''

    :param patch_center:
    :param attenm_img_np:
    :param radius:
    :return:
    '''
    # check limit
    x_leftlim = patch_center[0] - radius
    x_rightlim = patch_center[0] + radius
    y_leftlim = patch_center[1] - radius
    y_rightlim = patch_center[1] + radius
    z_leftlim = patch_center[2] - radius
    z_rightlim = patch_center[2] + radius
    if x_leftlim <= 0:
        x_leftlim = 0
    if x_rightlim >= attenm_img_np.shape[0]:
        x_rightlim = attenm_img_np.shape[0]
    if y_leftlim <= 0:
        y_leftlim = 0
    if y_rightlim >= attenm_img_np.shape[1]:
        y_rightlim = attenm_img_np.shape[1]
    if z_leftlim <= 0:
        z_leftlim = 0
    if z_rightlim >= attenm_img_np.shape[2]:
        z_rightlim = attenm_img_np.shape[2]

    for xcoord in range(x_leftlim, x_rightlim + 1):
        for ycoord in range(y_leftlim, y_rightlim + 1):
            for zcoord in range(z_leftlim, z_rightlim + 1):
                attenm_img_np[xcoord, ycoord, zcoord] = 0

    return attenm_img_np


class CreatDatasetStage2(Dataset):
    def __init__(self, name, label, data_rootdir, feature_name_suffix, traintestflag='train'):
        self.allname = name
        self.alllabel = label
        self.data_rootdir = data_rootdir
        self.traintestflag = traintestflag
        self.feature_name_suffix = feature_name_suffix

    def __getitem__(self, item):
        if self.traintestflag == 'train':
            allname, alllabel = downsampleforbalance(self.allname, self.alllabel)
            item = int((item / len(self.allname)) * len(allname))
            feature_name = 'train_' + self.feature_name_suffix
        else:
            allname = self.allname
            alllabel = self.alllabel
            feature_name = 'val_' + self.feature_name_suffix

        casename = allname[item]
        case_feature_pth = os.path.join(self.data_rootdir, casename, feature_name)
        case_feature_np = np.load(case_feature_pth)
        # case_feature_np = case_feature_np[np.newaxis, :]
        #
        # if len(case_feature_np) < 2560:
        #     pass

        # numpy to tensor
        case_feature_tensor = torch.from_numpy(case_feature_np).float()
        label = np.array([alllabel[item]])
        label = label[np.newaxis, :]
        label_tensor = torch.from_numpy(label).int()
        return case_feature_tensor, label_tensor

    def __len__(self):
        return len(self.alllabel)


