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
    def __init__(self, name, label, data_rootdir, radius, attennmp_name_suffix, traintestflag='train'):
        self.allname = name
        self.alllabel = label
        self.data_rootdir = data_rootdir
        self.radius = radius
        self.traintestflag = traintestflag
        self.attennmp_name_suffix = attennmp_name_suffix

    def __getitem__(self, item):
        if self.traintestflag == 'train':
            allname, alllabel = downsampleforbalance(self.allname, self.alllabel)
            item = int((item / len(self.allname)) * len(allname))
            attenmp_img_name = 'Train_' + self.attennmp_name_suffix
        else:
            allname = self.allname
            alllabel = self.alllabel
            attenmp_img_name = 'Val_' + self.attennmp_name_suffix

        casename = allname[item]
        case_imgdir = os.path.join(self.data_rootdir, casename, "t1_brain.nii.gz")
        case_img = sitk.ReadImage(case_imgdir)
        case_img_np = sitk.GetArrayFromImage(case_img)

        attentmp_imgdir = os.path.join(self.data_rootdir, casename, attenmp_img_name)
        attenmp_img = sitk.ReadImage(attentmp_imgdir)
        attenm_img_np = sitk.GetArrayFromImage(attenmp_img)

        # normalization
        case_img_np = (case_img_np - case_img_np.min()) / (case_img_np.max() - case_img_np.min())


        patch1_np = np.zeros((0, 0, 0))
        patch1_center = 0
        while patch1_np.shape != (self.radius*2, self.radius*2, self.radius*2):
            patch1_center = np.argwhere(attenm_img_np == attenm_img_np.max())[0]
            patch1_np = croppatchacrosscenter(center_coordinate_np=patch1_center,
                                               origin_img_np=case_img_np,
                                               patch_radius=self.radius)
        bubbled_attenmp_np = ZeroBubble(patch1_center, attenm_img_np, self.radius)
        patch1_np = patch1_np[np.newaxis, :, :, :]

        patch2_np = np.zeros((0, 0, 0))
        patch2_center = 0
        while patch2_np.shape != (self.radius*2, self.radius*2, self.radius*2):
            patch2_center = np.argwhere(attenm_img_np == attenm_img_np.max())[0]
            patch2_np = croppatchacrosscenter(center_coordinate_np=patch2_center,
                                               origin_img_np=case_img_np,
                                               patch_radius=self.radius)
        bubbled_attenmp_np = ZeroBubble(patch2_center, bubbled_attenmp_np, self.radius)
        patch2_np = patch2_np[np.newaxis, :, :, :]

        patch3_np = np.zeros((0, 0, 0))
        patch3_center = 0
        while patch3_np.shape != (self.radius*2, self.radius*2, self.radius*2):
            patch3_center = np.argwhere(attenm_img_np == attenm_img_np.max())[0]
            patch3_np = croppatchacrosscenter(center_coordinate_np=patch3_center,
                                               origin_img_np=case_img_np,
                                               patch_radius=self.radius)
        bubbled_attenmp_np = ZeroBubble(patch3_center, bubbled_attenmp_np, self.radius)
        patch3_np = patch3_np[np.newaxis, :, :, :]

        patch4_np = np.zeros((0, 0, 0))
        patch4_center = 0
        while patch4_np.shape != (self.radius*2, self.radius*2, self.radius*2):
            patch4_center = np.argwhere(attenm_img_np == attenm_img_np.max())[0]
            patch4_np = croppatchacrosscenter(center_coordinate_np=patch4_center,
                                               origin_img_np=case_img_np,
                                               patch_radius=self.radius)
        bubbled_attenmp_np = ZeroBubble(patch4_center, bubbled_attenmp_np, self.radius)
        patch4_np = patch4_np[np.newaxis, :, :, :]

        patch5_np = np.zeros((0, 0, 0))
        patch5_center = 0
        while patch5_np.shape != (self.radius*2, self.radius*2, self.radius*2):
            patch5_center = np.argwhere(attenm_img_np == attenm_img_np.max())[0]
            patch5_np = croppatchacrosscenter(center_coordinate_np=patch5_center,
                                               origin_img_np=case_img_np,
                                               patch_radius=self.radius)
        bubbled_attenmp_np = ZeroBubble(patch5_center, bubbled_attenmp_np, self.radius)
        patch5_np = patch5_np[np.newaxis, :, :, :]

        patch6_np = np.zeros((0, 0, 0))
        patch6_center = 0
        while patch6_np.shape != (self.radius*2, self.radius*2, self.radius*2):
            patch6_center = np.argwhere(attenm_img_np == attenm_img_np.max())[0]
            patch6_np = croppatchacrosscenter(center_coordinate_np=patch6_center,
                                               origin_img_np=case_img_np,
                                               patch_radius=self.radius)
        bubbled_attenmp_np = ZeroBubble(patch6_center, bubbled_attenmp_np, self.radius)
        patch6_np = patch6_np[np.newaxis, :, :, :]

        patch7_np = np.zeros((0, 0, 0))
        patch7_center = 0
        while patch7_np.shape != (self.radius*2, self.radius*2, self.radius*2):
            patch7_center = np.argwhere(attenm_img_np == attenm_img_np.max())[0]
            patch7_np = croppatchacrosscenter(center_coordinate_np=patch7_center,
                                               origin_img_np=case_img_np,
                                               patch_radius=self.radius)
        bubbled_attenmp_np = ZeroBubble(patch7_center, bubbled_attenmp_np, self.radius)
        patch7_np = patch7_np[np.newaxis, :, :, :]

        patch8_np = np.zeros((0, 0, 0))
        patch8_center = 0
        while patch8_np.shape != (self.radius*2, self.radius*2, self.radius*2):
            patch8_center = np.argwhere(attenm_img_np == attenm_img_np.max())[0]
            patch8_np = croppatchacrosscenter(center_coordinate_np=patch8_center,
                                               origin_img_np=case_img_np,
                                               patch_radius=self.radius)
        bubbled_attenmp_np = ZeroBubble(patch8_center, bubbled_attenmp_np, self.radius)
        patch8_np = patch8_np[np.newaxis, :, :, :]

        patch9_np = np.zeros((0, 0, 0))
        patch9_center = 0
        while patch9_np.shape != (self.radius*2, self.radius*2, self.radius*2):
            patch9_center = np.argwhere(attenm_img_np == attenm_img_np.max())[0]
            patch9_np = croppatchacrosscenter(center_coordinate_np=patch9_center,
                                               origin_img_np=case_img_np,
                                               patch_radius=self.radius)
        bubbled_attenmp_np = ZeroBubble(patch9_center, bubbled_attenmp_np, self.radius)
        patch9_np = patch9_np[np.newaxis, :, :, :]

        patch10_np = np.zeros((0, 0, 0))
        patch10_center = 0
        while patch10_np.shape != (self.radius*2, self.radius*2, self.radius*2):
            patch10_center = np.argwhere(attenm_img_np == attenm_img_np.max())[0]
            patch10_np = croppatchacrosscenter(center_coordinate_np=patch10_center,
                                               origin_img_np=case_img_np,
                                               patch_radius=self.radius)
        bubbled_attenmp_np = ZeroBubble(patch10_center, bubbled_attenmp_np, self.radius)
        patch10_np = patch10_np[np.newaxis, :, :, :]

        # numpy to tensor
        patch1_tensor = torch.from_numpy(patch1_np).float()
        patch2_tensor = torch.from_numpy(patch2_np).float()
        patch3_tensor = torch.from_numpy(patch3_np).float()
        patch4_tensor = torch.from_numpy(patch4_np).float()
        patch5_tensor = torch.from_numpy(patch5_np).float()
        patch6_tensor = torch.from_numpy(patch6_np).float()
        patch7_tensor = torch.from_numpy(patch7_np).float()
        patch8_tensor = torch.from_numpy(patch8_np).float()
        patch9_tensor = torch.from_numpy(patch9_np).float()
        patch10_tensor = torch.from_numpy(patch10_np).float()
        label = np.array([alllabel[item]])
        label = label[np.newaxis, :]
        label_tensor = torch.from_numpy(label).int()
        return patch1_tensor, \
               patch2_tensor, \
               patch3_tensor, \
               patch4_tensor,\
               patch5_tensor,\
               patch6_tensor,\
               patch7_tensor,\
               patch8_tensor,\
               patch9_tensor,\
               patch10_tensor,\
               label_tensor

    def __len__(self):
        return len(self.alllabel)


