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


class CreatDataset(Dataset):
    def __init__(self, name, label, data_rootdir, radius, traintestflag='train'):
        self.allname = name
        self.alllabel = label
        self.data_rootdir = data_rootdir
        self.radius = radius
        self.traintestflag = traintestflag

    def __getitem__(self, item):
        if self.traintestflag == 'train':
            allname, alllabel = downsampleforbalance(self.allname, self.alllabel)
            item = int((item / len(self.allname)) * len(allname))
        else:
            allname = self.allname
            alllabel = self.alllabel

        casename = allname[item]
        case_imgdir = os.path.join(self.data_rootdir, casename, "t1_brain.nii.gz")
        case_img = sitk.ReadImage(case_imgdir)
        case_img_np = sitk.GetArrayFromImage(case_img)

        # normalization
        case_img_np = (case_img_np - case_img_np.min()) / (case_img_np.max() - case_img_np.min())
        case_nonzero_coordinate_list_np = np.argwhere(case_img_np > 0)

        patch_np = np.zeros((0, 0, 0))
        while patch_np.shape != (self.radius*2, self.radius*2, self.radius*2):
            randomcenteridx = np.random.randint(len(case_nonzero_coordinate_list_np))
            patch_center = case_nonzero_coordinate_list_np[randomcenteridx]
            patch_np = croppatchacrosscenter(center_coordinate_np=patch_center,
                                               origin_img_np=case_img_np,
                                               patch_radius=self.radius)
        patch_np = patch_np[np.newaxis, :, :, :]

        # numpy to tensor
        patch_tensor = torch.from_numpy(patch_np).float()
        label = np.array([alllabel[item]])
        label = label[np.newaxis, :]
        label_tensor = torch.from_numpy(label).int()
        return patch_tensor, label_tensor

    def __len__(self):
        return len(self.alllabel)


