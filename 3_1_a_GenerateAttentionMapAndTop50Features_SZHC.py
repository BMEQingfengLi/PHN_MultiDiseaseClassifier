# Must picked the PDN_1 model manually
import pandas as pd
import torch
import random
import os
import time
import numpy as np
import SimpleITK as sitk
import torch.nn as nn
from Utils.CropPatchAcrossCenter import croppatchacrosscenter_probmap
import copy
from Utils.CropPatchAcrossCenter import croppatchacrosscenter_inGenerateFeature
from Utils.ZeroBubble import ZeroBubble
from torch.autograd import Variable
from Utils.MRNetforStage2 import MRNet

def ConfidenceMapGenerator(model_pth, data_rootdir, confidencemap_savename, val_namelist_np,
                               featurenp_savename):
    '''

    :param model_pth:
    :param data_rootdir:
    :param confidencemap_savename:
    :param val_namelist_np:
    :return:
    '''

    # define model
    model = MRNet(out_channel=2)
    model.cuda()
    model = nn.DataParallel(model)
    model.eval()
    param = torch.load(model_pth)
    model.load_state_dict(param)

    img_counter = 0
    for idx in range(len(val_namelist_np)):
        starttime = time.time()
        subj_name = val_namelist_np[idx]
        subj_dir = os.path.join(data_rootdir, subj_name)
        subj_imgpth = os.path.join(subj_dir, "t1_brain.nii.gz")
        img = sitk.ReadImage(subj_imgpth)
        origin = img.GetOrigin()
        direction = img.GetDirection()
        spacing = img.GetSpacing()

        img_np = sitk.GetArrayFromImage(img)
        mask = copy.deepcopy(img_np)
        mask[mask > 0] = 1

        # normalization
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        foreground_corrdinate_group_np = np.argwhere(img_np > 0)

        with torch.no_grad():
            # slide the box through the brain, and record the overlap time of each voxel
            radius = 24
            randomcenter_idx_list = []

            probmap_np = np.zeros(img_np.shape)

            # record patch overlap condition by a zero-value array
            overlap_record_np = np.zeros(img_np.shape)

            for counter in range(300):
                randomcenter_idx_list.append(np.random.randint(len(foreground_corrdinate_group_np)))
            for centercounter in range(300):
                randomcenter_idx = randomcenter_idx_list[centercounter]
                patch_center = foreground_corrdinate_group_np[randomcenter_idx]
                patch_np, refined_patch_center = croppatchacrosscenter_probmap(patch_center, img_np, radius)

                # record overlap condition
                overlap_record_np[(refined_patch_center[0]-radius):(refined_patch_center[0] + radius+1),
                                  (refined_patch_center[1]-radius):(refined_patch_center[1] + radius+1),
                                  (refined_patch_center[2]-radius):(refined_patch_center[2] + radius+1)] += 1
                patch_np = patch_np[np.newaxis, np.newaxis, :, :, :]
                patch_tensor = torch.from_numpy(patch_np).float().cuda()
                label_pred_prob, _ = model(patch_tensor)

                # record probability
                probmap_np[(refined_patch_center[0]-radius):(refined_patch_center[0] + radius+1),
                           (refined_patch_center[1]-radius):(refined_patch_center[1] + radius+1),
                           (refined_patch_center[2]-radius):(refined_patch_center[2] + radius+1)] += \
                    np.float32(label_pred_prob[0][1].cpu().detach())

            # fill 0 in overlap_record_np for further division
            # for zero_coordinate in np.argwhere(overlap_record_np == 0):
            #     overlap_record_np[zero_coordinate[0], zero_coordinate[1], zero_coordinate[2]] = 1
            overlap_record_np[np.where(overlap_record_np == 0)] = 1  # faster

            probmap_np_tmp = probmap_np / overlap_record_np
            probmap_np_tmp = np.nan_to_num(probmap_np_tmp) # remove nan
            inf_coodrinate_group = np.argwhere(np.isinf(probmap_np_tmp))
            for inf_coordinate in inf_coodrinate_group:
                probmap_np_tmp[inf_coordinate[0],
                               inf_coordinate[1],
                               inf_coordinate[2]] = 0

            # cofidence normalization
            probmap_np_tmp = 0.5 + abs(probmap_np_tmp - 0.5)

            # mask background
            probmap_np_tmp = probmap_np_tmp * mask
            probmap_np = probmap_np_tmp

            probmap_img = sitk.GetImageFromArray(probmap_np)
            probmap_img = sitk.Cast(probmap_img, sitk.sitkFloat32)
            probmap_img.SetOrigin(origin)
            probmap_img.SetDirection(direction)
            probmap_img.SetSpacing(spacing)
            sitk.WriteImage(probmap_img, os.path.join(subj_dir, confidencemap_savename))

            # Get top 50 feature vector
            # define model
            model_savefeature = MRNet(out_channel=2)
            model_savefeature.cuda()
            model_savefeature = nn.DataParallel(model_savefeature)
            model_savefeature.load_state_dict(torch.load(model_pth))

            model_savefeature.eval()
            top_featurenp_list = []
            patch_amount = 50
            for patchidx in range(patch_amount):
                currentpatch_center = np.argwhere(probmap_np == probmap_np.max())[0]
                currentpatch_np, refined_center_coordinate_np = croppatchacrosscenter_inGenerateFeature(
                    center_coordinate_np=currentpatch_center,
                    origin_img_np=img_np,
                    patch_radius=radius)
                # calculate bubbled attention map
                probmap_np = ZeroBubble(refined_center_coordinate_np, probmap_np, radius)
                currentpatch_np = currentpatch_np[np.newaxis, np.newaxis, :, :, :]

                # Extract feature from current patch
                currentpatch_tensor = torch.from_numpy(currentpatch_np).float()
                currentpatch_tensor = Variable(currentpatch_tensor).cuda()
                _, currentpatch_extractedfeature = model_savefeature(currentpatch_tensor)
                currentpatch_extractedfeature_aslist = currentpatch_extractedfeature[0].tolist()
                top_featurenp_list.append(currentpatch_extractedfeature_aslist)
            np.save(os.path.join(subj_dir, featurenp_savename), np.array(top_featurenp_list))

            img_counter += 1
            print('Total time cost: %.2fs' % (time.time() - starttime))
            print('%d/%d is done' % (img_counter, len(val_namelist_np)))


def GenerateConfidence(model_pth,
                       data_rootdir,
                       confidencemap_savename,
                       origdata_val_csvpth,
                               featurenp_savename):
    '''

    :param model_pth:
    :param data_rootdir:
    :param confidencemap_savename:
    :param origdata_val_csvpth:
    :return:
    '''
    random.seed(0)
    origdata_valcsv = pd.read_csv(origdata_val_csvpth)
    val_namelist_np = np.array(origdata_valcsv['name'])

    ConfidenceMapGenerator(model_pth,
                           data_rootdir,
                           confidencemap_savename,
                           val_namelist_np,
                               featurenp_savename)


if __name__ == '__main__':
    random.seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_rootdir = '/HuaweiData/sharehome/yxpt/homeDocument/Documents/DL_trainingdata/20220809_PHN_Data/PublicCombine/data/'
    origdatarootpth = r'./PublicDataSplit'
    pair = 'SZHC'
    # generate feature for all subjects in current fold, for 5-classification
    for fold in [1]:
        model_foldpth = os.path.join('./Stage1_pickedmodel/Public%s_fold%d' % (pair, fold))
        model_name = os.listdir(model_foldpth)[0]
        model_pth = os.path.join(model_foldpth, model_name)
        for trainvaltestflag in ['train', 'val', 'test']:
            confidencemap_savename = '%s_Confidencemap_%s_fold%d.nii.gz' % (trainvaltestflag, pair, fold)
            origdata_val_csvpth = os.path.join(origdatarootpth, '5foldcv_fold%d_%s.csv' % (fold, trainvaltestflag))
            featurenp_savename = '%s_Stage2FeatureTop50_%s.npy' % (trainvaltestflag, pair)
            GenerateConfidence(model_pth,
                               data_rootdir,
                               confidencemap_savename,
                               origdata_val_csvpth,
                               featurenp_savename)  # only run for validation set
