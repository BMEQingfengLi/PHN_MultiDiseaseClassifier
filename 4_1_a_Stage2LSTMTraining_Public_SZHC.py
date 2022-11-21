
import pandas as pd
import torch
import random
import os
import time
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score
from torch.autograd import Variable

from Utils.MRNetforStage2 import MRNet
from Utils.CreatDataset_forESNet_agegender import CreatDatasetStage2
from Utils.FocalLoss import Focalloss
from Utils.MultiClassACC import MulticlassACCScore
from Utils.EnsembleNet import EnsembleNet
from Utils.LSTM_agegender import LSTMclassifier

def sen_score(true_label_list, pred_label_list_np):

    return recall_score(true_label_list, pred_label_list_np)


def spe_score(true_label_list, pred_label_list_np):
    negcase_amount = 0
    true_neg_amount = 0
    for idx in range(len(true_label_list)):
        if true_label_list[idx] == 0:
            negcase_amount += 1
            if pred_label_list_np[idx] == 0:
                true_neg_amount += 1
    spe = true_neg_amount / negcase_amount
    return spe


def training(data_rootdir,
             train_namelist_np,
             train_labellist_np,
             val_namelist_np,
             val_labellist_np,
             modelandresult_savepth,
             label_amount,
             epoch_num,
             feature_name_suffix,
                 pretrained_modelpth):
    '''

    :param data_rootdir:
    :param train_namelist_np:
    :param train_labellist_np:
    :param val_namelist_np:
    :param val_labellist_np:
    :param modelandresult_savepth:
    :param label_amount:
    :param epoch_num:
    :param fold:
    :param feature_name_suffix:
    :return:
    '''
    print('############## Training beginning... ##############')

    average_patch_loss_list = []
    average_patch_valloss_list = []
    average_patch_acc_list = []
    average_patch_valacc_list = []

    average_valsen_list = []
    average_valspe_list = []

    name_train_list = train_namelist_np.tolist()
    label_train_list = train_labellist_np.tolist()
    name_val_list = val_namelist_np.tolist()
    label_val_list = val_labellist_np.tolist()

    train_dataset = CreatDatasetStage2(name_train_list, label_train_list, data_rootdir, feature_name_suffix, traintestflag='train')
    val_dataset = CreatDatasetStage2(name_val_list, label_val_list, data_rootdir, feature_name_suffix, traintestflag='val')
    data_train_loader = DataLoader(dataset=train_dataset,
                                   batch_size=21,
                                   shuffle=True,
                                   num_workers=4)
    data_val_loader = DataLoader(dataset=val_dataset,
                                  batch_size=21,
                                  shuffle=True,
                                  num_workers=4)
    # define model
    esnet = LSTMclassifier()
    esnet.cuda()
    esnet = nn.DataParallel(esnet)
    param = torch.load(pretrained_modelpth)
    esnet.load_state_dict(param)

    # optimizer set
    optimizer = optim.Adam(esnet.parameters(),
                           lr=1e-4,
                           weight_decay=1e-8)

    # loss function set
    loss_function = Focalloss(class_num=label_amount)

    for epoch in range(1, epoch_num):
        epoch_start_time = time.time()
        esnet.train()
        currentepoch_patch_avg_acc = 0.0
        currentepoch_patch_avg_loss = 0.0

        batch_counter = 0
        batch_start_time = time.time()
        for i, (case_feature_tensor, label_tensor, age, gender) in enumerate(data_train_loader):
            batch_counter += 1
            label_tensor = Variable(label_tensor).cuda()
            label_tensor_reshapeaspred = label_tensor[:, 0, 0]
            case_feature_tensor = Variable(case_feature_tensor).cuda()
            agegender_tensor = torch.cat((age, gender), dim=1)
            agegender_tensor = Variable(agegender_tensor).cuda()

            ##### LSTM
            case_feature_tensor = case_feature_tensor.view(case_feature_tensor.shape[0], 10, 256)
            label_pred_prob_patch = esnet(case_feature_tensor, agegender_tensor)
            #####
            label_pred_patch = label_pred_prob_patch.data.max(1)[1]
            acc_patch = MulticlassACCScore(label_tensor_reshapeaspred.cpu().detach().numpy(),
                                             label_pred_patch.cpu().detach().numpy())
            currentepoch_patch_avg_acc += acc_patch
            loss_patch = loss_function(label_pred_prob_patch, label_tensor)
            optimizer.zero_grad()
            loss_patch.backward()
            optimizer.step()
            currentepoch_patch_avg_loss += float(loss_patch)

            print('TRAINING EPOCH: %d  Batch: %d  ACC: %.4f  Loss: %.4f  time: %.2fs'
                  % (epoch, batch_counter, acc_patch, float(loss_patch), time.time() - batch_start_time))
            batch_start_time = time.time()
            torch.cuda.empty_cache()

        currentepoch_patch_avg_loss /= batch_counter
        currentepoch_patch_avg_acc /= batch_counter

        average_patch_loss_list.append(currentepoch_patch_avg_loss)
        average_patch_acc_list.append(currentepoch_patch_avg_acc)
        print('TRAINING EPOCH: %d  AvgACC: %.4f  AvgLoss: %.4f  time: %.2fs' %
              (epoch,
               currentepoch_patch_avg_acc,
               currentepoch_patch_avg_loss,
               time.time() - epoch_start_time))

        # testing
        val_start_time = time.time()
        currentepochval_patch_avg_acc = 0.0
        currentepochval_patch_avg_loss = 0.0

        batch_counter = 0
        esnet.eval()
        true_val_labellist = []
        pred_val_labellist = []

        for i, (case_feature_tensor, label_tensor, age, gender) in enumerate(data_val_loader):
            batch_counter += 1
            label_tensor = Variable(label_tensor).cuda()
            label_tensor_reshapeaspred = label_tensor[:, 0, 0]
            case_feature_tensor = Variable(case_feature_tensor).cuda()
            agegender_tensor = torch.cat((age, gender), dim=1)
            agegender_tensor = Variable(agegender_tensor).cuda()

            with torch.no_grad():
                ##### LSTM
                case_feature_tensor = case_feature_tensor.view(case_feature_tensor.shape[0], 10, 256)
                label_pred_prob_patch = esnet(case_feature_tensor, agegender_tensor)
                #####
                label_pred_patch = label_pred_prob_patch.data.max(1)[1]
                acc_patch = MulticlassACCScore(label_tensor_reshapeaspred.cpu().detach().numpy(),
                                                 label_pred_patch.cpu().detach().numpy())
                currentepochval_patch_avg_acc += acc_patch
                loss_patch = loss_function(label_pred_prob_patch, label_tensor)
                currentepochval_patch_avg_loss += float(loss_patch)

                for truelabel in label_tensor_reshapeaspred.cpu().detach().numpy():
                    true_val_labellist.append(truelabel)

                for predlabel in label_pred_patch.cpu().detach().numpy():
                    pred_val_labellist.append(predlabel)

                torch.cuda.empty_cache()

        currentepochval_patch_avg_loss /= batch_counter
        currentepochval_patch_avg_acc /= batch_counter

        currentepoch_val_sen = sen_score(true_val_labellist, pred_val_labellist)
        currentepoch_val_spe = spe_score(true_val_labellist, pred_val_labellist)
        # save model
        if epoch % 1 == 0:
            torch.save(esnet.state_dict(),
                       os.path.join(modelandresult_savepth, 'model_epoch_%d.pth' % (epoch)))
        average_patch_valloss_list.append(currentepochval_patch_avg_loss)
        average_patch_valacc_list.append(currentepochval_patch_avg_acc)

        average_valsen_list.append(currentepoch_val_sen)
        average_valspe_list.append(currentepoch_val_spe)

        print('TESTING EPOCH: %d  Avg3ACC: %.4f  Avg3SEN: %.4f  Avg3SPE: %.4f  Avg3Loss: %.4f  valtime: %.2fs' %
              (epoch,
               currentepochval_patch_avg_acc,
               currentepoch_val_sen,
               currentepoch_val_spe,
               currentepochval_patch_avg_loss,
               time.time() - val_start_time))

        # plot acc
        plt.figure(1)
        plt.xlabel('Epoch')
        plt.ylabel('ACC')
        plt.grid()
        x = range(1, epoch + 1)
        plt.plot(x, average_patch_acc_list, label='Training ACC', color='turquoise')
        plt.plot(x, average_patch_valacc_list, label='Validation ACC', color='lightseagreen')
        plt.plot(x, average_valsen_list, label='Val SEN', color='lime')
        plt.plot(x, average_valspe_list, label='Val SPE', color='green')

        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        plt.rcParams['figure.dpi'] = 1000
        plt.rcParams['savefig.dpi'] = 1000
        plt.tight_layout()
        plt.savefig(os.path.join(modelandresult_savepth, 'ACC.png'))
        plt.savefig(os.path.join(modelandresult_savepth, 'ACC.svg'))
        np.save(os.path.join(modelandresult_savepth, 'average_patch_acc.npy'),
                np.array(average_patch_acc_list))
        np.save(os.path.join(modelandresult_savepth, 'average_patch_valacc.npy'),
                np.array(average_patch_valacc_list))


        np.save(os.path.join(modelandresult_savepth, 'average_val_sen.npy'),
                np.array(average_valsen_list))
        np.save(os.path.join(modelandresult_savepth, 'average_test_spe.npy'),
                np.array(average_valspe_list))

        plt.clf()

        # plot loss
        plt.figure(1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        x = range(1, epoch + 1)
        plt.plot(x, average_patch_loss_list, label='Training loss', color='turquoise')
        plt.plot(x, average_patch_valloss_list, label='Validation loss', color='lightseagreen')
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        plt.rcParams['figure.dpi'] = 1000
        plt.rcParams['savefig.dpi'] = 1000
        plt.tight_layout()
        plt.savefig(os.path.join(modelandresult_savepth, 'Loss.png'))
        plt.savefig(os.path.join(modelandresult_savepth, 'Loss.svg'))
        plt.clf()
        np.save(os.path.join(modelandresult_savepth, 'average_patch3_loss.npy'),
                np.array(average_patch_loss_list))
        np.save(os.path.join(modelandresult_savepth, 'average_patch3_testloss.npy'),
                np.array(average_patch_valloss_list))
        plt.clf()


def main(data_rootdir,
         origdatarootpth,
         modelandresult_savepth,
         label_amount,
         epoch_num,
         feature_name_suffix,
         origdata_train_csvpth,
         origdata_val_csvpth,
         pair,
                 pretrained_modelpth):
    '''

    :param data_rootdir:
    :param origdatarootpth:
    :param modelandresult_savepth:
    :param label_amount:
    :param epoch_num:
    :param fold:
    :param feature_name_suffix:
    :return:
    '''
    random.seed(0)
    origdata_traincsv = pd.read_csv(origdata_train_csvpth)
    origdata_valcsv = pd.read_csv(origdata_val_csvpth)
    train_namelist_np = np.array(origdata_traincsv['name'])
    train_labellist_np = np.array(origdata_traincsv['label'])
    val_namelist_np = np.array(origdata_valcsv['name'])
    val_labellist_np = np.array(origdata_valcsv['label'])

    ##############################################################
    # # HC=1, others=0
    # new_trainnamelist = []
    # new_trainlabellist = []
    # for idx in range(len(train_namelist_np)):
    #     if train_labellist_np[idx] == 0:
    #         new_trainlabellist.append(1)
    #         new_trainnamelist.append(train_namelist_np[idx])
    #     if train_labellist_np[idx] != 0:
    #         new_trainlabellist.append(0)
    #         new_trainnamelist.append(train_namelist_np[idx])
    # train_namelist_np = np.array(new_trainnamelist)
    # train_labellist_np = np.array(new_trainlabellist)
    #
    # new_valnamelist = []
    # new_vallabellist = []
    # for idx in range(len(val_namelist_np)):
    #     if val_labellist_np[idx] == 0:
    #         new_vallabellist.append(1)
    #         new_valnamelist.append(val_namelist_np[idx])
    #     if val_labellist_np[idx] != 0:
    #         new_vallabellist.append(0)
    #         new_valnamelist.append(val_namelist_np[idx])
    # val_namelist_np = np.array(new_valnamelist)
    # val_labellist_np = np.array(new_vallabellist)

    if pair == 'HCOthers':
        # HC=1, others=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 0:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] != 0:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 0:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] != 0:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)


    elif pair == 'BDHC':
        # BD=1, HC=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 1:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] == 0:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 1:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] == 0:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)

    elif pair == 'BDMDD':
        # BD=1, MDD=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 1:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] == 2:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 1:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] == 2:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)

    elif pair == 'BDOCD':
        # BD=1, OCD=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 1:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] == 4:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 1:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] == 4:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)


    elif pair == 'BDOthers':
        # BD=1, others=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 1:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] != 1:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 1:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] != 1:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)

    elif pair == 'BDSZ':
        # BD=1, SZ=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 1:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] == 3:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 1:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] == 3:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)

    elif pair == 'MDDHC':
        # MDD=1, HC=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 2:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] == 0:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 2:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] == 0:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)

    elif pair == 'MDDOCD':
        # MDD=1, OCD=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 2:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] == 4:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 2:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] == 4:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)

    elif pair == 'MDDOthers':
        # MDD=1, others=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 2:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] != 2:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 2:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] != 2:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)

    elif pair == 'MDDSZ':
        # MDD=1, SZ=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 2:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] == 3:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 2:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] == 3:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)

    elif pair == 'OCDHC':
        # OCD=1, HC=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 4:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] == 0:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 4:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] == 0:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)

    elif pair == 'OCDOthers':
        # OCD=1, others=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 4:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] != 4:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)
        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 4:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] != 4:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)

    elif pair == 'SZHC':
        # SZ=1, HC=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 3:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] == 0:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 3:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] == 0:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)


    elif pair == 'SZOCD':
        # SZ=1, OCD=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 3:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] == 4:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 3:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] == 4:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)


    elif pair == 'SZOthers':
        # SZ=1, others=0
        new_trainnamelist = []
        new_trainlabellist = []
        for idx in range(len(train_namelist_np)):
            if train_labellist_np[idx] == 3:
                new_trainlabellist.append(1)
                new_trainnamelist.append(train_namelist_np[idx])
            if train_labellist_np[idx] != 3:
                new_trainlabellist.append(0)
                new_trainnamelist.append(train_namelist_np[idx])
        train_namelist_np = np.array(new_trainnamelist)
        train_labellist_np = np.array(new_trainlabellist)

        new_valnamelist = []
        new_vallabellist = []
        for idx in range(len(val_namelist_np)):
            if val_labellist_np[idx] == 3:
                new_vallabellist.append(1)
                new_valnamelist.append(val_namelist_np[idx])
            if val_labellist_np[idx] != 3:
                new_vallabellist.append(0)
                new_valnamelist.append(val_namelist_np[idx])
        val_namelist_np = np.array(new_valnamelist)
        val_labellist_np = np.array(new_vallabellist)
    ##############################################################

    if not os.path.isdir(modelandresult_savepth):
        os.mkdir(modelandresult_savepth)

    # training
    training(data_rootdir,
             train_namelist_np,
             train_labellist_np,
             val_namelist_np,
             val_labellist_np,
             modelandresult_savepth,
             label_amount,
             epoch_num,
             feature_name_suffix,
                 pretrained_modelpth)


if __name__ == '__main__':
    random.seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_rootdir = '/HuaweiData/sharehome/yxpt/homeDocument/Documents/DL_trainingdata/20220809_PHN_Data/PublicCombine/data/'
    origdatarootpth = r'./PublicDataSplit'

    current_pair_list = ['SZHC']
    for current_pair in current_pair_list:
        modelandresult_savepth_list = [r'./Save/Stage2_LSTMAgegen_%s' % current_pair]
        #val_Stage2Featurenp_HCOthers_fold3.npy
        feature_name_suffix_list = ['Stage2FeatureTop50_%s.npy' % current_pair]

        for fold in [1]:
            modelandresult_savepth = modelandresult_savepth_list[fold - 1]
            feature_name_suffix = feature_name_suffix_list[fold - 1]

            origdata_train_csvpth = os.path.join(origdatarootpth, '5foldcv_fold%d_train.csv' % fold)
            origdata_val_csvpth = os.path.join(origdatarootpth, '5foldcv_fold%d_val.csv' % fold)
            pretrained_modelpth = "/home/yxpt/Desktop/Projects/H_PDN_retrain_20210317/Stage2_pickedmodel/Stage2_LSTMAgegen_SZHC_fold1/model_epoch_56.pth"

            label_amount = 2
            epoch_num = 200
            main(data_rootdir,
                 origdatarootpth,
                 modelandresult_savepth,
                 label_amount,
                 epoch_num,
                 feature_name_suffix,
                 origdata_train_csvpth,
                 origdata_val_csvpth,
                 current_pair,
                 pretrained_modelpth)