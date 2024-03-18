"""
@ Description: Main Wrapper for Classification of Streaming Data (CSD)
@ Author: Rwad Khatib                  Date: Jan, 2024
@ param Self:
@ param my param 1:
@ param my param 2:
@ return:
"""
# region ################### imports - To Modified ######################
from __future__ import print_function
# import river.stream
from utils import (shuffle_tensor_data, ReservoirSamplingCifar10, sampleObj, calc_validation_gaps, evaluate,
                   make_dataloader_for_reservoirs, train_reservoir_rand_sample, validate_model_rand_sample,
                   test_model_rand_sample, setup_scenario_files, plot_save_hist, load_CIFAR_tensors, write_log_header,
                   BatchClassifierR, evaluate_bieR, predict_and_calc_acc_tr_RForest,
                   predict_and_calc_acc_val_RForest, predict_and_calc_acc_test_RForest_lastM, FeatureExtractor,
                   update_distilled_representatives_wang_like, update_distilled_representatives,
                   visualize_distilled_images_gen, train_reservoir_distill,
                   predict_and_calc_acc_test_RForest_BestM, preprocess_features_for_val_or_test,
                   validate_distillation_r,
                   plot_save_res_df, distill_features_generic, test_distillation_r, update_distilled_images,
                   initialize_distilled_representatives_imgs, plot_save_time_res_tr, dictionary_to_dataloader,
                   save_distilled_images)

import time
import numpy as np
import pandas as pd
# import datetime
# import uuid
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset  # , Subset, IterableDataset, Sampler
from torch import cuda
from torchvision import datasets
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
import keras
import logging
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from river.tree import HoeffdingTreeClassifier
from river.stream import iter_pandas
from keras.datasets import cifar10
import torch.optim.lr_scheduler as lr_scheduler
from skmultiflow.meta import AdaptiveRandomForestClassifier
from itertools import islice

matplotlib.use('TkAgg')
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


# endregion

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> S E P E R A T O R <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #


def main():
    # region General Parameters and tests
    debug_f = False  # flag that activate action needed during debug
    dataset = 'CIFAR10'  # Chosen Dataset
    algorithm = 'Distillation'  # choose of ['Random_Sampling', 'Hoeffding, 'BIE','Adaptive_Random_Forest', 'Distillation']
    loss_f = 'Cross_Entropy'  # ['MSE', 'Cross_Entropy']
    trans_flag = True  # flag to activate transfer learning
    tensor_flag = True  # if using tensor dataset or numpy one in [Hoeffding/BIE.]
    featuresON = False  # for distillation - if the wanted approach to distill features
    num_classes = 10  # number of classes in database
    tr_val_ratio = 0.8  # split dataset ratio - validation size = (1-train_val_ratio)*train_size
    tr_size = 40000  # 40000  # size of train data derived form train whole dat
    tr_batch_s = [500]  # [1000, 750, 500, 250, 100, 10, 1]  # BatchSize for training procedure
    val_batch_s = 500  # BatchSize for validation procedure
    test_batch_s = 500  # BatchSize for test procedure
    reservoir_s = [500]  # [1000, 750, 500, 250]  # size of reservoir initial size (distilled images)

    # new - original in backup files:
    # Important Note res_train_batch_s must be the minimum between (tr_batch_s & reservior_s) and they are multiples of one another
    if tr_batch_s[0] < 1000 and tr_batch_s[0] < reservoir_s[0]:
        res_train_batch_s = tr_batch_s[0]  # multiplication of tr_batch_size
    elif reservoir_s[0] < 1000 and reservoir_s[0] < tr_batch_s[0]:
        res_train_batch_s = reservoir_s[0]
    else:
        res_train_batch_s = 1000

    learning_rate = 0.0001  # 0.001  # learning rate for NN except for Distillation its  0.0001
    # bie_num_models = 100  # how much models into the ensemble
    features_f = True  # for BIE - says if to use Resnet34 features csv or not
    # distillation Wang and regular:
    # use res size = tr_batch_size = distill_res_size - important
    distill_M = 'Rwad'  # ['Avg' || 'Wang' || 'Rwad']
    distill_resrvoir_s = 500  # smaller or equal of reservoir size and tr_batch_size
    learning_rate_distill = 0.001
    loss_f_distill = 'MSE'
    opt_distill = 'ADAM'  # ['ADAM', 'SGD']
    cnn_type = "Intermediate"  # ["Simple" , "Intermediate"]

    # Adaptive Random Forest - Default is 1
    err_scale = 20 # default 1

    # Scheduler Params:
    start_factor = 0.1
    end_factor = 1.0
    total_iters = 20
    max_loss_degredation = 3  # maximum iteration that the loss allowed to go up
    # Hyperparameters:
    # random_seed = 199
    max_n_epochs = 10  # 10  # max number of allowed epochs for train/updating the weights of the NN per batch
    # Other Definitions:
    device = "cuda:0"
    plt.style.use('seaborn-v0_8-whitegrid')
    # >>>>>>> Size Sanity Check <<<<<<<#
    if tr_size > int(tr_val_ratio * 50000):  # 50000 is for CIFAR-10
        print('Invalid reservoir size')
        exit(-2)
    # >>>>>>> gpu availability test <<<<<<<#
    gpu_available = cuda.is_available()
    print(f'=============================================')
    print(f'GPU available: {gpu_available}')
    if gpu_available:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        print(f'=============================================')
    else:
        print(f'=============================================')  # Sanity
    # endregion
    # region 1st Algo. Random Sampling
    if algorithm == 'Random_Sampling':
        # region CIFAR-10 tensor loading:
        print('=============================== Start loading %s Tensor Dataset ==================================',
              dataset)
        # Define Transformers ->  Note transforms.ToTensor() scales input images
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),  # new combination
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        #  NEW RESIZE Transforms:
        # resize_transform_32 = transforms.Compose([
        #     transforms.Resize((32, 32)),  # Resize to (32, 32)
        #     transforms.ToTensor(),  # Convert the PIL Image to a PyTorch tensor
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # ])

        # load from pytorch datasets for each order train DS, and one test data set
        train_ds_t_full, test_dataset_t, test_dataset_t_orig_size, test_dataset_t_resized = load_CIFAR_tensors(
            tr_size, train_transform, test_transform)
        print('=============================== Finished loading %s Tensor Dataset ===============================',
              dataset)
        # endregion
        for TR in tr_batch_s:
            for RS in reservoir_s:
                try:
                    # region setting combination parameters and preparing datasets and loaders
                    val_gaps = calc_validation_gaps(TR, dataset)
                    item_per_class = int(RS / num_classes)
                    # # >>>>>>> create folders for the scenario at hand <<<<<<<#
                    res_path = setup_scenario_files(algorithm, TR, RS)

                    # Split Train DataSet to Train & Val subsets [80:20]
                    val_size = int(np.round((1 - tr_val_ratio), 2) * len(train_ds_t_full))
                    train_dataset, VD_T = random_split(train_ds_t_full, [len(train_ds_t_full) - val_size, val_size])
                    train_loader = DataLoader(train_dataset, batch_size=TR, shuffle=False, drop_last=True)

                    vd_t_loader = DataLoader(VD_T, batch_size=val_batch_s, shuffle=False, drop_last=True)
                    # Creating test dataset loader:
                    test_loader = DataLoader(dataset=test_dataset_t, batch_size=test_batch_s, shuffle=False,
                                             drop_last=True)

                    test_loader_orig_size = DataLoader(dataset=test_dataset_t_orig_size, batch_size=test_batch_s,
                                                       shuffle=False,
                                                       drop_last=True)

                    plot_save_hist(train_dataset, 'Train', res_path, debug_f)
                    plot_save_hist(VD_T, 'Validation', res_path, debug_f)
                    plot_save_hist(test_dataset_t, 'Test', res_path, debug_f)

                    print('===================================== BEGIN TRAINING =====================================')
                    # endregion
                    # region loading BaseModel RESNET-34 - Pretrained/Not & define optimizer and criterion <<<<<
                    weights = ResNet34_Weights.IMAGENET1K_V1
                    base_model = resnet34(weights)  # Pre-Trained
                    # Add a Custom Classifier
                    n_inputs = base_model.fc.in_features
                    # base_model.fc = nn.Linear(n_inputs, num_classes)
                    base_model.fc = nn.Sequential(nn.Linear(n_inputs, 512), nn.ReLU(), nn.Linear(n_inputs, num_classes),
                                                  nn.Softmax(dim=1))  # nn.LogSoftmax
                    # base_model = resnet34()                  # Not Trained
                    if trans_flag:  # Freeze early layers
                        for param in base_model.parameters():
                            param.requires_grad = False
                        base_model.fc.requires_grad_(True)  # <--- form Friday (08/02/24)

                    # def lr_lambda(epoch):  # <- Scheduler
                    #     # LR to be 0.1 * (1/1+0.01*epoch)
                    #     base_lr = 0.1
                    #     factor = 0.01
                    #     return base_lr / (1 + factor * epoch)

                    # Loss & optimizer & scheduler
                    criterion = nn.CrossEntropyLoss()
                    # optimizer = torch.optim.SGD(base_model.parameters(),learning_rate,0.9, weight_decay=0.0001)
                    optimizer = torch.optim.Adam(base_model.parameters(), learning_rate, weight_decay=0.0001,
                                                 betas=(0.9, 0.999))
                    # optimizer = torch.optim.Adam(base_model.parameters(), learning_rate)
                    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
                    scheduler = lr_scheduler.LinearLR(optimizer, start_factor, end_factor, total_iters)
                    # Move to GPU
                    base_model = base_model.to(device)
                    # endregion

                    # region combination run
                    start_time = time.time()
                    eta = 0
                    with open(os.path.join(res_path, 'LogFile.txt'), 'a') as f_txt:
                        # f_txt.write('Results Summary Log\n\n') # make function - Write header
                        # f_txt.write(str('===================================================\n'))
                        # f_txt.write('Data Base Under Test: CIFAR10\n')
                        # f_txt.write(str('Transfer learning   :  %s\n' % str(trans_flag)))
                        # f_txt.write(str('validation gaps [batches]: %s\n' % str(val_gaps)))
                        # f_txt.write(str('Reservoir Size  : %d\n' % RS))
                        # f_txt.write(str('Batch Train Size: %d\n' % TR))
                        # f_txt.write(str('Batch Eval Size : %d\n' % val_batch_s))
                        # f_txt.write(str('Batch test Size : %d\n' % test_batch_s))
                        # f_txt.write(str('===================================================\n'))
                        write_log_header(f_txt, dataset, algorithm, TR, val_batch_s, test_batch_s,
                                         RS, val_gaps, learning_rate, res_train_batch_s, max_n_epochs)
                        ####################################################################################
                        batch_cnt = 0
                        val_cnt = 0
                        valid_max_acc = 0
                        full_ind = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        best_model = []
                        train_acc_all_batches = []
                        train_loss_all_batches = []
                        valid_acc_gap_batches = []
                        valid_loss_all_batches = []
                        reservoirs = [[] for i in range(num_classes)]
                        for i in range(num_classes):  # Initiate 10 reservoir [one for each class]
                            reservoirs[i] = ReservoirSamplingCifar10(
                                item_per_class)  # items per class = max size of each class res

                        # Set to training
                        base_model.train()

                        for batch_idx, (data, targets) in enumerate(train_loader):  # Train all data in each order
                            batch_cnt += 1

                            # region Reservoir Handle & preparation
                            # sampling for new item in current batch
                            for k in range(TR):
                                for i in range(num_classes):
                                    if int(targets[k]) == i:
                                        curr_obj = sampleObj(TR * batch_idx + k, data[k], targets[k])
                                        reservoirs[i].sampling(curr_obj)

                            # check if the reservoir is full
                            if sum(full_ind) < RS:
                                for f in range(num_classes):
                                    full_ind[f] = reservoirs[f].samples.__len__()

                            elif sum(full_ind) == RS:
                                # combining the 10 res to 1 & create data loader
                                curr_res_raw = make_dataloader_for_reservoirs(num_classes,
                                                                              item_per_class,
                                                                              reservoirs, device)
                                curr_res = DataLoader(curr_res_raw, batch_size=res_train_batch_s, shuffle=False,
                                                      drop_last=True)
                                # endregion

                                # region Training the current reservoir
                                base_model, tr_acc, tr_loss = train_reservoir_rand_sample(max_n_epochs, curr_res,
                                                                                          max_loss_degredation,
                                                                                          batch_idx, RS,
                                                                                          optimizer, device,
                                                                                          base_model,
                                                                                          criterion, scheduler, debug_f)

                                elapsed_time = (time.time() - start_time) / 3600  # in hours
                                f_txt.write('batch %03d/%03d | Tr_ACC: %8.f | Tr_Loss: %.8f| time_elapsed: %.2f '
                                            '[h]\n' % (batch_cnt, np.floor(tr_size / TR), tr_acc, tr_loss,
                                                       elapsed_time))
                                print('******* batch %03d/%03d | time_elapsed: %.2f[h] *******\n'
                                      % (batch_cnt, np.floor(tr_size / TR), elapsed_time))
                                train_acc_all_batches.append(tr_acc)
                                train_loss_all_batches.append(tr_loss)
                                print('-------------------------------------------------------------------------')
                                # endregion

                                # region validate the model after val_gaps training cycles:
                                # ----------- infer over validation set ------------------- #
                                #
                                # Set to evaluation mode
                                if (not batch_cnt % val_gaps) or (batch_cnt == np.floor(tr_size / TR)):
                                    print('validate after %d batches ...' % batch_cnt)
                                    base_model.eval()
                                    # torch.cuda.empty_cache()

                                    best_model, valid_acc, valid_loss, valid_max_acc = validate_model_rand_sample(
                                        vd_t_loader, valid_max_acc, batch_cnt, tr_size,
                                        start_time, val_gaps, TR, base_model, optimizer, device, criterion, val_cnt,
                                        best_model, VD_T.__len__())
                                    f_txt.write(
                                        'batch %03d/%03d | valid_acc: %.4f | valid_Loss: %.4f | time_elapsed/eta: '
                                        '%.2f[h]/%.2f[h]\n' % (batch_cnt, np.floor(tr_size / TR), valid_acc,
                                                               valid_loss, elapsed_time, eta))
                                    valid_acc_gap_batches.append(valid_acc)
                                    valid_loss_all_batches.append(valid_loss)
                                    # Setback to training
                                    base_model.train()
                                # endregion

                        # region Testing [last model & best model]
                        # -----------  last base model performance of test data set ------------------- #
                        base_model.eval()
                        test_acc_l, test_loss_last = test_model_rand_sample(test_loader, test_dataset_t.__len__(),
                                                                            device, criterion, base_model)

                        print('last Model Test accuracy: %04f | last Model Test Loss: %04f' % (
                            test_acc_l, test_loss_last))
                        # log:
                        f_txt.write(str('===================================================\n'))
                        f_txt.write(str('last Model Test accuracy: %04f | last Model Test Loss: %04f' % (
                            test_acc_l, test_loss_last)))
                        f_txt.write(str('===================================================\n'))

                        # -----------  best model performance of test data set ------------------- #
                        best_model.eval()
                        b_test_acc, b_test_loss = test_model_rand_sample(test_loader, test_dataset_t.__len__(),
                                                                         device, criterion, best_model)

                        print(
                            'Best Model Validation accuracy: %04f \nBest Model Test Accuracy / loss : %04f / %04f' % (
                                valid_max_acc, b_test_acc, b_test_loss))
                        # log:
                        f_txt.write(str('===================================================\n'))
                        f_txt.write(
                            str('Best Model Validation accuracy: %04f\nBest Model Test Accuracy / loss : %04f / %04f\n' % (
                                valid_max_acc, b_test_acc, b_test_loss)))
                        f_txt.write(str('===================================================\n'))

                        # -----------  best model performance of ORIGINAL SIZE test data set ------------------- #
                        best_model.eval()
                        test_acc_orig, test_loss_orig = test_model_rand_sample(test_loader_orig_size,
                                                                               test_dataset_t.__len__(),
                                                                               device, criterion, best_model)

                        print('Best Model Test (over ORIGINAL Size) Accuracy / loss : %04f / %04f' % (
                            test_acc_orig, test_loss_orig))
                        # log:
                        f_txt.write(str('===================================================\n'))
                        f_txt.write(str('Best Model Test (over ORIGINAL Size) Accuracy / loss : %04f / %04f\n' % (
                            test_acc_orig, test_loss_orig)))
                        f_txt.write(str('===================================================\n'))
                        # endregion
                        # -----------  save best model ------------------- #
                        torch.save(best_model, os.path.join(res_path, 'best_model.pt'))

                        # region save raw data:
                        file_n_1 = 'Tr_acc_n_loss_batch_s_' + str(TR) + '.csv'
                        filepath1 = os.path.join(res_path, file_n_1)
                        tr_acc_loss_df = pd.DataFrame({'Tr_acc': train_acc_all_batches,
                                                       'Tr_loss': train_loss_all_batches})

                        file_n_2 = 'Val_acc_n_loss_each_' + str(val_gaps) + '_batches.csv'
                        filepath2 = os.path.join(res_path, file_n_2)
                        val_acc_loss_df = pd.DataFrame({'Val_acc': valid_acc_gap_batches,
                                                        'Val_loss': valid_loss_all_batches})

                        tr_acc_loss_df.to_csv(filepath1)
                        val_acc_loss_df.to_csv(filepath2)
                        # endregion

                        # region Figures create & save:
                        # Train Accuracy & loss
                        fig, ax = plt.subplots()
                        ax.plot(np.arange(train_acc_all_batches.__len__()), train_acc_all_batches, color='b')
                        ax.set_title('Train ACC / LOSS vs update cycle batch', fontsize=20)
                        ax.set_xlabel('update cycle [batch #]', fontsize=18, color='k')
                        ax.set_ylabel('Train ACC [%]', fontsize=18, color='b')
                        ax2 = ax.twinx()
                        ax2.plot(np.arange(train_acc_all_batches.__len__()), train_loss_all_batches, color='r')
                        ax2.set_ylabel('LOSS', fontsize=18, color='r')
                        ax.legend()
                        fig_name = 'Tr_acc_loss_vs_update_cycle'
                        fig_name_s = os.path.join(res_path, fig_name)
                        plt.savefig(fig_name_s, dpi=1200)
                        plt.show(block=False)

                        # Train Accuracy & loss
                        fig, ax = plt.subplots()
                        ax.plot(np.arange(valid_acc_gap_batches.__len__()), valid_acc_gap_batches, color='b')
                        ax.set_title('Validation ACC / LOSS vs after each ' + str(val_gaps) + ' batches', fontsize=20)
                        ax.set_xlabel('after [batch * ]' + str(val_gaps), fontsize=18, color='k')
                        ax.set_ylabel('Validation ACC [%]', fontsize=18, color='b')
                        ax2 = ax.twinx()
                        ax2.plot(np.arange(valid_loss_all_batches.__len__()), valid_loss_all_batches, color='r')
                        ax2.set_ylabel('LOSS', fontsize=18, color='r')
                        ax.legend()
                        fig_name = 'Val_loss_after_each_' + str(val_gaps) + '_batches'
                        fig_name_s = os.path.join(res_path, fig_name)
                        plt.savefig(fig_name_s, dpi=1200)
                        plt.show(block=False)
                        # endregion
                    # endregion
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print('...')
                    raise
                    print('Scenario FAIL with Train BS = %d and Reservoir S = %d' % (TR, RS))
    # endregion

    # region 2rd Algo. Hoeffding
    if algorithm == 'Hoeffding':
        if not tensor_flag:
            # Load CIFAR-10 data
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()

            # Convert to grayscale (if needed) and flatten the images
            X_train_flattened = np.mean(X_train, axis=3).reshape(len(X_train), -1)
            X_test_flattened = np.mean(X_test, axis=3).reshape(len(X_test), -1)

            # Combine features and labels into a Pandas DataFrame
            # Combine features and labels into Pandas DataFrames for both training and testing sets
            train_df = pd.DataFrame(data=np.c_[X_train_flattened, y_train.flatten()],
                                    columns=[f'pixel_{i}' for i in range(X_train_flattened.shape[1])] + ['label'])
            test_df = pd.DataFrame(data=np.c_[X_test_flattened, y_test.flatten()],
                                   columns=[f'pixel_{i}' for i in range(X_test_flattened.shape[1])] + ['label'])
        else:
            # region CIFAR-10 tensor loading:
            print('=============================== Start loading %s Tensor Dataset ==================================',
                  dataset)
            # Define Transformers ->  Note transforms.ToTensor() scales input images
            train_transform = transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomHorizontalFlip(),  # new combination
                # transforms.RandomCrop(size=112, padding=4),  # new combination
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            #  NEW RESIZE Transforms:
            # resize_transform_32 = transforms.Compose([
            #     transforms.Resize((32, 32)),  # Resize to (32, 32)
            #     transforms.ToTensor(),  # Convert the PIL Image to a PyTorch tensor
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # ])

            # load from pytorch datasets for each order train DS, and one test data set
            train_ds_t_full, test_dataset_t, test_dataset_t_orig_size, test_dataset_t_resized = load_CIFAR_tensors(
                tr_size, train_transform, test_transform)
            print('=============================== Finished loading %s Tensor Dataset ===============================',
                  dataset)
            train_dataset = DataLoader(train_ds_t_full, batch_size=1, shuffle=False, drop_last=True)
            test_dataset = DataLoader(dataset=test_dataset_t, batch_size=1, shuffle=False, drop_last=True)
            # endregion

            # region Extract feature of train and test datasets _ DONE ONLY ONCE:
            # resnet_modified = ResNet34ModifiedLastLayer()
            # resnet_modified = resnet_modified.to(device)
            # arr_train_features_labels_mat = CreateFeaturesLabelsMatrixOfLayerBeforeLast(train_dataset, resnet_modified,
            #                                                                             device)
            # print('debug')
            # arr_test_features_labels_mat = CreateFeaturesLabelsMatrixOfLayerBeforeLast(test_dataset, resnet_modified,
            #                                                                            device)  # Need To Complete
            # print('debug')
            #
            # df_train_features_labels = pd.DataFrame(arr_train_features_labels_mat) # Done
            # df_test_features_labels = pd.DataFrame(arr_test_features_labels_mat) # Need To Complete
            # df_train_features_labels.to_csv('cifar_train_features_labels_from_tensors.csv', index=False) # Done
            # df_test_features_labels.to_csv('cifar_test_features_labels_from_tensors.csv', index=False)   # Need To Complete

            # Save the DataFrames as CSV files
            # train_df.to_csv('cifar10_train_data.csv', index=False)
            # test_df.to_csv('cifar10_test_data.csv', index=False)
            # endregion

        for TR in tr_batch_s:
            for RS in reservoir_s:
                res_path = setup_scenario_files(algorithm, TR)
                val_gaps = calc_validation_gaps(TR, dataset)
                with open(os.path.join(res_path, 'LogFile.txt'), 'a') as f_txt:
                    write_log_header(f_txt, dataset, algorithm, TR, val_batch_s, test_batch_s,
                                     RS, val_gaps)

                    df = pd.read_csv("./cifar_train_features_labels_from_tensors.csv")
                    label_col = df.columns[-1]
                    feature_cols = list(df.columns)
                    feature_cols.pop()
                    X = df[feature_cols]
                    Y = df[label_col]

                    # bie = BatchClassifier(window_size=100, max_models=100)
                    # bie_results = evaluate(stream=iter_pandas(X=X, y=Y), model=bie)
                    ht = HoeffdingTreeClassifier()

                    ht_results = evaluate(stream=iter_pandas(X=X, y=Y), model=ht)
    # endregion

    # region 3rd Algo. BIE:
    if algorithm == 'BIE':
        for TR in tr_batch_s:
            for RS in reservoir_s:
                res_path = setup_scenario_files(algorithm, TR, RS)
                val_gaps = calc_validation_gaps(TR, dataset)
                with open(os.path.join(res_path, 'LogFile.txt'), 'a') as f_txt:
                    write_log_header(f_txt, dataset, algorithm, TR, val_batch_s, test_batch_s,
                                     RS, val_gaps, featuresON=features_f)

                    if features_f:
                        df_Tr = pd.read_csv("./cifar_train_features_labels_from_tensors.csv")
                        train_df, val_df = train_test_split(df_Tr, test_size=0.2, random_state=42)

                        label_col_tr = train_df.columns[-1]
                        feature_cols_tr = list(train_df.columns)
                        feature_cols_tr.pop()

                        label_col_val = val_df.columns[-1]
                        feature_cols_val = list(val_df.columns)
                        feature_cols_val.pop()

                        # If we want to use the features from the last layer of Trained ResNet34
                        X_Tr = train_df[feature_cols_tr]
                        X_Tr = X_Tr.values
                        y_Tr = train_df[label_col_tr]
                        y_Tr = y_Tr.values
                        num_batches = len(X_Tr) // TR
                        X_batches = np.array_split(X_Tr, num_batches)
                        y_batches = np.array_split(y_Tr, num_batches)

                        X_val = val_df[feature_cols_val]
                        X_val = X_val.values
                        y_val = val_df[label_col_val]
                        y_val = y_val.values
                        # val_num_batches = len(X_val) // val_batch_s
                        # X_batches_v = np.array_split(X_val, val_num_batches)
                        # y_batches_v = np.array_split(y_val, val_num_batches)
                        X_val_f = X_val.reshape(X_val.shape[0], -1)

                        df_Test = pd.read_csv("./cifar_test_features_labels_from_tensors.csv")

                        label_col_test = df_Test.columns[-1]
                        feature_cols_test = list(df_Test.columns)
                        feature_cols_test.pop()

                        X_test = df_Test[feature_cols_val]
                        X_test = X_test.values
                        y_test = df_Test[label_col_val]
                        y_test = y_test.values
                        X_test_f = X_test.reshape(X_test.shape[0], -1)

                    else:
                        # Load CIFAR-10 data
                        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
                        # Split the training data into training and validation sets
                        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                                          random_state=42)

                        # Split data into batches:
                        num_batches = len(X_train) // TR  # TR = train batch size
                        X_batches = np.array_split(X_train, num_batches)
                        y_batches = np.array_split(y_train, num_batches)

                        val_num_batches = len(X_val) // val_batch_s
                        X_batches_v = np.array_split(X_val, val_num_batches)
                        y_batches_v = np.array_split(y_val, val_num_batches)
                        X_val_f = X_val.reshape(X_val.shape[0], -1)

                        test_num_batches = len(X_test) // test_batch_s
                        X_batches_t = np.array_split(X_test, test_num_batches)
                        y_batches_t = np.array_split(y_test, test_num_batches)

                    # Initiate modified BIE Classifier
                    bie = BatchClassifierR(bie_num_models)

                    bie_model, train_log, val_log = evaluate_bieR(bie, X_batches, y_batches, num_batches, val_gaps,
                                                                  X_val_f,
                                                                  y_val, f_txt, features_f)
                    # test the final ensemble model:
                    X_test_f = X_test.reshape(X_test.shape[0], -1)
                    tot_test_acc = 0
                    for item in bie_model.H:
                        y_test_pred = item.predict(X_test_f)
                        accuracy = accuracy_score(y_test, y_test_pred)
                        tot_test_acc += accuracy
                    test_acc = tot_test_acc / len(bie_model.H) * 100
                    print('test accuracy: %.3f [%%]' % test_acc)
                    f_txt.write(str('===================================================\n'))
                    f_txt.write(str('BIE Test Accuracy: %04f [%%]\n' % test_acc))
                    f_txt.write(str('===================================================\n'))
                    # 1.ask Yehudit but when fitting one tree to 100 pixel[images] it predict with one 100 percent for the
                    # same images
                    # 2. Still need to do the same on the features form .csvs

                    # save data and figures:
                    file_n_1 = 'Training_log_each_' + str(val_gaps) + '_batches' + '.csv'
                    filepath1 = os.path.join(res_path, file_n_1)
                    train_log.to_csv(filepath1)

                    file_n_2 = 'Validation_log_each_' + str(val_gaps) + '_batches' + '.csv'
                    filepath2 = os.path.join(res_path, file_n_2)
                    val_log.to_csv(filepath2)

                    # Train Accuracy & Time
                    fig, ax = plt.subplots()
                    ax.plot(train_log["batch_id"], train_log["tr_acc"], color='b')
                    ax.set_title('Train Accuracy & Time vs batch num', fontsize=20)
                    ax.set_xlabel('batch #', fontsize=18, color='k')
                    ax.set_ylabel('Train ACC [%]', fontsize=18, color='b')
                    ax2 = ax.twinx()
                    ax2.plot(train_log["batch_id"], train_log["train_time"], color='darkorange')
                    ax2.set_ylabel('Train Time [ms]', fontsize=18, color='darkorange')
                    ax.legend()
                    fig_name = 'Train_Acc_n_Timing'
                    fig_name_s = os.path.join(res_path, fig_name)
                    plt.savefig(fig_name_s, dpi=1200)
                    plt.show(block=False)

                    # Valid Accuracy & loss
                    fig, ax = plt.subplots()
                    ax.plot(val_log["idx"], val_log["val_acc"], color='b')
                    ax.set_title('Validation Accuracy & Time vs batch num', fontsize=20)
                    ax.set_xlabel('batch #', fontsize=18, color='k')
                    ax.set_ylabel('Train ACC [%]', fontsize=18, color='b')
                    ax2 = ax.twinx()
                    ax2.plot(val_log["idx"], val_log["validation_time"], color='darkorange')
                    ax2.set_ylabel('Validation Time [ms]', fontsize=18, color='darkorange')
                    ax.legend()
                    fig_name = 'Validation_Acc_n_Timing'
                    fig_name_s = os.path.join(res_path, fig_name)
                    plt.savefig(fig_name_s, dpi=1200)
                    plt.show(block=False)

    # endregion
    if algorithm == 'Adaptive_Random_Forest':
        for TR in tr_batch_s:
            for RS in reservoir_s:
                res_path = setup_scenario_files(algorithm, TR, RS)
                val_gaps = calc_validation_gaps(TR, dataset)
                with open(os.path.join(res_path, 'LogFile.txt'), 'a') as f_txt:
                    write_log_header(f_txt, dataset, algorithm, TR, val_batch_s, test_batch_s,
                                     RS, val_gaps, featuresON=features_f)

                    if features_f:
                        df_Tr = pd.read_csv("./cifar_train_features_labels_from_tensors.csv")
                        train_df, val_df = train_test_split(df_Tr, test_size=0.2, random_state=42)

                        label_col_tr = train_df.columns[-1]
                        feature_cols_tr = list(train_df.columns)
                        feature_cols_tr.pop()

                        label_col_val = val_df.columns[-1]
                        feature_cols_val = list(val_df.columns)
                        feature_cols_val.pop()

                        # If we want to use the features from the last layer of Trained ResNet34
                        X_Tr = train_df[feature_cols_tr]
                        X_Tr = X_Tr.values
                        y_Tr = train_df[label_col_tr]
                        y_Tr = y_Tr.values
                        num_batches = 1 #len(X_Tr) // TR
                        X_batches = np.array_split(X_Tr, num_batches)
                        y_batches = np.array_split(y_Tr, num_batches)
                        X_Tr_f = X_Tr.reshape(X_Tr.shape[0], -1)
                        y_Tr_f = y_Tr.ravel()

                        X_val = val_df[feature_cols_val]
                        X_val = X_val.values
                        y_val = val_df[label_col_val]
                        y_val = y_val.values
                        # val_num_batches = len(X_val) // val_batch_s
                        # X_batches_v = np.array_split(X_val, val_num_batches)
                        # y_batches_v = np.array_split(y_val, val_num_batches)
                        num_batches_v = 1 % len(X_val) // TR
                        X_val_f = X_val.reshape(X_val.shape[0], -1)
                        y_val_f = y_val.ravel()

                        df_Test = pd.read_csv("./cifar_test_features_labels_from_tensors.csv")

                        label_col_test = df_Test.columns[-1]
                        feature_cols_test = list(df_Test.columns)
                        feature_cols_test.pop()

                        X_test = df_Test[feature_cols_val]
                        X_test = X_test.values
                        y_test = df_Test[label_col_val]
                        y_test = y_test.values
                        num_batches_t = 1 % X_test // TR
                        X_test_f = X_test.reshape(X_test.shape[0], -1)
                        y_test_f = y_test.ravel()

                    else:
                        # Load CIFAR-10 data
                        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
                        # Split the training data into training and validation sets
                        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                                          random_state=42)

                        # Split data into batches:
                        num_batches = len(X_train) // TR  # TR = train batch size
                        X_batches = np.array_split(X_train, num_batches)
                        y_batches = np.array_split(y_train, num_batches)

                        val_num_batches = len(X_val) // val_batch_s
                        X_batches_v = np.array_split(X_val, val_num_batches)
                        y_batches_v = np.array_split(y_val, val_num_batches)
                        X_val_f = X_val.reshape(X_val.shape[0], -1)

                        test_num_batches = len(X_test) // test_batch_s
                        X_batches_t = np.array_split(X_test, test_num_batches)
                        y_batches_t = np.array_split(y_test, test_num_batches)

                    # Initialize the Mondrian Forest Classifier
                    mar_f = AdaptiveRandomForestClassifier(n_estimators=30)
                    batch_size = TR

                    start_t = time.time()
                    raw_tr_res = []
                    raw_val_res = []
                    idx = 0
                    best_model = []
                    best_model_acc = 1e-16
                    # Train in batches
                    classes = np.unique(y_Tr_f)
                    for start in range(0, len(X_Tr_f), batch_size):
                        idx += 1
                        end = min(start + batch_size, len(X_Tr_f))
                        mar_f.partial_fit(X_Tr_f[start:end], y_Tr[start:end], classes=classes)
                        tr_acc, tr_time = predict_and_calc_acc_tr_RForest(mar_f, X_Tr_f[start:end], y_Tr_f[start:end],
                                                                          idx, num_batches, f_txt, start_t)
                        raw_tr_res.append([algorithm, idx, tr_acc, tr_time])
                        if not idx % val_gaps:
                            val_acc, curr_val_t, best_model, best_model_acc = predict_and_calc_acc_val_RForest(mar_f,
                                                                                                               X_val_f,
                                                                                                               y_val_f,
                                                                                                               idx,
                                                                                                               num_batches_v,
                                                                                                               f_txt,
                                                                                                               best_model,
                                                                                                               best_model_acc,
                                                                                                               start_t,
                                                                                                               100)

                            raw_val_res.append([algorithm, idx, val_acc, curr_val_t])

                    # Test acc Last Model:
                    predict_and_calc_acc_test_RForest_lastM(mar_f, X_test_f, y_test_f, f_txt, 100)
                    # Test acc Best Model:
                    predict_and_calc_acc_test_RForest_BestM(best_model, X_test_f, y_test_f, f_txt, 100)

                    train_log = pd.DataFrame(raw_tr_res, columns=['algo', 'batch_id', 'tr_acc', 'train_time [s]'])
                    val_log = pd.DataFrame(raw_val_res, columns=['algo', 'batch_id', 'val_acc', 'validation_time [s]'])
                    over_all_time = time.time() - start_t
                    f_txt.write(str("over all time for train an testing Adaptive random forest algorithm is %.2f" %
                                    over_all_time))

                    # Train Accuracy & Time
                    fig, ax = plt.subplots()
                    ax.plot(train_log["batch_id"], train_log["tr_acc"], color='b')
                    ax.set_title('Train Accuracy & Accumulated Time vs batch num', fontsize=20)
                    ax.set_xlabel('batch #', fontsize=18, color='k')
                    ax.set_ylabel('Train ACC [%]', fontsize=18, color='b')
                    ax2 = ax.twinx()
                    ax2.plot(train_log["batch_id"], train_log["train_time [s]"], color='darkorange')
                    ax2.set_ylabel('Train Time [s]', fontsize=18, color='darkorange')
                    ax.legend()
                    fig_name = 'Train_Acc_n_Timing'
                    fig_name_s = os.path.join(res_path, fig_name)
                    plt.savefig(fig_name_s, dpi=1200)
                    plt.show(block=False)

                    # Valid Accuracy & loss
                    fig, ax = plt.subplots()
                    ax.plot(val_log["batch_id"], val_log["val_acc"], color='b')
                    ax.set_title('Validation Accuracy & Accumulated Time vs batch num', fontsize=20)
                    ax.set_xlabel('batch #', fontsize=18, color='k')
                    ax.set_ylabel('Train ACC [%]', fontsize=18, color='b')
                    ax2 = ax.twinx()
                    ax2.plot(val_log["batch_id"], val_log["validation_time [s]"], color='darkorange')
                    ax2.set_ylabel('Validation Time [s]', fontsize=18, color='darkorange')
                    ax.legend()
                    fig_name = 'Validation_Acc_n_Timing'
                    fig_name_s = os.path.join(res_path, fig_name)
                    plt.savefig(fig_name_s, dpi=1200)
                    plt.show(block=False)

    if algorithm == 'Distillation':
        for TR in tr_batch_s:
            for RS in reservoir_s:
                RS = distill_resrvoir_s  # test
                res_path = setup_scenario_files(algorithm, TR, RS)
                val_gaps = calc_validation_gaps(TR, dataset)
                with open(os.path.join(res_path, 'LogFile.txt'), 'a') as f_txt:
                    write_log_header(f_txt, dataset, algorithm, TR, val_batch_s, test_batch_s,
                                     RS, val_gaps, learning_rate, res_train_batch_s, max_n_epochs, distill_M,
                                     featuresON, learning_rate_distill, opt_distill, cnn_type)

                    # region CIFAR-10 tensor loading:
                    print(
                        '=============================== Start loading %s Tensor Dataset ==================================',
                        dataset)
                    # Define Transformers ->  Note transforms.ToTensor() scales input images
                    train_transform = transforms.Compose([
                        transforms.Resize(224),
                        transforms.RandomHorizontalFlip(),  # new combination
                        # transforms.RandomCrop(size=112, padding=4),  # new combination
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                    test_transform = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
                    #  NEW RESIZE Transforms:
                    # resize_transform_32 = transforms.Compose([
                    #     transforms.Resize((32, 32)),  # Resize to (32, 32)
                    #     transforms.ToTensor(),  # Convert the PIL Image to a PyTorch tensor
                    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    # ])

                    # load from pytorch datasets for each order train DS, and one test data set
                    train_ds_t_full, test_dataset_t, test_dataset_t_orig_size, test_dataset_t_resized = load_CIFAR_tensors(
                        tr_size, train_transform, test_transform)
                    print(
                        '=============================== Finished loading %s Tensor Dataset ===============================',
                        dataset)
                    # endregion

                    # region setting combination parameters and preparing datasets and loaders
                    val_gaps = calc_validation_gaps(TR, dataset)
                    items_per_class_TR = int(RS / num_classes)
                    # Split Train DataSet to Train & Val subsets [80:20]
                    val_size = int(np.round((1 - tr_val_ratio), 2) * len(train_ds_t_full))
                    train_dataset, VD_T = random_split(train_ds_t_full, [len(train_ds_t_full) - val_size, val_size])
                    train_loader = DataLoader(train_dataset, batch_size=TR, shuffle=False, drop_last=True)
                    vd_t_loader = DataLoader(VD_T, batch_size=val_batch_s, shuffle=False, drop_last=True)
                    # Creating test dataset loader:
                    test_loader = DataLoader(dataset=test_dataset_t, batch_size=test_batch_s, shuffle=False,
                                             drop_last=True)
                    plot_save_hist(train_dataset, 'Train', res_path, debug_f)
                    plot_save_hist(VD_T, 'Validation', res_path, debug_f)
                    plot_save_hist(test_dataset_t, 'Test', res_path, debug_f)

                    features_loader_v = []  # will be inhibited later in Validation
                    features_loader_t = []  # will be inhibited later in Test

                    classes_tensors_num = torch.arange(0, 10)
                    labels_tensor = classes_tensors_num.repeat_interleave(int(distill_resrvoir_s / num_classes))
                    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                                   'ship', 'truck']
                    # endregion

                    # Resnet will be used to train on distilled images
                    # region loading BaseModel RESNET-34 - Pretrained/Not & define optimizer and criterion <<<<<
                    weights = ResNet34_Weights.IMAGENET1K_V1
                    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)  # Pre-Trained
                    # model = resnet34()                  # Not Trained
                    # Add a Custom Classifier
                    n_inputs = model.fc.in_features
                    # model.fc = nn.Linear(n_inputs, num_classes)
                    model.fc = nn.Sequential(nn.Linear(n_inputs, 512), nn.ReLU(), nn.Linear(n_inputs, num_classes),
                                             nn.Softmax(dim=1))  # nn.LogSoftmax
                    if trans_flag:  # Freeze early layers
                        for param in model.parameters():
                            param.requires_grad = False
                        model.fc.requires_grad_(True)  # <--- form Friday (08/02/24)
                    # def lr_lambda(epoch):  # <- Scheduler
                    #     # LR to be 0.1 * (1/1+0.01*epoch)
                    #     base_lr = 0.1
                    #     factor = 0.01
                    #     return base_lr / (1 + factor * epoch)

                    # Loss & optimizer & scheduler
                    if loss_f == 'Cross_Entropy':
                        criterion = nn.CrossEntropyLoss()
                    elif loss_f == 'MSE':
                        criterion = nn.MSELoss()
                    # optimizer = torch.optim.SGD(model.parameters(),learning_rate,0.9, weight_decay=0.0001)
                    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0.0001,
                                                 betas=(0.9, 0.999))
                    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
                    scheduler = lr_scheduler.LinearLR(optimizer, start_factor, end_factor, total_iters)
                    # Move to GPU
                    model = model.to(device)
                    # endregion

                    res_built = False
                    reservoir = []
                    reservoir_class_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    reservoir_idx = [[]] * RS
                    items_per_class_Dist = int(distill_resrvoir_s / num_classes)
                    batch_infer_started = 0
                    # region Distillation model + loss + optimizer
                    distill_model = FeatureExtractor()
                    if loss_f_distill == 'MSE':
                        loss_fn_distill = nn.MSELoss()
                    else:
                        loss_fn_distill = nn.CrossEntropyLoss()
                    optimizer_distill = torch.optim.Adam(model.parameters(), lr=0.001)
                    distill_model = distill_model.to(device)
                    valid_max_acc = 0
                    best_model = 0
                    val_cnt = 0
                    raw_tr_res = []
                    raw_val_res = []
                    num_of_batches = int(len(train_dataset) / TR)
                    start_time = time.time()

                    for batch_idx, (data, targets) in enumerate(train_loader):
                        model.train()
                        if not res_built:  # Fill first reservoir
                            for k in range(RS):
                                for i in range(num_classes):
                                    if int(targets[k]) == i and reservoir_class_counter[i] < items_per_class_TR:
                                        reservoir_idx[i * items_per_class_TR + reservoir_class_counter[i]] = (
                                                TR * batch_idx + k), data[k], targets[k]
                                        reservoir_class_counter[i] = reservoir_class_counter[
                                                                         i] + 1  # verifying only X samples per each class
                                if np.sum(reservoir_class_counter) == RS:
                                    res_built = True
                                    batch_infer_started = batch_idx
                        else:
                            # Important Note: We first train the current updated Reservoir, then train the new batch
                            #                 finally we update the reservoir
                            if batch_idx == batch_infer_started + 1:
                                ###
                                indices_r = [item[0] for item in reservoir_idx]
                                curr_reservoir_subset_data = torch.utils.data.Subset(train_dataset,
                                                                                     indices_r)  # create subset (ONLY FROM DATASET)!!!! NOT SUBSET
                                # *ATTENTION *
                                reservoir = DataLoader(curr_reservoir_subset_data, batch_size=RS, shuffle=False,
                                                       drop_last=True)  # create dedicated DataLoder    < MODIFY ******** BATCH SIZE RES
                                if not featuresON:
                                    # region - first batch if work with real pixels distillations
                                    distilled_res_DL = reservoir
                                    distilled_imgs_dict = initialize_distilled_representatives_imgs(distilled_res_DL,
                                                                                                    device,
                                                                                                    k=items_per_class_Dist)
                                    # save distilled first ones - initialization images
                                    curr_distill_images_name_first = "Distill_results\dist_imgs_after_" + str(batch_idx) + "_batches.png"
                                    curr_distill_images_name_first = os.path.join(res_path,curr_distill_images_name_first)
                                    save_distilled_images(distilled_imgs_dict, curr_distill_images_name_first)
                                    print('real images distillation\n')
                                    # endregion
                                else:
                                    # region - first batch if work with features
                                    distill_representatives = []
                                    for i in range(distill_resrvoir_s):
                                        distill_representatives.append(
                                            curr_reservoir_subset_data.dataset[
                                                curr_reservoir_subset_data.indices[i * int(TR / distill_resrvoir_s)]])
                                    print('features distillation\n')
                                    indices_res = [element[1] for element in distill_representatives]
                                    images_res = [element[0] for element in distill_representatives]
                                    indices_res = torch.tensor(indices_res, dtype=torch.long)
                                    image_res_tensor = torch.stack(images_res, dim=0)
                                    distilled_res = TensorDataset(image_res_tensor, indices_res)
                                    distilled_res_DL = DataLoader(distilled_res, batch_size=distill_resrvoir_s,
                                                                  shuffle=False, drop_last=True)

                                    distilled_features = distill_features_generic(distilled_res_DL, distill_model,
                                                                                  device, k=items_per_class_Dist)
                                    # test not to train on real images in first iteration: --------------
                                    dist_images = visualize_distilled_images_gen(distilled_features, class_names,
                                                                                 batch_idx, res_path)
                                    image_res_tensor = torch.stack(dist_images, dim=0)
                                    indices_res = labels_tensor.clone().detach().to(torch.long)
                                    # indices_res = torch.tensor(labels_tensor, dtype=torch.long)
                                    distilled_res = TensorDataset(image_res_tensor, indices_res)
                                    distilled_res_DL = DataLoader(distilled_res, batch_size=distill_resrvoir_s,
                                                                  shuffle=False, drop_last=True)
                                    # endregion

                            else:
                                # first create subset of the batch data:
                                indices_batch = []
                                for r in range(TR):
                                    indices_batch.append(batch_idx * TR + r)
                                curr_batch_subset_data = torch.utils.data.Subset(train_dataset,
                                                                                 indices_batch)  # create subset (ONLY FROM DATASET)!!!! NOT SUBSET
                                reservoir = DataLoader(curr_batch_subset_data, batch_size=RS,
                                                       shuffle=True, drop_last=True)
                                if not featuresON:
                                    # distilled imgs = dict of images and labels
                                    distilled_imgs_dict = update_distilled_images(reservoir, distilled_imgs_dict,
                                                                                  device,
                                                                                  learning_rate_distill, opt_distill,
                                                                                  cnn_type, err_scale)

                                    distilled_res_DL = dictionary_to_dataloader(distilled_imgs_dict,distill_resrvoir_s)
                                    print("debug...")
                                else:
                                    # region update distilled representatives - work with features
                                    if not distill_M == 'Wang':
                                        distilled_features = update_distilled_representatives(reservoir,
                                                                                              distilled_features,
                                                                                              distill_model,
                                                                                              optimizer_distill,
                                                                                              loss_fn_distill,
                                                                                              device)
                                    else:
                                        if not loss_f_distill == 'MSE':
                                            print('Wang Needs MSE loss Function, change in parameters section')
                                            break
                                        distilled_features = update_distilled_representatives_wang_like(reservoir,
                                                                                                        distilled_features,
                                                                                                        distill_model,
                                                                                                        device,
                                                                                                        learning_rate_distill)
                                    dist_images = visualize_distilled_images_gen(distilled_features, class_names,
                                                                                 batch_idx, res_path)
                                    image_res_tensor = torch.stack(dist_images, dim=0)
                                    indices_res = labels_tensor.clone().detach().to(torch.long)
                                    # indices_res = torch.tensor(labels_tensor, dtype=torch.long)
                                    distilled_res = TensorDataset(image_res_tensor, indices_res)
                                    distilled_res_DL = DataLoader(distilled_res, batch_size=distill_resrvoir_s,
                                                                  shuffle=False, drop_last=True)
                                    # endregion
                            model, tr_acc, tr_loss = train_reservoir_distill(max_n_epochs, distilled_res_DL,
                                                                             max_loss_degredation,
                                                                             batch_idx, RS,
                                                                             optimizer, device,
                                                                             model,
                                                                             criterion, scheduler, debug_f, loss_f)
                            f_txt.write(str('***** batch%04d/%04d |train_acc:%.4f [%%] |time_elapsed:%.2f[s] '
                                            '*****\n' % (batch_idx, num_of_batches, tr_acc, time.time() - start_time)))
                            raw_tr_res.append([algorithm, batch_idx, tr_acc, tr_loss, time.time() - start_time])

                            # if not batch_idx % val_gaps:
                            if (not batch_idx % val_gaps) or (batch_idx == np.floor(tr_size / TR) - 1):
                                # if not batch_idx % 1:
                                if val_cnt == 0:
                                    if featuresON:
                                        features_loader_v = preprocess_features_for_val_or_test(vd_t_loader,
                                                                                                distill_model, device,
                                                                                                val_batch_s, VD_T,
                                                                                                int(len(
                                                                                                    VD_T) / num_classes))
                                    else:
                                        features_loader_v = vd_t_loader
                                    val_cnt += 1
                                # save distilled images each time we do validation
                                curr_distill_images_name = "Distill_results\dist_imgs_after_" + str(batch_idx) + "_batches.png"
                                curr_distill_images_name = os.path.join(res_path, curr_distill_images_name)
                                save_distilled_images(distilled_imgs_dict, curr_distill_images_name)

                                model, val_acc, val_loss, best_model, idx, elaps_t, valid_max_acc = validate_distillation_r(
                                    model, val_cnt, loss_f, features_loader_v,
                                    valid_max_acc, start_time, criterion,
                                    tr_size, TR, val_gaps, batch_idx, num_of_batches,
                                    best_model, f_txt, device, num_classes)
                                raw_val_res.append([algorithm, idx, val_acc, val_loss, elaps_t])

                    # print to logfile the best validation accuracy the best model:
                    f_txt.write(str('==============================================================================\n'))
                    f_txt.write(str('***** Best validation_acc Recorded:%.4f [%%] *****\n' % valid_max_acc))
                    f_txt.write(str('==============================================================================\n'))
                    print('==============================================================================\n')
                    print('***** Best validation_acc Recorded:%.4f [%%] *****\n' % valid_max_acc)
                    # test the best model:
                    if featuresON:
                        features_loader_t = preprocess_features_for_val_or_test(test_loader, distill_model, device,
                                                                                test_batch_s, test_dataset_t,
                                                                                int(len(test_dataset_t) / num_classes))
                    else:
                        features_loader_t = test_loader
                    test_acc, test_loss = test_distillation_r(best_model, val_cnt, loss_f, features_loader_t,
                                                              start_time,
                                                              criterion, tr_size, TR, val_gaps, f_txt, device,
                                                              num_classes)

                    # region SAVE DATA and Plotting:
                    train_log = pd.DataFrame(raw_tr_res, columns=["algo", "idx", "tr_acc", "tr_loss", "elaps_t"])
                    val_log = pd.DataFrame(raw_val_res, columns=["algo", "idx", "val_acc", "val_loss", "elaps_t"])

                    plot_save_res_df(train_log, res_path, 'train_acc_loss', "TR_acc_and_loss_vs_batch_idx",
                                     "batch_idx", "acc [%]", "loss", "idx", "tr_acc", "tr_loss")
                    plot_save_res_df(val_log, res_path, 'validation_acc_loss', "VAL_acc_and_loss_vs_batch_idx",
                                     "batch_idx", "acc [%]", "loss", "idx", "val_acc", "val_loss")
                    plot_save_time_res_tr(train_log, res_path, 'Accumulated train_time_per_batch',
                                          "Accumulated Train Time vs batch idx", "batch_idx",
                                          "Time [sec]", "idx", "elaps_t")
                    plot_save_time_res_tr(val_log, res_path, 'Accumulated validation_time_per_batch',
                                          "Accumulated validation Time vs batch idx", "batch_idx",
                                          "Time [sec]", "idx", "elaps_t")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.exception("Fatal Error:")
        raise

# LIST OF TBD's:
# TBD 1:
