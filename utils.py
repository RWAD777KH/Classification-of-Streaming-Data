import random
import time
import datetime
import uuid

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from sklearn.tree import DecisionTreeClassifier
from statistics import mode
from sklearn.metrics import accuracy_score
from river import metrics
import os
# from river import datasets as Datasets
import pandas as pd
# from river.utils import Rolling
from river.metrics import Accuracy, CohenKappa, Rolling
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import mode
import torch.nn.functional as F
from itertools import islice
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from PIL import Image
import warnings

# Suppress warnings about data range for imshow
# warnings.filterwarnings('ignore', 'Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).')

def shuffle_tensor_data(data_set, size):
    """ Shuffle Tensor dataset
    # Arguments
        :param data_set: data set to shuffle
        :param size: size of the data set
        :return new_data_set: the shuffled dataset
    """
    new_data_set = data_set
    order = np.arange(size)
    random.shuffle(order)
    # orig_train_data = data_set.train_data   #for MNIST
    orig_train_data = data_set.data
    # orig_train_label = data_set.train_labels  #for MNIST
    orig_train_label = data_set.targets
    for i in np.arange(size):
        new_data_set.data[i] = orig_train_data[order[i]]  # MODIFY HERE according to DS Syntax
        new_data_set.targets[i] = orig_train_label[order[i]]  # MODIFY HERE according to DS Syntax
    return new_data_set


class sampleObj():
    def __init__(self, g_idx, img_t, true_class):
        self.g_idx = g_idx
        self.imgT = img_t
        self.true_class = true_class


class ReservoirSamplingCifar10(object):
    """ random sampling into per defined reservoir fo CIFAR10 data
    # Arguments
    :param max_size: max size for reservoir
    :return object: the reservoir
    """

    def __init__(self, max_size):
        self.samples = []  # list to store the sample #
        self.max_size = max_size
        self.i = 0

    def sampling(self, element):
        """ update reservoir after new sample
        :param element: The new sample/candidate to enter the reservoir
        :return None
        """
        size = len(self.samples)
        if size >= self.max_size:
            idx = random.randint(0, self.i - 1)     # Original
            # idx = (self.i - 1) % self.max_size        # test for changing all the reservoir
            if idx < size:                          # Original
            # if True:                                  # test for changing all the reservoir
                self.samples[idx] = element
        else:
            self.samples.append(element)
        self.i += 1


class ReservoirSamplingCifar10Allin(object):
    """ random sampling into per defined reservoir fo CIFAR10 data # FOR TEST only
    # Arguments
    :param max_size: max size for reservoir
    :return object: the reservoir
    """

    def __init__(self, max_size):
        self.samples = []  # list to store the sample #
        self.max_size = max_size
        self.i = 0

    def sampling(self, element):
        """ update reservoir after new sample
        :param element: The new sample/candidate to enter the reservoir
        :return None
        """
        size = len(self.samples)
        if size >= self.max_size:
            idx = random.randint(0, self.max_size - 1)
            self.samples[idx] = element
        else:
            self.samples.append(element)
        self.i += 1


# # COMMON utils ##:
def calc_validation_gaps(tr: int, data_set: str):
    """
    :param tr:
    :param data_set:
    :return: validation batch gaps
    """
    val_gaps = -1
    if data_set == "CIFAR10":
        if tr >= 2000:
            val_gaps = 1
        elif 1000 <= tr < 1999:
            val_gaps = 2
        elif 500 <= tr < 999:
            val_gaps = 4
        elif 100 <= tr < 499:
            val_gaps = 10
        elif tr < 100:
            val_gaps = 100
        return val_gaps
    else:
        return None


def make_dataloader_for_reservoirs(num_classes: int, item_per_class: int, reservoirs: list,
                                   device: str):
    """
    :param device:
    :param reservoirs:
    :param item_per_class:
    :param num_classes:
    :return: curr_res_raw
    """
    curr_res_images = []
    curr_res_labels = []

    for m in range(num_classes):
        for c in range(item_per_class):
            sample = reservoirs[m].samples[c]
            # curr_res_images.append(
            #     torch.tensor(sample.imgT.data / 255, dtype=torch.float, device=device))
            curr_res_images.append(sample.imgT)
            curr_res_labels.append(sample.true_class)

    curr_res_raw = TensorDataset(torch.stack(curr_res_images),
                                 torch.tensor(curr_res_labels, dtype=torch.long,
                                              device=device).view(-1))
    # curr_res = DataLoader(curr_res_raw, batch_size=res_train_batch_s, shuffle=False,
    #                       drop_last=True)
    return curr_res_raw


def train_reservoir_rand_sample(max_n_epochs: int, curr_res: DataLoader, max_loss_degredation: int, batch_idx, RS,
                                optimizer, device: str, base_model, criterion, scheduler, debug_flag):
    """
    :param max_n_epochs:
    :param curr_res:
    :param max_loss_degredation:
    :param batch_idx:
    :param RS:
    :param optimizer:
    :param device:
    :param base_model:
    :param criterion:
    :param scheduler
    :param debug_flag
    :return: base_model, tr_acc, tr_loss
    """
    tr_loss = 0.0
    tr_prev_loss = 0.0
    tr_acc = 0
    degradation_cnt = 0
    correct_tensor = 0
    total = 0
    for i in range(max_n_epochs):
        for batch_idx_t, (data_t, targets_t) in enumerate(curr_res):
            data_t, targets_t = data_t.to(device), targets_t.to(device)
            if debug_flag:
                present_data(data_t)
            # base_model.eval()  # just for 1 sample iteration
            out = base_model(data_t)
            # base_model.train()  # just for 1 sample iteration
            loss = criterion(out, targets_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * data_t.size(0)
            _, pred = torch.max(out, dim=1)
            total += targets_t.size(0)  ##<
            correct_tensor += (pred == targets_t).sum().item()
        # scheduler.step()
        tr_acc = 100 * correct_tensor / total
        tr_loss = tr_loss / RS
        if tr_loss > tr_prev_loss and i != 0:
            degradation_cnt += 1
        else:
            degradation_cnt = 0
        tr_prev_loss = tr_loss
        if degradation_cnt >= max_loss_degredation:
            break
        print("batch %d > epoch %d || train acc: %.8f ** train loss: %.10f ** deg cnt:"
              " %d" %
              (batch_idx, i, tr_acc, tr_loss, degradation_cnt))
        # current_lr = optimizer.param_groups[0]["lr"]
        # scheduler.step()
        # next_lr = optimizer.param_groups[0]["lr"]
    return base_model, tr_acc, tr_loss



def validate_model_rand_sample(vd_t_loader: DataLoader, valid_max_acc, batch_cnt, tr_size, start_time, val_gaps, TR,
                               base_model, optimizer, device: str, criterion, val_cnt, best_model, vd_len):
    """
    :param Val_loader:
    :param valid_max_acc:
    :param batch_cnt:
    :param tr_size:
    :param start_time:
    :param val_gaps:
    :param TR:
    :param base_model:
    :param optimizer:
    :param device:
    :param criterion:
    :param val_cnt:
    :return:
    """
    # valid_acc = 0
    valid_loss = 0.0
    val_cnt += 1
    total = 1e-16
    correct_tensor = 0
    # Validation loop
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(vd_t_loader):
            data, targets = data.to(device), targets.to(device)
            output = base_model(data)
            loss = criterion(output, targets)
            valid_loss += loss.item() * data.size(0)
            _, pred = torch.max(output.data, dim=1)  ##<
            total += targets.size(0)  ##<
            correct_tensor += (pred == targets).sum().item()
    valid_acc = 100 * correct_tensor / total  # the actual train_accuracy_calc
    valid_loss = valid_loss / vd_len

    if valid_acc > valid_max_acc:
        valid_max_acc = valid_acc
        best_model = base_model

    elapsed_time = (time.time() - start_time) / 3600  # in hours
    eta_mean = elapsed_time / val_cnt
    eta = eta_mean * int(np.floor((tr_size / TR) / val_gaps))
    print('******* batch %03d/%03d | valid_acc: %.f | valid_Loss: %.4f |'
          'time_elapsed/eta: %.2f[h]/%.2f[h] *******\n' % (
              batch_cnt, np.floor(tr_size / TR), valid_acc,
              valid_loss, elapsed_time, eta))

    return best_model, valid_acc, valid_loss, valid_max_acc


def test_model_rand_sample(test_loader: DataLoader, test_data_len, device: str, criterion, model):
    """
    :param test_loader:
    :param test_data_len:
    :return:
    """
    total_t_last = 1e-16
    test_loss_last = 0.0
    correct_tensor = 0
    for data_l, targets_l in test_loader:  # TEMP
        data_l, targets_l = data_l.to(device), targets_l.to(device)
        output = model(data_l)
        loss = criterion(output, targets_l)
        loss_r = float(loss.cpu().detach().numpy())
        test_loss_last += loss.item() * data_l.size(0)
        _, pred = torch.max(output, dim=1)
        # correct_tensor = pred.eq(targets.data.view_as(pred))  # indicates if the pred. is true
        correct_tensor += torch.sum(pred == targets_l).item()
        total_t_last += targets_l.size(0)
    test_acc_l = 100 * correct_tensor / total_t_last
    test_loss_last = test_loss_last / test_data_len
    return test_acc_l, test_loss_last


#  # EXPERIMENTAL FOR HOEFFDING and BIE ##
#  Consider deleting diffs that irrelevant  <<<<<
# class BatchClassifier_orig:
#     """ class for use in (BIE) Batch-Incremental Ensemble Classifier
#     initialize attributes
#     """
#
#     def __init__(self, window_size, max_models=100):
#         self.H = []
#         self.h = None
#         # TODO
#         self.window_size = window_size
#         self.max_models = max_models
#         self.init = False
#         self.X_batch = []
#         self.y_batch = []
#
#     def learn_one(self, x, y=None):
#         """
#         once new [window_size] data points are fed the model are trained and the models are appended it to the ensemble
#         until max of [max_models]
#         """
#         # self.X_batch.append(dict2numpy(x))
#         self.X_batch.append(x)
#         self.y_batch.append(y)
#         if len(self.X_batch) == self.window_size:
#             h = DecisionTreeClassifier()
#
#             h.fit(self.X_batch, self.y_batch)
#             self.H.append(h)
#
#             self.X_batch.clear()
#             self.y_batch.clear()
#         if len(self.H) > self.max_models:
#             self.H.pop(0)
#         return self
#
#     def predict_one(self, x):
#         """
#         prediction is made by every model in ensemble and outputs with majority rule.
#         """
#         if len(self.H) == 0:
#             return 0
#         preds = []
#         for model in self.H:
#             preds.append(model.predict([dict2numpy(x)]))
#             preds.append(model.predict([x]))
#         preds = [x[0] for x in preds]
#         return mode(preds)  # mode: Return the most common data point from discrete or nominal data

class BatchClassifierTest:  # <<< Test This:
    """Class for use in (BIE) Batch-Incremental Ensemble Classifier."""

    def __init__(self, window_size, max_models=100):
        self.H = []
        self.h = None
        self.window_size = window_size
        self.max_models = max_models
        self.init = False
        self.X_batch = []
        self.y_batch = []

    def learn_one(self, x, y=None):
        """
        Once new [window_size] data points are fed, the models are trained and appended to the ensemble
        until a maximum of [max_models].
        """
        self.X_batch.append(x)
        self.y_batch.append(y)
        if len(self.X_batch) == self.window_size:
            h = DecisionTreeClassifier()
            h.fit(self.X_batch, self.y_batch)
            self.H.append(h)

            self.X_batch.clear()
            self.y_batch.clear()

        if len(self.H) > self.max_models:
            self.H.pop(0)

        return self

    def predict_one(self, x):
        """
        Prediction is made by every model in the ensemble, and outputs are combined using majority rule.
        """
        if not self.H:
            return 0

        preds = [model.predict([x])[0] for model in self.H]
        return mode(preds).mode[0]  # Return the most common data point from discrete or nominal data



class BatchClassifierR:  # <<< Test This:
    """Class for use in (BIE) Batch-Incremental Ensemble Classifier."""

    def __init__(self, max_models=100):
        self.H = []
        self.h = None
        self.max_models = max_models
        self.init = False
        self.X_batch = []
        self.y_batch = []

    def learn_batch(self, x_data, y_label=None ):
        """
        Once new [window_size] data points are fed, the models are trained and appended to the ensemble
        until a maximum of [max_models].
        """
        # self.X_batch = x_data
        self.X_batch = x_data.reshape(x_data.shape[0], -1)
        self.y_batch = y_label

        h = DecisionTreeClassifier()
        h.fit(self.X_batch, self.y_batch)
        self.H.append(h)

        self.X_batch = []
        self.y_batch = []

        if len(self.H) > self.max_models:
            self.H.pop(0)

        return self

    def predict_batch_tr(self, x_data, y_batch, features_f):
        """
        Prediction is made by every model in the ensemble, and outputs are combined using majority rule.
        """
        if not self.H:
            return 0

        x_data_f = x_data.reshape(x_data.shape[0], -1)
        # preds = [model.predict([x_data])[0] for model in self.H]
        preds = [model.predict(x_data_f) for model in self.H]
        tot_correct_items = 0
        for item in preds:
            curr_preds_arr = item.reshape(item.shape[0],1)
            # curr_preds_arr = np.array(item)
            # curr_preds_arr = curr_preds_arr.reshape(curr_preds_arr.shape[0], 1)
            if features_f:
                y_batch = y_batch.reshape(y_batch.shape[0],1)
            correct_items = np.sum(np.array(curr_preds_arr) == y_batch)
            tot_correct_items += correct_items
        tr_acc = tot_correct_items/(len(y_batch)*len(preds))*100
        # print("Training Acc of Batch %d: %.3f" % (idx, tr_acc))
        return mode(preds).mode[0], tr_acc   # Return the most common data point from discrete or nominal data


def evaluate_bieR(model, x_batches, y_batches, num_batches, val_gaps, X_val_f, y_val, file, features_f):
    """
    :param model:
    :param x_batches:
    :param y_batches:
    :param num_batches:
    :param val_gaps:
    :param X_val_f:
    :param y_val:
    :param file:
    :return:
    """
    raw_results = []
    raw_val_results = []
    model_name = model.__class__.__name__
    start_t = time.time()
    idx = 0
    for x_batch, y_batch in zip(x_batches, y_batches):
        idx += 1
        curr_train_t_start = time.time()
        # learn
        start = time.time()
        model.learn_batch(x_batch, y_batch)
        tr_preds, tr_acc = model.predict_batch_tr(x_batch, y_batch, features_f)
        curr_train_t = time.time() - curr_train_t_start
        # if idx < len(model.H):
        #     tr_acc = -1
        raw_results.append([model_name, idx, tr_acc, curr_train_t])

        if not idx % val_gaps:   # evaluate each
            curr_v_t_start = time.time()
            tot_val_acc = 0
            for item in model.H:
                y_val_pred = item.predict(X_val_f)
                accuracy = accuracy_score(y_val, y_val_pred)
                tot_val_acc += accuracy
            val_acc = tot_val_acc/len(model.H)*100
            curr_v_t = time.time() - curr_v_t_start
            print("----- batch%04d/%04d |validation_acc:%.4f [%%]|time_elapsed:%.2f[h] -----" % (idx, num_batches, val_acc,
                                                                                         (time.time() - start_t)
                                                                                         / 3600))
            file.write(str('---------------------------------------------------------------------------\n'))
            file.write(str('***** batch%04d/%04d |validation_acc:%.4f [%%] |time_elapsed:%.2f[h] *****\n' % (idx, num_batches,
                                                                                                       val_acc,
                                                                                                       (time.time() -
                                                                                                        start_t)
                                                                                                       / 3600)))
            file.write(str('---------------------------------------------------------------------------\n'))
            raw_val_results.append([model_name, idx, val_acc, curr_v_t])

        print('***** batch%04d/%04d |train accuracy: %.3f [%%] | time_elapsed:%.2f[h] *****' % (idx, num_batches,
                                                                                               tr_acc, (time.time() -
                                                                                                        start_t)
                                                                                               / 3600))
        file.write(str('***** batch%04d/%04d | train accuracy: %.3f [%%] | time_elapsed:%.2f[h] *****\n' % (idx, num_batches,
                                                                                               tr_acc, (time.time() -
                                                                                                        start_t)
                                                                                                        / 3600)))
    train_log = pd.DataFrame(raw_results, columns=['model', 'batch_id', 'tr_acc', 'train_time'])
    val_log = pd.DataFrame(raw_val_results, columns=['model', 'idx', 'val_acc', 'validation_time'])
    return model, train_log, val_log
    # return pd.DataFrame(raw_results, columns=['model', 'id', 'acc', 'acc_roll', 'kappa', 'kappa_roll', 'training_time',
    #                                           'testing_time'])


def print_progress(sample_id, acc, kappa, training_time, testing_time):
    """
    for HoeffdingTree and BIE
    prints the progress afetr [sample_id] samples
    """
    print(f'Samples processed: {sample_id}')
    print(acc)
    print(kappa)
    print("total train time:", training_time)
    print("total testing time:", testing_time)


def evaluate(stream, model, n_wait=1000, verbose=False):
    acc = metrics.Accuracy()
    acc_rolling = metrics.Rolling(metric=metrics.Accuracy(), window_size=n_wait)
    kappa = metrics.CohenKappa()
    kappa_rolling = metrics.Rolling(metric=metrics.CohenKappa(), window_size=n_wait)
    raw_results = []
    model_name = model.__class__.__name__
    list_of_trainings = []
    list_of_testings = []

    training_time = 0
    testing_time = 0
    start_t = time.time()
    for i, (x, y) in enumerate(stream):
        # Predict
        start = time.time()
        y_pred = model.predict_one(x)
        end = time.time()
        testing_time = end - start
        list_of_testings.append(testing_time)

        # Update metrics and results
        acc.update(y_true=y, y_pred=y_pred)
        acc_rolling.update(y_true=y, y_pred=y_pred)
        kappa.update(y_true=y, y_pred=y_pred)
        kappa_rolling.update(y_true=y, y_pred=y_pred)
        if i % n_wait == 0 and i > 0:
            if verbose:
                print_progress(i, acc, kappa)
            raw_results.append(
                [model_name, i, acc.get(), acc_rolling.get(), kappa.get(), kappa_rolling.get(), training_time,
                 testing_time])
        # Learn (train)
        start = time.time()
        model.learn_one(x, y)
        end = time.time()
        training_time = end - start
        list_of_trainings.append(training_time)

        if not i % 100:
            print('***** batch%04d/50000 | time_elapsed:%.2f[h] *****' % (i, (time.time() - start_t) / 3600))
    print_progress(i, acc, kappa, sum(list_of_trainings), sum(list_of_testings))
    return pd.DataFrame(raw_results, columns=['model', 'id', 'acc', 'acc_roll', 'kappa', 'kappa_roll', 'training_time',
                                              'testing_time'])


def setup_scenario_files(algo: str, tr='N_A', rs='N_A', isDistill ='N_A'):
    # # >>>>>>> create folders for the scenario at hand <<<<<<<#
    project_dir = os.getcwd()
    d = datetime.date.isoformat(datetime.date.today())
    hash = uuid.uuid4().hex
    hash_sh = hash[0:9]
    date_hashed = d + '_' + hash_sh  # date time with unique hash
    scenario_file_name = (
            'TEST_' + algo + '_Tr_BS_' + str(tr) + 'Res_S_' + str(rs) + '_DATE_'
            + date_hashed)
    res_path = os.path.join(project_dir, scenario_file_name)
    if not os.path.exists(res_path):
        os.mkdir(res_path)
        print("results Directory was created: ", res_path)
    else:
        print("result folder already exists: ", res_path)
    distill_dir_name = 'Distill_results'
    distill_path = os.path.join(res_path, distill_dir_name)
    if not os.path.exists(distill_path):
        os.mkdir(distill_path)
    return res_path


def plot_save_hist(subset, type_f: str = None, save_path: str = None, debug_f: bool = False):
    """
    :param subset:
    :param type_f:
    :param save_path:
    :param debug_f:
    :return: None
    """

    if type_f == 'Train' or type_f == 'Validation':
        df = pd.DataFrame(subset.dataset.targets)
        indices_int = np.array(subset.indices, dtype=int)
        if debug_f:
            print(type_f)
            print(subset.dataset.transform)
            print(subset.dataset.targets[0])
            plt.imshow(subset.dataset.data[0])
            plt.show(block=True)

    elif type_f == 'Test':
        df = pd.DataFrame(subset.targets)
        indices_int = np.array(range(10000), dtype=int)
        if debug_f:
            print(type_f)
            print(subset.transform)
            print(subset.targets[0])
            plt.imshow(subset.data[0])
            plt.show(block=True)
    df_to_plot = df.iloc[indices_int]
    fig, ax = plt.subplots()
    ax.hist(df_to_plot[:][0], bins=10, color='skyblue', edgecolor='black')
    ax.set_title(type_f + '_histogram')
    ax.set_ylabel('num of representatives')
    ax.set_xlabel('class')
    fig_name = 'hist_' + type_f
    fig_name_s = os.path.join(save_path, fig_name)
    plt.savefig(fig_name_s, dpi=1200)
    plt.close(fig)


def load_CIFAR_tensors(tr_size, train_transform, test_transform, resize_transform_32=None):
    """
    :param tr_size:
    :param train_transform:
    :param test_transform:
    :param resize_transform_32:
    :return:
    """

    train_ds_t_full = datasets.CIFAR10(root='data', train=True, transform=train_transform, download=True)
    train_ds_t_full = shuffle_tensor_data(train_ds_t_full, tr_size)  # do the shuffle :)
    # train_ds_t_full[i] already shuffled ! :)
    test_dataset_t = datasets.CIFAR10(root='data', train=False, transform=test_transform, download=True)

    test_dataset_t_orig_size = datasets.CIFAR10(root='data', train=False, transform=transforms.ToTensor(),
                                                download=True)
    if not (resize_transform_32 is None):
        test_dataset_t_resized = datasets.CIFAR10(root='data', train=True, transform=resize_transform_32, download=True)
    else:
        test_dataset_t_resized = None
    # D_Hyp['d_test'] = test_dataset_t_resized

    return train_ds_t_full, test_dataset_t, test_dataset_t_orig_size, test_dataset_t_resized


class ResNet34ModifiedLastLayer(nn.Module):
    def __init__(self):
        super(ResNet34ModifiedLastLayer, self).__init__()
        # load Weights:
        weights = ResNet34_Weights.IMAGENET1K_V1
        resnet = resnet34(weights)  # Pre-Trained
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        # Forward pass up to the desired layer (one layer before the last layer)
        x = self.features(x)
        return x


def CreateFeaturesLabelsMatrixOfLayerBeforeLast(dataset, model, device: str):
    """
    :param dataset:
    :param model:
    :param device:
    :return:
    """
    full_arr_features_labels = []
    start_t = time.time()
    for batch_idx, (data, targets) in enumerate(dataset):
        data, targets = data.to(device), targets.to(device)
        out = model(data)
        out_features_array = out.cpu().detach().numpy().reshape(-1)
        target_np = targets.cpu().detach().numpy()
        array_line = np.append(out_features_array, target_np)
        if batch_idx == 0:
            full_arr_features_labels = array_line
        else:
            full_arr_features_labels = np.vstack([full_arr_features_labels, array_line])
        if not batch_idx % 100:
            print('***** batch%04d/%04d | time_elapsed:%.2f[h] *****' % (batch_idx, len(dataset),(time.time()-start_t)
                                                                         / 3600))
    return full_arr_features_labels


def present_data(data):
    """
    :param data:
    :return:
    """
    plt.show()
    plt.imshow(data[0].cpu().detach().permute(1, 2, 0))
    plt.imshow(data[20].cpu().detach().permute(1, 2, 0))
    plt.imshow(data[40].cpu().detach().permute(1, 2, 0))
    plt.imshow(data[90].cpu().detach().permute(1, 2, 0))


def write_log_header(file, dataset: str, algorithm: str, train_b_size: int, val_b_size: int,test_b_size: int,
                     res_size=None, val_gaps=None, learning_rate=None, res_train_batch_s=None, max_n_epochs=None,
                     distill_m=None, featuresON=False, learning_rate_distill=None, opt_distill=None,
                     cnn_type=None):
    """

    :param file:
    :param res_size:
    :param dataset:
    :param algorithm:
    :param val_gaps:
    :param learning_rate=None
    :param train_b_size:
    :param val_b_size:
    :param test_b_size:
    :param res_train_batch_s
    :param max_n_epochs
    :param distill_m
    :return:
    """

    file.write('Results Summary Log\n\n')
    file.write(str('===================================================\n'))
    file.write(str('Data Base Under Test: %s\n' % dataset))
    file.write(str('Algorithm Tested    :  %s\n' % algorithm))
    if featuresON is True:
        file.write(str('%s using: Features\n' % algorithm))
    elif featuresON is False:
        file.write(str('%s using: Real Images\n' % algorithm))
    if distill_m is not None:
        file.write(str('Distillation Method: %s\n' % distill_m))
    if val_gaps is not None:
        file.write(str('validation gaps [batches]: %s\n' % str(val_gaps)))
    if res_size is not None:
        file.write(str('Reservoir Size  : %d\n' % res_size))
    file.write(str('Batch Train Size: %d\n' % train_b_size))
    file.write(str('Batch Eval Size : %d\n' % val_b_size))
    file.write(str('Batch test Size : %d\n' % test_b_size))
    if res_train_batch_s is not None:
        file.write(str('Batch Reservoir Train size: %d\n' % res_train_batch_s))
    if max_n_epochs is not None:
        file.write(str('Max Number of Train Epochs: %d\n' % max_n_epochs))
    if learning_rate is not None:
        file.write(str('Resnet LR: %06f\n' % learning_rate))
    if learning_rate_distill is not None:
        file.write(str('Distillation CNN LR: %06f\n' % learning_rate_distill))
    if opt_distill is not None:
        file.write(str('CNN Distillation Optimizer Type: %s\n' % opt_distill))
    if cnn_type is not None:
        file.write(str('CNN Type: %s\n' % cnn_type))
    file.write(str('===================================================\n'))


def predict_and_calc_acc_tr_RForest(model, X_tr_f, y_Tr_f, idx, num_batches, file, start_t):
    pred = model.predict(X_tr_f)
    correct_cnt = np.sum(pred == y_Tr_f)
    tr_acc = correct_cnt/len(y_Tr_f)*100
    tr_time = time.time() - start_t
    print(("----- batch%04d/%04d |train_acc:%.4f [%%]|time_elapsed:%.4f[s] -----" % (idx, num_batches, tr_acc, tr_time
                                                                                     )))
    file.write(str('***** batch%04d/%04d |train_acc:%.4f [%%] |time_elapsed:%.4f[s] *****\n' % (
            idx, num_batches, tr_acc, tr_time)))

    return tr_acc, tr_time


def predict_and_calc_acc_val_RForest(model, x_f, y_f, idx, num_batches, file, best_model, best_val_acc, start_t
                                     ,batch_size:int = 100):

    correct_cnt = 0
    for i in range(0, len(x_f), batch_size):
        x_f_batch = x_f[i:i + batch_size]
        y_f_batch = y_f[i:i + batch_size]
        pred = model.predict(x_f_batch)
        correct_cnt += np.sum(pred == y_f_batch)
        print("validation in progress... %.4f" % (i/len(y_f)))
    acc = correct_cnt/len(y_f)*100
    elapsed_t = time.time() - start_t
    print("----------------------------------------------------------------------------------------------")
    print(("----- batch%04d/%04d |validation_acc:%.4f [%%]|time_elapsed:%.4f[s] -----" % (idx, num_batches, acc,
                                                                                          elapsed_t)))
    print("----------------------------------------------------------------------------------------------")
    file.write(str('---------------------------------------------------------------------------\n'))
    file.write(str('***** batch%04d/%04d |validation_acc:%.4f [%%] |time_elapsed:%.4f[s] *****\n' % (
            idx, num_batches, acc, elapsed_t)))
    file.write(str('---------------------------------------------------------------------------\n'))
    if acc > best_val_acc:
        best_val_acc = acc
        best_model = model
    return acc, elapsed_t, best_model, best_val_acc


def predict_and_calc_acc_test_RForest_lastM(model, x_f, y_f, file, batch_size: int = 100):
    start_t = time.time()
    correct_cnt = 0
    for i in range(0, len(x_f), batch_size):
        x_f_batch = x_f[i:i + batch_size]
        y_f_batch = y_f[i:i + batch_size]
        pred = model.predict(x_f_batch)
        correct_cnt += np.sum(pred == y_f_batch)
        print("Last Model Testing in progress... %.4f" % (i/len(y_f)))
    acc = correct_cnt/len(y_f)*100
    elapsed_t = time.time() - start_t
    print("----------------------------------------------------------------------------------------------")
    print(("----- Last Model Test acc:%.4f [%%]|time_elapsed:%.4f[s] -----" % (acc, elapsed_t)))
    print("----------------------------------------------------------------------------------------------")
    file.write(str('---------------------------------------------------------------------------\n'))
    file.write(str('***** Last Model Test acc:%.4f [%%] |time_elapsed:%.4f[s] *****\n' % (acc, elapsed_t)))
    file.write(str('---------------------------------------------------------------------------\n'))


def predict_and_calc_acc_test_RForest_BestM(model, x_f, y_f, file, batch_size: int = 100):
    start_t = time.time()
    correct_cnt = 0
    for i in range(0, len(x_f), batch_size):
        x_f_batch = x_f[i:i + batch_size]
        y_f_batch = y_f[i:i + batch_size]
        pred = model.predict(x_f_batch)
        correct_cnt += np.sum(pred == y_f_batch)
        print("Best Model Testing in progress... %.4f" % (i/len(y_f)))
    acc = correct_cnt/len(y_f)*100
    elapsed_t = time.time() - start_t
    print("----------------------------------------------------------------------------------------------")
    print(("----- Best Test Model acc:%.4f [%%]|time_elapsed:%.4f[s] -----" % (acc, elapsed_t)))
    print("----------------------------------------------------------------------------------------------")
    file.write(str('---------------------------------------------------------------------------\n'))
    file.write(str('***** Best Model Test acc:%.4f [%%] |time_elapsed:%.4f[s] *****\n' % (acc, elapsed_t)))
    file.write(str('---------------------------------------------------------------------------\n'))


# Region - Classes & Functions for out Distillation approach:

# Base Neural Network for feature extraction with 224x224 input size
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),  # Assuming 10 classes for CIFAR-10
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# def distill_features(dataloader, model, device):
#     model.eval()  # Ensure the model is in evaluation mode
#     features_per_class = {i: torch.zeros((256 * 6 * 6,)) for i in range(10)}  # Adjusted for feature size
#     counts_per_class = {i: 0 for i in range(10)}
#
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             features = model.features(inputs)  # Use the feature part of the model
#             features = torch.flatten(model.avgpool(features), 1)  # Flatten the features
#
#             for feature, label in zip(features, labels):
#                 features_per_class[label.item()] += feature.cpu()
#                 counts_per_class[label.item()] += 1
#
#     # Average the features for each class
#     for label in features_per_class:
#         features_per_class[label] /= counts_per_class[label]
#
#     return features_per_class
def distill_features_generic(dataloader, model, device, k=5):
    model.eval()  # Ensure the model is in evaluation mode

    features_per_class = {}
    current_indices = {}  # Track the current index for each class

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = model.features(inputs)  # Use the feature part of the model
            features = torch.flatten(model.avgpool(features), 1)  # Flatten the features

            for feature, label in zip(features, labels):
                label_item = label.item()
                # Initialize storage for new classes
                if label_item not in features_per_class:
                    features_per_class[label_item] = [torch.zeros_like(feature.cpu()) for _ in range(k)]
                    current_indices[label_item] = 0
                # Updating the features
                # Normalize the feature vector before storing
                normalized_feature = feature.cpu() / torch.norm(feature.cpu(), p=2)

                # Store and cyclically replace feature vectors
                idx = current_indices[label_item]
                features_per_class[label_item][idx] = normalized_feature

                # Update the current index, wrapping around cyclically
                current_indices[label_item] = (idx + 1) % k

    # convert list to dicionary
    features_per_class_dict = {key: torch.stack(tensor) for key, tensor in features_per_class.items()}

    # No need to normalize again outside the loop since we're normalizing each vector before storing
    return features_per_class_dict

def update_distilled_representatives(dataloader, distilled_features, model, optimizer, loss_fn, device):
    model.train()  # Ensure the model is in training mode for feature extraction

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model.features(inputs)  # Extract features
        outputs = torch.flatten(model.avgpool(outputs), 1)  # Flatten the features
        optimizer.zero_grad()

        losses = []
        # Update distilled features based on new data
        for output, label in zip(outputs, labels):
            distilled_feature = distilled_features[label.item()].to(inputs.device)
            loss = loss_fn(output, distilled_feature)
            losses.append(loss)

        # Average the loss from all examples in the batch
        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()  # Backpropagate based on the average loss
        optimizer.step()  # Update model parameters


        # Update the distilled features with new features
        with torch.no_grad():
            for output, label in zip(outputs, labels):
                # Assuming a simple update rule to average the features; adjust as necessary
                feature_update = (distilled_features[label.item()] + output.cpu()) / 2
                distilled_features[label.item()] = feature_update

    return distilled_features


def update_distilled_representatives_wang_like(dataloader, distilled_features, model, device, learning_rate):
    model.eval()  # Set model to evaluation mode to extract features without dropout effects

    # Assuming distilled_features is a dictionary with class keys and synthetic feature tensors as values
    synthetic_features = {label: features.clone().detach().requires_grad_(True) for label, features in
                          distilled_features.items()}


    # Define an optimizer for the synthetic features. Here, we use a single optimizer for simplicity.
    params_to_optimize = [features for features in synthetic_features.values()]
    optimizer = torch.optim.Adam(params_to_optimize, lr=learning_rate)

    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        with torch.no_grad():
            real_features = model.features(inputs)  # Extract features
            real_features = torch.flatten(model.avgpool(real_features), 1)  # Flatten the features

        optimizer.zero_grad()

        # Calculate loss for each class's synthetic features compared to real features
        losses = []
        for _, synthetic_feature_tensor in synthetic_features.items():
            # If synthetic_feature_tensor is already a tensor of shape [num_vectors, feature_length],
            # directly calculate the mean across the first dimension (num_vectors)
            synthetic_feature_avg = torch.mean(synthetic_feature_tensor, dim=0).to(device)

            # Ensure synthetic_feature_avg is properly broadcasted:
            synthetic_feature_repeated = synthetic_feature_avg.unsqueeze(0).repeat(real_features.size(0), 1)

            loss = F.mse_loss(synthetic_feature_repeated, real_features)
            losses.append(loss)

        # Average the losses across all classes and backpropagate
        total_loss = torch.mean(torch.stack(losses))
        total_loss.backward()
        optimizer.step()

    # Update the distilled_features with the updated synthetic features
    # updated_distilled_features = {label: features.detach().clone() for label, features in synthetic_features.items()}
    #  test:
    # upd_org = {label: features.detach().clone() for label, features in synthetic_features.items()}
    updated_distilled_features = {label: features.detach().clone() for label, features in synthetic_features.items()}
    return updated_distilled_features


def update_features_wang(distilled_features, output, label):
    print('Wang')

# def visualize_distilled_images(distilled_features, class_names, batch_idx, res_path):
#     """
#     Visualize distilled features by interpolating them to a desired shape.
#     Each feature vector is visualized as one RGB image per class.
#     """
#     # Determine subplot grid size
#     n = len(distilled_features)
#     fig, axs = plt.subplots(1, n, figsize=(n * 5, 5))
#     list_of_distilled_images = []
#     for i, (label, feature) in enumerate(distilled_features.items()):
#         # Assuming feature is a 1D Tensor of 9216 elements
#         # First, reshape to (3, 48, 64) as a closer approximation to desired aspect ratio
#         feature_reshaped = feature.view(3, 48, 64)
#
#         # Interpolate the reshaped feature to the desired size (3, 224, 224)
#         feature_upscaled = F.interpolate(feature_reshaped.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
#
#         # Normalize the upscaled feature for visualization [0-1]
#         feature_upscaled = (feature_upscaled - feature_upscaled.min()) / (feature_upscaled.max() - feature_upscaled.min())
#         # feature_upscaled = feature_upscaled * 255 # another normalization range [0 to 255]
#         # feature_upscaled = feature_upscaled * 6 - 3  # # another normalization range [-3 to 3]
#
#         # Convert to the desired floating-point dtype
#         feature_upscaled = feature_upscaled.to(torch.float32)
#
#         # Convert the tensor to a numpy array for visualization with matplotlib
#         feature_upscaled_np = feature_upscaled.numpy().transpose(1, 2, 0)  # Change from (C, H, W) to (H, W, C) for plotting
#
#         # Visualize the image
#         axs[i].imshow(feature_upscaled_np)
#         axs[i].set_title(class_names[label])
#         axs[i].axis('off')
#         list_of_distilled_images.append(feature_upscaled)
#
#     plt.tight_layout()
#     plt.show(block = False)
#     title = str('Distill after %d' % batch_idx)
#     plt.suptitle(title)
#     fig_name = 'Distillation_Imgs_after' + str(batch_idx)
#     fig_name_s = os.path.join(res_path,'Distill_results', fig_name)
#     plt.savefig(fig_name_s, dpi=1200)
#     plt.close("all")
#     return list_of_distilled_images

def visualize_distilled_images_gen(distilled_features, class_names, batch_idx, res_path):
    """
    Visualize and save a grid of distilled features, one image per class representative,
    compiled into a single image (10xN grid for 10 classes and N representatives each).
    Returns a list of individual image tensors.
    """
    # Number of classes and representatives per class (N)
    num_classes = len(class_names)
    N = len(distilled_features[next(iter(distilled_features))])  # Number of representatives
    # Prepare an empty list to collect all feature images as tensors
    list_of_distilled_images = []
    for label, features_list in distilled_features.items():
        for feature in features_list:
            # Reshape and upscale each feature
            feature_reshaped = feature.view(3, 48, 64)
            feature_upscaled = F.interpolate(feature_reshaped.unsqueeze(0), size=(224, 224), mode='bilinear',
                                             align_corners=False).squeeze(0)
            # Normalize
            feature_upscaled = (feature_upscaled - feature_upscaled.min()) / (
                        feature_upscaled.max() - feature_upscaled.min())
            # feature_upscaled = feature_upscaled * 255  # another normalization range [0 to 255]
            list_of_distilled_images.append(feature_upscaled)

    # Create a grid of images
    # nrow should be the number of classes if N=1, otherwise it should be N
    nrow = num_classes if N == 1 else N
    grid_img = make_grid(list_of_distilled_images, nrow=nrow, padding=10, normalize=False, pad_value=255)

    # Convert grid to a NumPy array for plotting - python do clip range grid image for [0 - 1]
    np_grid_img = grid_img.numpy().transpose(1, 2, 0)
    np_grid_img = np.clip(np_grid_img,0 , 1)
    # Plotting
    fig, ax = plt.subplots(figsize=(num_classes * 2, 2))  # Adjust the figure size as needed
    ax.imshow(np_grid_img)
    ax.axis('off')

    # Add class names as labels below the grid
    # if N == 1:
    for idx, class_name in enumerate(reversed(class_names)):
        # plt.figtext(x=(idx + 0.5) / num_classes, y=0, s=class_name, ha='center', va='bottom')
        plt.figtext(x=0.4, y=((idx+1)*0.8 + 0.3)/ num_classes, s=class_name, ha='left', va='center', rotation_mode ='anchor')

    plt.title('Distilled Features Grid')
    fig_name = f'Distilled_Features_Grid_after_{batch_idx}.png'
    # fig_name = f'Distilled_Features_Grid_after_{batch_idx}.pdf'
    fig_name_s = os.path.join(res_path,'Distill_results', fig_name)
    plt.savefig(fig_name_s, dpi=1200, bbox_inches='tight')
    # plt.savefig(fig_name_s, dpi=1200, format='pdf', bbox_inches='tight')
    plt.close()

    # Return the list of individual upscaled feature tensors
    return list_of_distilled_images


def visualize_features_images_for_val_test(distilled_features, debug_f=False):
    """
    Visualize distilled features by interpolating them to a desired shape.
    Each feature vector is visualized as one RGB image per class.
    """
    list_of_distilled_images = []
    list_of_labels = []
    for k in range(len(distilled_features)):
        for j in range(len(distilled_features[k])):
            label = next(islice(distilled_features, k, len(distilled_features)))
            # Assuming feature is a 1D Tensor of 9216 elements
            # First, reshape to (3, 48, 64) as a closer approximation to desired aspect ratio
            # feature_reshaped = feature.view(3, 48, 64)
            feature_reshaped = distilled_features[k][j].view(3, 48, 64)
            # Interpolate the reshaped feature to the desired size (3, 224, 224)
            feature_upscaled = F.interpolate(feature_reshaped.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

            # Normalize the upscaled feature for visualization [0-1]
            feature_upscaled = (feature_upscaled - feature_upscaled.min()) / (feature_upscaled.max() - feature_upscaled.min())
            # feature_upscaled = feature_upscaled * 255 # another normalization range [0 to 255]
            # feature_upscaled = feature_upscaled * 6 - 3 # # another normalization range [-3 to 3]
            # Convert the tensor to a numpy array for visualization with matplotlib
            feature_upscaled_np = feature_upscaled.numpy().transpose(1, 2, 0)  # Change from (C, H, W) to (H, W, C) for plotting

            # Visualize the image for debug
            if debug_f:
                plt.imshow(feature_upscaled_np)
                plt.title('debug test image')
                plt.axis('off')
                plt.close("all")

            list_of_distilled_images.append(feature_upscaled)
            list_of_labels.append(label)
    return list_of_distilled_images, list_of_labels


def preprocess_features_for_val_or_test(orig_dataloader: DataLoader, model, device, batch_size, orig_dataset, ipc:int):
    """
    :param orig_dataloader:
    :param model:
    :param device:
    :param batch_size:
    :param orig_dataset:
    :param ipc:
    :return:
    """
    distilled_features = distill_features_generic(orig_dataloader, model, device, ipc)
    feature_images, feature_labels = visualize_features_images_for_val_test(distilled_features, False)
    dist_images_tensor = torch.stack(feature_images, dim=0)
    indices_res = torch.tensor(feature_labels, dtype=torch.long) #  orig
    # indices_res = feature_labels.clone().detach().to(torch.long) # tested option 1
    dataset_ordered = TensorDataset(dist_images_tensor, indices_res)
    dataset_s = shuffle_distilled_tensor_data(dataset_ordered, len(dataset_ordered))
    dataloader = DataLoader(dataset_s, batch_size=batch_size, shuffle=False, drop_last=True)
    return dataloader


def distill_features_generic(dataloader, model, device, k=5):
    model.eval()  # Ensure the model is in evaluation mode

    features_per_class = {}
    current_indices = {}  # Track the current index for each class

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = model.features(inputs)  # Use the feature part of the model
            features = torch.flatten(model.avgpool(features), 1)  # Flatten the features

            for feature, label in zip(features, labels):
                label_item = label.item()
                # Initialize storage for new classes
                if label_item not in features_per_class:
                    features_per_class[label_item] = [torch.zeros_like(feature.cpu()) for _ in range(k)]
                    current_indices[label_item] = 0
                # Updating the features
                # Normalize the feature vector before storing
                normalized_feature = feature.cpu() / torch.norm(feature.cpu(), p=2)

                # Store and cyclically replace feature vectors
                idx = current_indices[label_item]
                features_per_class[label_item][idx] = normalized_feature

                # Update the current index, wrapping around cyclically
                current_indices[label_item] = (idx + 1) % k

    # convert list to dicionary
    features_per_class_dict = {key: torch.stack(tensor) for key, tensor in features_per_class.items()}

    # No need to normalize again outside the loop since we're normalizing each vector before storing
    return features_per_class_dict


def shuffle_distilled_tensor_data(data_set, size):
    """ Shuffle Tensor dataset For Customized Datasets
    # Arguments
        :param data_set: data set to shuffle
        :param size: size of the data set
        :return new_data_set: the shuffled dataset
    """
    new_data_set = data_set
    order = np.arange(size)
    random.shuffle(order)
    orig_train_data = data_set.tensors[0]
    orig_train_label = data_set.tensors[1]
    for i in np.arange(size):
        new_data_set.tensors[0][i] = orig_train_data[order[i]]  # MODIFY HERE according to DS Syntax
        new_data_set.tensors[1][i] = orig_train_label[order[i]]  # MODIFY HERE according to DS Syntax
    return new_data_set

def train_reservoir_distill(max_n_epochs: int, curr_res: DataLoader, max_loss_degredation: int, batch_idx, RS,
                                optimizer, device: str, base_model, criterion, scheduler, debug_flag, loss_f=False):
    """
    :param max_n_epochs:
    :param curr_res:
    :param max_loss_degredation:
    :param batch_idx:
    :param RS:
    :param optimizer:
    :param device:
    :param base_model:
    :param criterion:
    :param scheduler
    :param debug_flag
    :param MSE_f
    :return: base_model, tr_acc, tr_loss
    """
    tr_loss = 0.0
    tr_prev_loss = 0.0
    tr_acc = 0
    degradation_cnt = 0
    correct_tensor = 0
    total = 1e-16
    for i in range(max_n_epochs):
        for batch_idx_t, (data_t, targets_t) in enumerate(curr_res):
            data_t, targets_t = data_t.to(device), targets_t.to(device)
            if loss_f == 'MSE':
                targets_t_MSE = targets_t.float()
                targets_t_MSE = torch.nn.functional.one_hot(targets_t.long(), num_classes=10).float().to(device)
            if debug_flag:
                present_data(data_t)
            # base_model.eval()  # just for 1 sample iteration
            out = base_model(data_t)
            if loss_f == 'MSE':
                loss = criterion(out, targets_t_MSE)
            else:
                loss = criterion(out, targets_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * data_t.size(0)
            _, pred = torch.max(out, dim=1)
            total += targets_t.size(0)  ##<
            correct_tensor += (pred == targets_t).sum().item()
        # scheduler.step()
        tr_acc = 100 * correct_tensor / total
        tr_loss = tr_loss / RS
        if ((tr_loss > tr_prev_loss) or tr_loss == 0) and i != 0: # was : if tr_loss > tr_prev_loos and i != 0:
            degradation_cnt += 1
        else:
            degradation_cnt = 0
        tr_prev_loss = tr_loss
        if degradation_cnt >= max_loss_degredation:
            break
        print("batch %d > epoch %d || train acc: %.8f ** train loss: %.10f ** deg cnt:"
              " %d" %
              (batch_idx, i, tr_acc, tr_loss, degradation_cnt))
        # scheduler.step()
    return base_model, tr_acc, tr_loss

def validate_distillation_r(model, val_cnt, loss_f, data_loader, valid_max_acc, start_time, criterion, tr_size,
                            TR, val_gaps, batch_idx, num_of_batches, best_model, f_txt, device, num_classes):

    model.eval()
    valid_loss = 0.0
    val_cnt += 1
    total = 1e-16
    correct_tensor = 0
    with torch.no_grad():
        for batch_idx_v, (data_v, targets_v) in enumerate(data_loader):
            data_v, targets_v = data_v.to(device), targets_v.to(device)
            if loss_f == 'MSE':
                targets_v = targets_v.float()  # if we use an MSEloss
            output = model(data_v)
            # safity - if there NaN values - replace with 1e-15
            # nan_count = torch.isnan(output.data).sum()
            output = torch.where(torch.isnan(output), torch.tensor(1e-15,
                                                                   device=output.device,
                                                                   dtype=output.dtype),
                                 output)
            if loss_f == 'MSE':
                targets_v_int = targets_v.to(torch.int64)
                targets_v_hot_one = torch.nn.functional.one_hot(targets_v_int,
                                                                num_classes=num_classes).float()
                loss = criterion(output, targets_v_hot_one)
            else:
                loss = criterion(output, targets_v)
            valid_loss += loss.item() * data_v.size(0)
            _, pred = torch.max(output.data, dim=1)  ##<
            total += targets_v.size(0)  ##<
            correct_tensor += (pred == targets_v).sum().item()
    valid_acc = 100 * correct_tensor / total  # the actual train_accuracy_calc
    valid_loss = valid_loss / len(data_loader) # check for CIFAR = 10000!

    if valid_acc > valid_max_acc:
        valid_max_acc = valid_acc
        best_model = model

    elapsed_time = (time.time() - start_time)  # in hours
    eta_mean = elapsed_time / val_cnt
    eta = eta_mean * int(np.floor((tr_size / TR) / val_gaps))
    print('******* batch %03d/%03d | valid_acc: %.f | valid_Loss: %.4f |'
          'time_elapsed/eta: %.2f[s]/%.2f[s] *******\n' % (
              batch_idx, np.floor(tr_size / TR), valid_acc,
              valid_loss, elapsed_time, eta))
    f_txt.write(str('--------------------------------------------------------------------'
                    '-------\n'))
    f_txt.write(
        str('***** batch%04d/%04d |validation_acc:%.4f [%%] |time_elapsed:%.2f/.%2f[s] '
            '*****\n' % (batch_idx, num_of_batches, valid_acc, elapsed_time, eta)))
    f_txt.write(str('--------------------------------------------------------------------'
                    '-------\n'))

    return model, valid_acc, valid_loss, best_model, batch_idx, elapsed_time, valid_max_acc

def test_distillation_r(model, val_cnt, loss_f, data_loader, start_time, criterion, tr_size,
                            TR, val_gaps, f_txt, device, num_classes):

    model.eval()
    test_loss = 0.0
    val_cnt += 1
    total = 1e-16
    correct_tensor = 0
    with torch.no_grad():
        for batch_idx_v, (data_v, targets_v) in enumerate(data_loader):
            data_v, targets_v = data_v.to(device), targets_v.to(device)
            if loss_f == 'MSE':
                targets_v = targets_v.float()  # if we use an MSEloss
            output = model(data_v)
            # safity - if there NaN values - replace with 1e-15
            # nan_count = torch.isnan(output.data).sum()
            output = torch.where(torch.isnan(output), torch.tensor(1e-15,
                                                                   device=output.device,
                                                                   dtype=output.dtype),
                                 output)
            if loss_f == 'MSE':
                targets_v_int = targets_v.to(torch.int64)
                targets_v_hot_one = torch.nn.functional.one_hot(targets_v_int,
                                                                num_classes=num_classes).float()
                loss = criterion(output, targets_v_hot_one)
            else:
                loss = criterion(output, targets_v)
            test_loss += loss.item() * data_v.size(0)
            _, pred = torch.max(output.data, dim=1)  ##<
            total += targets_v.size(0)  ##<
            correct_tensor += (pred == targets_v).sum().item()
    test_acc = 100 * correct_tensor / total  # the actual train_accuracy_calc
    test_loss = test_loss / len(data_loader) # check for CIFAR = 10000!

    elapsed_time = (time.time() - start_time)  # in hours
    eta_mean = elapsed_time / val_cnt
    eta = eta_mean * int(np.floor((tr_size / TR) / val_gaps))
    print('******* Test acc of Best model: %.4f [%%]| test_loss: %.4f |time_elapsed/eta: %.2f[s]/%.2f[s] *******\n' %
          (test_acc, test_loss, elapsed_time, eta))
    f_txt.write(str('==================================================================================\n'))
    f_txt.write(str('***** Test acc of Best model:%.4f [%%] | test_loss: %.4f |time_elapsed:%.2f/.%2f[s] *****\n' %
                    (test_acc, test_loss, elapsed_time, eta)))
    f_txt.write(str('==================================================================================\n'))

    return test_acc, test_loss

def plot_save_res_df(res_log, res_path, fig_name:str, title:str, xlabel:str, yl_label:str, yr_label:str, xaxis_n:str,
                     ylaxis_n:str, yraxis_n:str):
    fig, ax = plt.subplots()
    ax.plot(res_log[xaxis_n], res_log[ylaxis_n], color='b')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=18, color='k')
    ax.set_ylabel(yl_label, fontsize=18, color='b')
    ax2 = ax.twinx()
    ax2.plot(res_log[xaxis_n], res_log[yraxis_n], color='darkorange')
    ax2.set_ylabel(yr_label, fontsize=18, color='darkorange')
    ax.legend()
    os.makedirs(res_path, exist_ok=True)
    fig_nameA = fig_name+".png"
    fig_name_s = os.path.join(res_path, fig_nameA)
    plt.savefig(fig_name_s, dpi=1200)
    plt.show(block=False)
    plt.close()
    csv_name_A = fig_name +".csv"
    csv_path = os.path.join(res_path, csv_name_A)
    res_log.to_csv(csv_path)

def plot_save_time_res_tr(res_log, res_path, fig_name:str, title:str, xlabel:str, y_label:str, xaxis_n:str, yaxis_n:str):
    # fig, ax = plt.subplot()
    plt.plot(res_log[xaxis_n], res_log[yaxis_n], color='b')
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18, color='k')
    plt.ylabel(y_label, fontsize=18, color='k')
    plt.legend()
    # plt.grid()
    fig_nameA = fig_name + ".png"
    fig_name_s = os.path.join(res_path, fig_nameA)
    plt.savefig(fig_name_s, dpi=1200)
    plt.show(block=False)
    plt.close()


# region Distillation of Images _ 06/03/24: For Testing

class SimpleCNN(nn.Module):  # first CNN used for distillation
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class IntermediateCNN(nn.Module):
    def __init__(self):
        super(IntermediateCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv3(x)  # No activation to keep the output range flexible
        return x


class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=3, padding=1)  # Output channels back to 3

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)  # No activation after the last layer, to keep output in the same range as input
        return x


def initialize_distilled_representatives_imgs(dataloader, device, k=5):
    # Initialize storage for distilled representatives
    distilled_representatives = {}
    current_indices = {}  # Track the current index for each class

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        for input, label in zip(inputs, labels):
            label_item = label.item()
            # Initialize storage for new classes
            if label_item not in distilled_representatives:
                distilled_representatives[label_item] = [torch.zeros_like(input.cpu()) for _ in range(k)]
                current_indices[label_item] = 0

            # Store and cyclically replace image pixels
            idx = current_indices[label_item]
            distilled_representatives[label_item][idx] = input.cpu()

            # Update the current index, wrapping around cyclically
            current_indices[label_item] = (idx + 1) % k

    # Convert list to dictionary
    distilled_representatives_dict = {key: torch.stack(tensor) for key, tensor in distilled_representatives.items()}
    return distilled_representatives_dict


def dictionary_to_dataloader(distilled_images_final, batch_size):
    images_list = []
    labels_list = []

    # Iterate over each class label and its corresponding images
    for label, images in distilled_images_final.items():
        for image in images:
            images_list.append(image)  # Add image to the list
            labels_list.append(label)  # Add label to the list

    # Convert lists to tensors
    images_tensor = torch.stack(images_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    # Create a TensorDataset
    dataset = TensorDataset(images_tensor, labels_tensor)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader


def update_distilled_images(dataloader, distilled_images, device, learning_rate, opt_distill, cnn_t, err_scale=1):
    if cnn_t == "Simple":
        cnn_model = SimpleCNN().to(device)
    if cnn_t == "Intermediate":
        cnn_model = IntermediateCNN().to(device)
    # cnn_model = EnhancedCNN().to(device)
    cnn_model.train()

    distilled_images_updated = {label: images.clone().detach().requires_grad_(True) for label, images in
                                distilled_images.items()}

    params_to_optimize = [images for images in distilled_images_updated.values()] + list(cnn_model.parameters())
    if opt_distill == 'SGD':
        optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(params_to_optimize, lr=learning_rate)


    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Initialize variable to accumulate loss
        total_loss = torch.tensor(0.0, device=device)

        # Track if we have processed any matching labels to ensure we call backward
        processed_labels = False

        for label, distilled_image_tensor in distilled_images_updated.items():
            # Check if the current label exists in the batch
            if label in labels:
                # Filter inputs and corresponding distilled images by the current label
                matching_indices = labels == label
                matching_inputs = inputs[matching_indices]

                # Since each distilled image tensor corresponds to one class, we directly use it
                distilled_output = cnn_model(distilled_image_tensor.to(device))
                input_output = cnn_model(matching_inputs)

                # Calculate loss between distilled images and their corresponding input images
                for i in range(distilled_output.size(0)):
                    distilled_single = distilled_output[i].unsqueeze(0).expand_as(input_output)
                    loss = F.mse_loss(distilled_single, input_output) * err_scale
                    total_loss += loss

                processed_labels = True

        # Backpropagate the average loss for this batch
        if processed_labels:
            total_loss /= len(distilled_images_updated)
            total_loss.backward()
            optimizer.step()

    distilled_images_final = {label: images.detach() for label, images in distilled_images_updated.items()}
    return distilled_images_final


def save_distilled_images(distilled_images_final, save_path='distilled_image_grid.png'):
    rows = []

    for class_images in distilled_images_final.values():
        # Convert tensors to PIL Images and append to list for horizontal concatenation
        pil_images = []
        for image in class_images:
            # Simple denormalization to shift values from around [-1, 1] to [0, 1] range
            denormalized_img = (image - image.min()) / (image.max() - image.min())
            # Convert to PIL Image, ensuring correct value scaling and channel order for RGB
            pil_image = Image.fromarray((denormalized_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            pil_images.append(pil_image)

        # Concatenate images horizontally
        class_row = np.hstack([np.array(image) for image in pil_images])
        rows.append(class_row)

    # Concatenate all class rows vertically
    full_image = np.vstack(rows)

    # Convert numpy array back to PIL Image
    full_image_pil = Image.fromarray(full_image)

    # Save the image
    full_image_pil.save(save_path)