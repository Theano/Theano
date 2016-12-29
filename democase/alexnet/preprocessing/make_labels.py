# map labels to shuffled images

import os
import yaml
import numpy as np


def div_labels(src_file, orig_batch_size, num_div):
    '''
    src_file, is a .npy file
    orig_batch_size, original batch size
    num_div, is number of batches to divide, can only be 2 or 4 for now
    '''

    labels = np.load(src_file)

    labels = labels[:labels.size / orig_batch_size * orig_batch_size]
    assert labels.size % orig_batch_size == 0

    batch_size = orig_batch_size / num_div

    if num_div == 2:
        labels_0 = labels.reshape((-1, batch_size))[::num_div].reshape(-1)
        labels_1 = labels.reshape((-1, batch_size))[1::num_div].reshape(-1)

        # sanity check
        for ind in range(labels.size / batch_size):

            assert np.all(labels.reshape((-1, batch_size))[ind] ==
                          labels[batch_size * ind: batch_size * (ind + 1)])

            labels_sub = labels_1 if ind % 2 else labels_0

            ind_sub = ind / 2
            assert np.all(labels[batch_size * ind: batch_size * (ind + 1)] ==
                          labels_sub[batch_size * ind_sub: batch_size * (ind_sub + 1)])
        # sanity check finished

        tar_file = src_file[:-4] + '_0.npy'
        np.save(tar_file, labels_0)

        tar_file = src_file[:-4] + '_1.npy'
        np.save(tar_file, labels_1)

    elif num_div == 4:

        labels_00 = labels.reshape((-1, batch_size))[::num_div].reshape(-1)
        labels_10 = labels.reshape((-1, batch_size))[1::num_div].reshape(-1)
        labels_01 = labels.reshape((-1, batch_size))[2::num_div].reshape(-1)
        labels_11 = labels.reshape((-1, batch_size))[3::num_div].reshape(-1)

        tar_file = src_file[:-4] + '_00.npy'
        np.save(tar_file, labels_00)
        tar_file = src_file[:-4] + '_10.npy'
        np.save(tar_file, labels_10)
        tar_file = src_file[:-4] + '_01.npy'
        np.save(tar_file, labels_01)
        tar_file = src_file[:-4] + '_11.npy'
        np.save(tar_file, labels_11)

        # sanity check
        dict_labels = {0: labels_00, 1: labels_10, 2: labels_01, 3: labels_11}
        for ind in range(labels.size / batch_size):

            assert np.all(labels.reshape((-1, batch_size))[ind] ==
                          labels[batch_size * ind: batch_size * (ind + 1)])
            labels_sub = dict_labels[ind % 4]
            ind_sub = ind / 4
            assert np.all(labels[batch_size * ind: batch_size * (ind + 1)] ==
                          labels_sub[batch_size * ind_sub: batch_size * (ind_sub + 1)])

    else:
        NotImplementedError("num_sub_batch has to be 1, 2, or 4")


def save_train_labels(misc_dir, train_label_name):
    ### TRAIN LABELS ###
    label_dict = {}
    # read the labels from train.txt
    with open(os.path.join(misc_dir, 'train.txt'), 'r') as text_labels:
        lines = text_labels.readlines()
    for line in lines:
        filename, label = line.split()
        filename = filename.split('/')[1]
        label_dict[filename] = int(label)

    # save the label npy file according to the shuffled filenames
    train_filenames = np.load(os.path.join(misc_dir,
                                           'shuffled_train_filenames.npy'))
    final_labels = []
    for train_filename in train_filenames:
        key = train_filename.split('/')[-1]
        final_labels.append(label_dict[key])

    np.save(train_label_name, final_labels)


def save_val_labels(misc_dir, val_label_name):
    ### VALIDATION LABELS ###
    with open(os.path.join(misc_dir, 'val.txt'), 'r') as text_labels:
        lines = text_labels.readlines()
    labels = []
    for line in lines:
        labels.append(int(line.split()[1]))
    np.save(val_label_name, labels)

if __name__ == '__main__':
    with open('paths.yaml', 'r') as f:
        paths = yaml.load(f)

    tar_root_dir = paths['tar_root_dir']

    misc_dir = os.path.join(tar_root_dir, 'misc')
    label_dir = os.path.join(tar_root_dir, 'labels')
    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)
    train_label_name = os.path.join(label_dir, 'train_labels.npy')
    val_label_name = os.path.join(label_dir, 'val_labels.npy')
    orig_batch_size = 256
    
    save_val_labels(misc_dir, val_label_name)
    save_train_labels(misc_dir, train_label_name)

    num_div = 2
    div_labels(train_label_name, orig_batch_size, num_div)
    div_labels(val_label_name, orig_batch_size, num_div)
