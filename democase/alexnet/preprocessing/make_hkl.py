# Preprocessing: From JPEG to HKL

from __future__ import division
import math
import multiprocessing as mp

import os
import glob
import sys

import yaml
import scipy.misc
import numpy as np

import hickle as hkl


def get_img(img_name, img_size=256, batch_size=256):

    target_shape = (img_size, img_size, 3)
    img = scipy.misc.imread(img_name)  # x*x*3
    assert img.dtype == 'uint8', img_name
    # assert False

    if len(img.shape) == 2:
        img = scipy.misc.imresize(img, (img_size, img_size))
        img = np.asarray([img, img, img])
    else:
        if img.shape[2] > 3:
            img = img[:, :, :3]
        img = scipy.misc.imresize(img, target_shape)
        img = np.rollaxis(img, 2)
    if img.shape[0] != 3:
        print img_name
    return img


def save_batches(file_list, tar_dir, batch_count, mean_output, 
                 img_size=256, batch_size=256, flag_avg=False, 
                 num_sub_batch=1, data_format='nchw'):
    '''
    num_sub_batch is for parallelling using multiple gpus, it should be
    2, 4, or 8,
    where the indexing is reverted binary number
    when 2, the files ends with _0.pkl and _1.pkl
    when 4, with _00.pkl, _10.pkl, _01.pkl and _11.pkl

    '''

    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

    if data_format == 'bchw':
        img_batch = np.zeros((batch_size, 3, img_size, img_size), np.uint8)
    elif data_format == 'chwb':
        img_batch = np.zeros((3, img_size, img_size, batch_size), np.uint8)
    else:
        NotImplementedError("data_format has to be bchw or chwb")

    if flag_avg:
        img_sum = np.zeros((3, img_size, img_size))

    #batch_count = 0
    count = 0

    if data_format == 'bchw':
        for file_name in file_list:
            #print 'bchw:', file_name
            img_batch[count % batch_size, :, :, :] = \
                get_img(file_name, img_size=img_size, batch_size=batch_size)

            count += 1
            if count % batch_size == 0:
                batch_count += 1

                if flag_avg:
                    img_sum += img_batch.mean(axis=0)

                if num_sub_batch == 1:
                    save_name = '%04d' % (batch_count - 1) + '.hkl'
                    hkl.dump(img_batch, os.path.join(tar_dir, save_name), mode='w')

                elif num_sub_batch == 2:
                    half_size = batch_size / 2
                    save_name = '%04d' % (batch_count - 1) + '_0.hkl'
                    hkl.dump(img_batch[:half_size, :, :, :],
                             os.path.join(tar_dir, save_name), mode='w')

                    save_name = '%04d' % (batch_count - 1) + '_1.hkl'
                    hkl.dump(img_batch[half_size:, :, :, :],
                             os.path.join(tar_dir, save_name), mode='w')

                elif num_sub_batch == 4:
                    q1 = batch_size / 4
                    q2 = batch_size / 2
                    q3 = batch_size / 4 * 3

                    save_name = '%04d' % (batch_count - 1) + '_00.hkl'
                    hkl.dump(img_batch[:q1, :, :, :],
                             os.path.join(tar_dir, save_name), mode='w')
                    save_name = '%04d' % (batch_count - 1) + '_10.hkl'
                    hkl.dump(img_batch[q1:q2, :, :, :],
                             os.path.join(tar_dir, save_name), mode='w')
                    save_name = '%04d' % (batch_count - 1) + '_01.hkl'
                    hkl.dump(img_batch[q2:q3, :, :, :],
                             os.path.join(tar_dir, save_name), mode='w')
                    save_name = '%04d' % (batch_count - 1) + '_11.hkl'
                    hkl.dump(img_batch[q3:, :, :, :],
                             os.path.join(tar_dir, save_name), mode='w')
                else:
                    NotImplementedError("num_sub_batch has to be 1, 2, or 4")
    elif data_format == 'chwb':
        for file_name in file_list:
            #print 'chwb', file_name
            img_batch[:, :, :, count % batch_size] = \
                get_img(file_name, img_size=img_size, batch_size=batch_size)

            count += 1
            if count % batch_size == 0:
                batch_count += 1

                if flag_avg:
                    img_sum += img_batch.mean(axis=3)

                if num_sub_batch == 1:
                    save_name = '%04d' % (batch_count - 1) + '.hkl'
                    hkl.dump(img_batch, os.path.join(tar_dir, save_name), mode='w')

                elif num_sub_batch == 2:
                    half_size = batch_size / 2
                    save_name = '%04d' % (batch_count - 1) + '_0.hkl'
                    hkl.dump(img_batch[:, :, :, :half_size],
                             os.path.join(tar_dir, save_name), mode='w')

                    save_name = '%04d' % (batch_count - 1) + '_1.hkl'
                    hkl.dump(img_batch[:, :, :, half_size:],
                             os.path.join(tar_dir, save_name), mode='w')

                elif num_sub_batch == 4:
                    q1 = batch_size / 4
                    q2 = batch_size / 2
                    q3 = batch_size / 4 * 3

                    save_name = '%04d' % (batch_count - 1) + '_00.hkl'
                    hkl.dump(img_batch[:, :, :, :q1],
                             os.path.join(tar_dir, save_name), mode='w')
                    save_name = '%04d' % (batch_count - 1) + '_10.hkl'
                    hkl.dump(img_batch[:, :, :, q1:q2],
                             os.path.join(tar_dir, save_name), mode='w')
                    save_name = '%04d' % (batch_count - 1) + '_01.hkl'
                    hkl.dump(img_batch[:, :, :, q2:q3],
                             os.path.join(tar_dir, save_name), mode='w')
                    save_name = '%04d' % (batch_count - 1) + '_11.hkl'
                    hkl.dump(img_batch[:, :, :, q3:],
                             os.path.join(tar_dir, save_name), mode='w')
                else:
                    NotImplementedError("num_sub_batch has to be 1, 2, or 4")
    else:
        NotImplementedError("data_format has to be bchw or chwb")

    mean_output.put(img_sum) if flag_avg else None
    #return img_sum / batch_count if flag_avg else None


def get_train_filenames(src_train_dir, misc_dir, seed=1):

    if os.path.exists(os.path.join(misc_dir, 'shuffled_train_filenames.npy')):
        return np.load(os.path.join(misc_dir, 'shuffled_train_filenames.npy'))

    if not os.path.exists(misc_dir):
        os.makedirs(misc_dir)

    print 'shuffled_train_filenames not found, generating ...'

    subfolders = [name for name in os.listdir(src_train_dir)
                  if os.path.isdir(os.path.join(src_train_dir, name))]

    train_filenames = []
    for subfolder in subfolders:
        train_filenames += glob.glob(src_train_dir + subfolder + '/*JPEG')

    train_filenames = np.asarray(sorted(train_filenames))

    np.random.seed(seed)
    np.random.shuffle(train_filenames)
    np.save(os.path.join(misc_dir, 'shuffled_train_filenames.npy'),
            train_filenames)

    return train_filenames


if __name__ == '__main__':
    with open('paths.yaml', 'r') as f:
        paths = yaml.load(f)

    train_img_dir = paths['train_img_dir']
    val_img_dir = paths['val_img_dir']
    misc_dir = paths['misc_dir']

    if len(sys.argv) < 2:
        gen_type = 'full'
    else:
        gen_type = sys.argv[1]

    if gen_type == 'full':
        print 'generating full dataset ...'
    elif gen_type == 'toy':
        print 'generating toy dataset ...'
    else:
        NotImplementedError("gen_type (2nd argument of make_hkl.py) can only be full or toy")

    train_filenames = get_train_filenames(train_img_dir, misc_dir)
    val_filenames = np.asarray(sorted(glob.glob(val_img_dir + '/*JPEG')))

    img_size = 256
    train_batch_size = 256
    val_batch_size = 256


    if gen_type == 'toy':
        # generate 100 batches each
        train_filenames = train_filenames[:2560]
        val_filenames = val_filenames[:2560]

    # added to support multi-process 
    num_process = 16
    train_batch_num = int( math.ceil(len(train_filenames)/train_batch_size) )
    train_batchs_per_core = train_batch_num // num_process

    val_batch_num = int( math.ceil(len(val_filenames)/val_batch_size) )
    val_batchs_per_core = val_batch_num // num_process
    #print batchs_per_core

    for num_sub_batch in [1]:
    #for num_sub_batch in [1, 2]:
        for data_format in ['bchw']:
        #for data_format in ['bchw', 'chwb']:
            tar_train_dir = paths['tar_train_dir']
            tar_val_dir = paths['tar_val_dir']
            tar_train_dir += '_b' + str(train_batch_size) + \
                '_b' + str(train_batch_size // num_sub_batch) + "_" + data_format
            tar_val_dir += '_b' + str(val_batch_size) + \
                '_b' + str(val_batch_size // num_sub_batch) + "_" + data_format

            # added to support multi-process: 2015-10-19
            mean_output = mp.Queue() # Define an output queue
            processes = [mp.Process(target=save_batches,
                                args=(
                                    train_filenames[batch_id * train_batch_size:
                                                    (batch_id + train_batchs_per_core) * train_batch_size],
                                    tar_train_dir, 
                                    batch_id, 
                                    mean_output, 
                                    img_size,
                                    train_batch_size, 
                                    True,   # flag_avg 
                                    num_sub_batch, 
                                    data_format)) 
                                        for batch_id in xrange(0,
                                                               train_batch_num,
                                                               train_batchs_per_core)
                        ]

            # Run processes
            print "Generate training dataset with %d processes ...." % len(processes)
            for p in processes:
                p.start()
                
            # Get process results from the output queue
            results = [mean_output.get() for p in processes]
            img_mean = sum(results) / train_batch_num
            #print img_mean
            #print img_mean.shape

            # Exit the completed processes
            for p in processes:
                p.join()

            np.save(os.path.join(misc_dir, 'img_mean.npy'), img_mean)
            

            processes = [mp.Process(target=save_batches,
                                args=(
                                    val_filenames[batch_id * val_batch_size:
                                                  (batch_id + val_batchs_per_core) * val_batch_size],
                                    tar_val_dir, 
                                    batch_id, 
                                    mean_output, 
                                    img_size,
                                    val_batch_size, 
                                    False, 
                                    num_sub_batch, 
                                    data_format)) 
                                        for batch_id in xrange(0, val_batch_num,
                                                               val_batchs_per_core)
                        ]

            # Run processes
            print "Generate validation dataset with %d processes ...." % len(processes)
            for p in processes:
                p.start()
                
            # Exit the completed processes
            for p in processes:
                p.join()
