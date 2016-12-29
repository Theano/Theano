'''
Load data in parallel with train.py
'''

import time
import math

import numpy as np
import zmq
import hickle as hkl


def get_params_crop_and_mirror(param_rand, data_shape, cropsize):

    center_margin = (data_shape[2] - cropsize) / 2
    crop_xs = round(param_rand[0] * center_margin * 2)
    crop_ys = round(param_rand[1] * center_margin * 2)
    if False:
        # this is true then exactly replicate Ryan's code, in the batch case
        crop_xs = math.floor(param_rand[0] * center_margin * 2)
        crop_ys = math.floor(param_rand[1] * center_margin * 2)

    flag_mirror = bool(round(param_rand[2]))

    return crop_xs, crop_ys, flag_mirror


def crop_and_mirror(data, param_rand, flag_batch=True, cropsize=227):
    '''
    when param_rand == (0.5, 0.5, 0), it means no randomness
    '''
    # print param_rand

    # if param_rand == (0.5, 0.5, 0), means no randomness and do validation
    if param_rand[0] == 0.5 and param_rand[1] == 0.5 and param_rand[2] == 0:
        flag_batch = True

    if flag_batch:
        # mirror and crop the whole batch
        crop_xs, crop_ys, flag_mirror = \
            get_params_crop_and_mirror(param_rand, data.shape, cropsize)

        # random mirror
        if flag_mirror:
            data = data[:, :, :, ::-1]

        # random crop
        data = data[:, :, crop_xs:crop_xs + cropsize,
                    crop_ys:crop_ys + cropsize]

    else:
        # mirror and crop each batch individually
        # to ensure consistency, use the param_rand[1] as seed
        np.random.seed(int(10000 * param_rand[1]))

        data_out = np.zeros((data.shape[0], data.shape[1], cropsize, cropsize)).astype('float32')

        for ind in range(data.shape[0]):
            # generate random numbers
            tmp_rand = np.float32(np.random.rand(3))
            tmp_rand[2] = round(tmp_rand[2])

            # get mirror/crop parameters
            crop_xs, crop_ys, flag_mirror = \
                get_params_crop_and_mirror(tmp_rand, data.shape, cropsize)

            # do image crop/mirror
            img = data[ind, :, :, :]
            if flag_mirror:
                img = img[:, :, ::-1]
            img = img[:, crop_xs:crop_xs + cropsize,
                      crop_ys:crop_ys + cropsize]
            data_out[ind, :, :, :] = img

        data = data_out

    return np.ascontiguousarray(data, dtype='float32')


def fun_load(config, sock_data=5000):

    send_queue = config['queue_l2t']
    recv_queue = config['queue_t2l']
    # recv_queue and send_queue are multiprocessing.Queue
    # recv_queue is only for receiving
    # send_queue is only for sending

    # if need to do random crop and mirror
    flag_batch = config['batch_crop_mirror']

    sock = zmq.Context().socket(zmq.PAIR)
    sock.bind('tcp://*:{0}'.format(sock_data))

    shared_x = sock.recv_pyobj()
    print 'shared_x information received'

    img_mean = recv_queue.get()
    print 'img_mean received'

    # The first time, do the set ups and other stuff

    # receive information for loading

    while True:
        # getting the hkl file name to load
        hkl_name = recv_queue.get()

        data = hkl.load(hkl_name) - img_mean

        param_rand = recv_queue.get()

        data = crop_and_mirror(data, param_rand, flag_batch=flag_batch)

        shared_x.set_value(data)

        # wait for computation on last minibatch to finish
        msg = recv_queue.get()
        assert msg == 'calc_finished'

        send_queue.put('copy_finished')
