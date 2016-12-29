'''
For generating caffe style train and validation label txt files
'''
import os
import yaml
import scipy.io
import numpy as np
import sys

with open('paths.yaml', 'r') as f:
    paths = yaml.load(f)

meta_clsloc_mat = paths['meta_clsloc_mat']
train_img_dir = paths['train_img_dir']
val_img_dir = paths['val_img_dir']
val_label_file = paths['val_label_file']
misc_dir = paths['misc_dir']


valtxt_filename = os.path.join(misc_dir, 'val.txt')
traintxt_filename = os.path.join(misc_dir, 'train.txt')

sorted_train_dirs = sorted([name for name in os.listdir(train_img_dir)
                            if os.path.isdir(os.path.join(train_img_dir, name))])

synsets = scipy.io.loadmat(meta_clsloc_mat)['synsets'][0]

synsets_wnid = [str(synset[1][0]) for synset in synsets]
dict_wnid_to_origid = {
    str(synset[1][0]): int(synset[0][0]) for synset in synsets}

# extract actual image correspondence
#for wnid in sorted_train_dirs:

dict_train_id = {wnid: dict_wnid_to_origid[wnid] for wnid in sorted_train_dirs}
sorted_train_ids = np.asarray([[wnid, dict_wnid_to_origid[wnid]]
                               for wnid in sorted_train_dirs])

sorted_id_list = range(len(sorted_train_ids))

dict_wnid_to_sorted_id = {sorted_train_ids[ind, 0]: ind
                          for ind in sorted_id_list}
dict_orig_id_to_sorted_id = {int(sorted_train_ids[ind, 1]): ind
                             for ind in sorted_id_list}
dict_sorted_id_to_orig_id = {ind: int(sorted_train_ids[ind, 1])
                             for ind in sorted_id_list}


# generate val.txt
val_img_list = sorted([name for name in os.listdir(val_img_dir)
                       if '.JPEG' in name])

with open(val_label_file, 'r') as f:
    val_labels = f.readlines()

assert len(val_labels) == len(val_img_list), \
    'Validation data: Numbers of images and labels should be the same.'

with open(valtxt_filename, 'w') as f:
    for ind in range(len(val_labels)):
        str_write = val_img_list[ind] + ' ' + \
            str(dict_orig_id_to_sorted_id[int(val_labels[ind])]) + '\n'
        f.write(str_write)


# generate train.txt
train_filenames = []
for folder in sorted_train_dirs:
    train_filenames += sorted(
        [folder + '/' + name + ' ' + str(dict_wnid_to_sorted_id[folder]) + '\n'
         for name in os.listdir(os.path.join(train_img_dir, folder))
         if '.JPEG' in name])

with open(traintxt_filename, 'w') as f:
    f.writelines(train_filenames)
    
