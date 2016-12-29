#!/bin/bash
python make_hkl.py toy
python make_train_val_txt.py
python make_labels.py
