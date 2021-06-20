#!/bin/sh
CUDA_VISIBLE_DEVICES=2,6 python3 main.py --base configs/str_vqgan.yaml -t True --gpus 0,1 --accelerator ddp --precision 16 --val_check_interval 1000
