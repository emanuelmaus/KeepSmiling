#!/bin/bash
python encoder_splitGAN.py --dataroot MTFL --workers 1 --batchSize 1 --imageSize 64 --mode testing --cuda --ngpu 1 --netE netE_folder/netE_epoch_199.pth --outf result_smiling

