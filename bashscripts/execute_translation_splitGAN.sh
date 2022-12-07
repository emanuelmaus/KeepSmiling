#!/bin/bash
python translation_splitGAN.py --dataroot MTFL --workers 1 --batchSize 1 --imageSize 64 --mode testing --cuda --ngpu 1 --netDe netDe_folder/netDe_epoch_199.pth --netE netE_folder/netE_epoch_199.pth --outf result_smiling

