#!/bin/bash
python main_comlexGAN.py --dataroot MTFL --workers 1 --batchSize 16 --imageSize 64 --mode training --niter 1000 --cuda --ngpu 1 --outf output_images

