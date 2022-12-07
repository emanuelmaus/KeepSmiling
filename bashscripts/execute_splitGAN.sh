#!/bin/bash
python main_splitGAN.py --dataroot MTFL --workers 1 --batchSize 16 --imageSize 64 --mode training --nstart 0 --niter 200 --cuda --ngpu 1 --outf output_images

