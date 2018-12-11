# In-Paint-Quickly

In this repository we are presenting an architecture which is capable of performing In-painting faster using lesser number of parameters and lesser FLOPs compared to Baseline Models.

# Training

1. First choose the proper resolution of the input image for training. To select 128X128 resolution set res=128, IMAGE_SIZE=128,
LOCAL_SIZE=64 and for 256X256 resolution set res=256, IMAGE_SIZE=256, LOCAL_SIZE=128

      python train.py --res=128 --IMAGE_SIZE=128 --LOCAL_SIZE=64
