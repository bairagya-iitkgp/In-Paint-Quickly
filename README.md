# In-Paint-Quickly

In this repository we are presenting an architecture which is capable of performing In-painting faster using lesser number of parameters and lesser FLOPs compared to Baseline Models.

# Training

1. First choose the proper resolution of the input image for training. To select 128X128 resolution set res=128, IMAGE_SIZE=128,
LOCAL_SIZE=64 and for 256X256 resolution set res=256, IMAGE_SIZE=256, LOCAL_SIZE=128
e.g: To choose 128X128 resolution for training

       python train.py --res=128 --IMAGE_SIZE=128 --LOCAL_SIZE=64
    
2. Here one of the three models can be chosen for in-painting.
   
   1=> Baseline
   
   2=> Bilinear Resize Separable
   
   3=> Pixel Shuffle
   
e.g: To choose Bilinear Resize Separable model set model=2
   
       python train.py --model=2
       
3. To set range for hole-size set the minimum dimension at HOLE_MIN and set maximum dimension at HOLE_MAX.

e.g: To generate holes in the range (24,48) use
   
       python train.py --HOLE_MIN=24 --HOLE_MAX=48
       
4. To change value of learning rate set LEARNING_RATE, to change value of batch-size set BATCH_SIZE, to change value of
hyperparameter alpha set alpha.

e.g: One example of usage is as follows

       python train.py --LEARNING_RATE=1e-3 --BATCH_SIZE=32 --alpha=1.0

5. PRETRAIN_EPOCH sets the value of number of epochs to be spend on Completion Network training phase,Td_EPOCH sets the
   value of number of epochs for Discriminator training phase and Tot_EPOCH sets the value of number of epochs for 
   Total training.
   
e.g: One such example is as follows

       python train.py --PRETRAIN_EPOCH=4 --Td_EPOCH=1 --Tot_EPOCH=11

