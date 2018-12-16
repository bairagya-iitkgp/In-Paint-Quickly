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
   
   For 256X256 model there is no need to choose 1 for baseline as in this case we have used in-painting model from 
   'http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Generative_Image_Inpainting_CVPR_2018_paper.pdf' as our
   baseline model. 
   
e.g: To choose Bilinear Resize Separable model set model=2
   
       python train.py --model=2
       
3. To set range for hole-size set the minimum dimension at HOLE_MIN and set maximum dimension at HOLE_MAX.

e.g: To generate holes in the range (24,48) use
   
       python train.py --HOLE_MIN=24 --HOLE_MAX=48
       
4. To change value of learning rate set LEARNING_RATE, to change value of batch-size set BATCH_SIZE, to change value of
hyperparameter alpha set alpha.

e.g: One example of usage is as follows

       python train.py --LEARNING_RATE=0.001 --BATCH_SIZE=32 --alpha=1.0

5. PRETRAIN_EPOCH sets the value of number of epochs to be spend on Completion Network training phase,Td_EPOCH sets the
   value of number of epochs for Discriminator training phase and Tot_EPOCH sets the value of number of epochs for 
   Total training.
   
e.g: One such example is as follows

       python train.py --PRETRAIN_EPOCH=4 --Td_EPOCH=1 --Tot_EPOCH=11
       
       
# How to Set Up Paths to Various Folders

1. Path to save checkpoints can be set as follows:

       python train.py --checkpoints_path = ./backup

2. Path to restore a model from saved checkpoints or to save current model can be set as follows:

       python train.py --restoration_path = ./backup/latest

3. Path to access the training data stored in numpy format can be set as follows:

       python train.py --data_path = ./npy
       
4. Path to save the original input images can be set as follows: 

       python train.py --original = ./original

5. Path to save the output images can be set as follows:

       python train.py --output = ./output

6. Path to save the perturbed images can be set as follows:

       python train.py --perturbed = ./perturbed


# How to Arrange Training Data

Before training the dataset should be converted to an numpy array and then the path to that folder should be paased to 
'data_path' either by making changes to config.py file or using command line arguments.

Train data should be saved as 'x_train.npy' and Test data should be saved as 'x_test.npy' otherwise, necessary changes should be
done in load.py file.

# How to Resume Training from a Previously Saved Checkpoint

To resume training from a previously saved checkpoint first save the checkpoint files in a specific folder and pass the address
to the 'restoration_path'.

# How to Use Weights from a Pre-trained Model for Training another model

To accomplish this purpose first keep the Pre-trained model in a particular folder and pass the address of the folder to the
'pretrain_path'.

Then set 'use_pretrain'=True and start training.

# How to run inference on test images

To run inference on some test images, use "test.py" code. 

First, load the required checkpoints in some folder and add that path to 'restoration_path' in "config.py" file or through
command line arguments as shown above.

Convert the test images into a numpy array and privide the path to 'test_data_path' in "config.py" or through command line argument.

Similarly provide the path of the output folder (to save outputs) to 'test_out' in "config.py" or through command line argument. 

Then choose the desired resolution of operation and proper model option (same as shown for case of 'training test') and run
"test.py". 

# Link to download the pre-trained checkpoint files for different models

    https://drive.google.com/open?id=1f12naqJbiQSkk3B7a9GghXf5TFXUJdKb

128X128 model has been trained on celeba dataset and 256X256 celeba-HQ dataset.




