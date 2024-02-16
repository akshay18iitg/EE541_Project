# EE541_Project

## Structure of Major files

The main file structure for the project is:

- ESRGAN folder :<br>
  This folder consists of all the python files, configuration and requirement enviroment version files to train and test ESRGAN model. The structure of this folder is discussed in   
  detail in later section.  
- SRGAN folder :<br>
  This folder consists of all the python files, configuration and requirement enviroment version files to train and test SRGAN model. The structure of this folder is discussed in       
  detail in later section. 
- Super_Resolution_ESRGAN.ipynb : <br>
  This file helps to train the default ESRGAN model using configuration files, load the high resolution images and low resolution images, save best and last checkpopints for generator 
  and discriminator model and save and plot training and testing metrics in tensorboard. The detailed information abpout how to run ESRGAN model on yout oum hyperparamater and custom 
  data is discussed below.  
- Super_Resolution_SRGAN.ipynb :<br>
  This file helps to train the default SRGAN model by passing arguments, load the high resolution images and low resolution images, save best and last checkpopints for generator and 
  discriminator model and save and plot training and testing metrics in tensorboard. The detailed information abpout how to run SRGAN model on yout oum hyperparamater and custom data 
  is discussed below. 

## Sturcture of Major files in ESRGAN folder
- Config folder :<br>
This folder consists of all the configuration files required to train and test a particular ESRGAN model. It consists information regarding train data path, validation data path and various hyperparameter related to ESRGAN model.
- Data folder :<br>
This folder contains the training high resolution images, training low resolution images , testing high resolution images and testing low resolution images.
- Results folder :<br>
This folder consists of last and best checkpoints for generator and discriminator models.
- Sample folder :<br>
This folder consists of log files consisting of training and testing metrics for the model in tensorboard format.  
- Dataset.py :<br>
This python scripts consists of codes handling dataloader for test and train dataset, cudafetcher for train to fasten the process of loading the data.
- Image_quality_assessment.py:<br>
This python scripts calculates the PSNR ans SSIM values for validation dataset.
- Imgproc.py :<br>
This python script includes all the preprocessing methods used on images tensors.
- Inference.py :<br>
This python script runs the model in inference mode.
- Model.py :<br>
This python script includes th implementation of VGG, basic RRDB block, disciminator and generator architecture in pytorch. Changes to backbone of ESRGAN model can be made ion this file.
- Test.py :<br>
This python script runs the model in testing mode.
- Train_gan.py :<br>
This python scripts trainbs the ESRGAN model.
- Utils.py :<br>
This python script contains supplementery modules.

## Structure of major files in SRGAN folder
- Data folder :<br>
This folder contains the training high resolution images, training low resolution images , testing high resolution images and testing low resolution images.
- Results folder :<br>
This folder consists of last and best checkpoints for generator and discriminator models.
- Sample folder :<br>
This folder consists of log files consisting of training and testing metrics for the model in tensorboard format.  
-  losses.py: <br>
This python script consists of all the loss module required to train the SRGAN model. New loss function should be implemented in this file.
-  main.py : <br>
This python script runs the SRGAN model in training, validation and testing mode.
-  mode.py : <br>
This python script consists of tarining, validation and testing code for SRGAN model.
-  srgan_model.py : <br>
This python script implements the architecture for generator and discriminator model. 
-  vgg19.py : <br>
This python script implements the vgg architecure.


## REQUIREMENT FILES
The version required for ESRGAN is mentiond in requirement.txt file in the ESRGAN folder. 

## Custom training for ESRGAN Model
We create 5 configuration files, ESRGAN_x4_DIV2K, ESRGAN_x4_DIV2K_1, ESRGAN_x4_DIV2K_2, ESRGAN_x4_DIV2K_3 and ESRGAN_x4_DIV2K_4. These files are runned as:<br>
!python3 train_gan.py --config_path ./configs/train/ESRGAN_x4-DIV2K.yaml <br>
!python3 train_gan.py --config_path ./configs/train/ESRGAN_x4-DIV2K_1.yaml<br>
!python3 train_gan.py --config_path ./configs/train/ESRGAN_x4-DIV2K_2.yaml<br>
!python3 train_gan.py --config_path ./configs/train/ESRGAN_x4-DIV2K_3.yaml<br>
!python3 train_gan.py --config_path ./configs/train/ESRGAN_x4-DIV2K_4.yaml<br>

Custom training for ESRGAN can be done by chaging the confiouration files. Few of the important options to change are:
- Model <br>
  - EMA:<br>
    ENABLE: True/Fasle (Network Interplotation)<br>
  - G: <br>
    NAME: rrdbnet_x4   (Block)<br>
    CHANNELS: 64       <br>
    NUM_RRDB: 23       (Number of RRDB layers)<br>
  -  D:<br>
    NAME: discriminator_for_vgg  (backbone for discriminator)<br>

- TRAIN:<br>
  - DATASET:<br>
    TRAIN_GT_IMAGES_DIR: ./data/DIV2K_train_HR  (Training High image resolution data path)<br>
    TRAIN_LR_IMAGES_DIR: ./data/DIV2K_train_LR_mild (Training low resolution data path)<br>
    GT_IMAGE_SIZE: 128  (Path size)<br>

  - CHECKPOINT:<br>
    PRETRAINED_G_MODEL: ""       (pretrained weight path) <br>
    PRETRAINED_D_MODEL: ""       (pretrained weight path) <br>
    RESUMED_G_MODEL: "" <br>
    RESUMED_D_MODEL: "" <br>

  - HYP:<br>
    IMGS_PER_BATCH: 16           (batch size) <br>
    EPOCHS: 50                   (No of epochs)<br>

  - OPTIM: <br>
    NAME: Adam                   (Optimizer) <br>
    LR: 0.0001<br>
    BETAS: [0.9, 0.999]<br>
    EPS: 0.0001<br>
    WEIGHT_DECAY: 0.0<br>

  - LR_SCHEDULER:<br>
    NAME: MultiStepLR <br>
    MILESTONES: [ 16, 32, 64, 104 ] # 0.125, 0.250, 0.500, 0.800 <br>
    GAMMA: 0.5 <br>

  
  - LOSSES:<br>
    PIXEL_LOSS:<br>
      NAME: L1Loss<br>
      WEIGHT: [0.01]               (pixel loss weightage)<br>
    CONTENT_LOSS:<br>
      NAME: ContentLoss<br> 
      WEIGHT: [1.0]                (content loss weightage)<br>
    ADVERSARIAL_LOSS:<br>
      WEIGHT: [0.005]              (adversarial loss weightage)<br>


- TEST:<br>
  - DATASET:<br>
    PAIRED_TEST_GT_IMAGES_DIR: ./data/DIV2K_valid_HR                (Testing High Resolution path)<br>
    PAIRED_TEST_LR_IMAGES_DIR: ./data/DIV2K_valid_LR_mild           (Testing low resolution path)<br>

  
   - HYP:<br>
    IMGS_PER_BATCH: 1<br>

## Custom training for SRGAN Model

!python3 main.py --pre_train_epoch 20 --fine_train_epoch 50  <br>
!python3 main.py --pre_train_epoch 40 --fine_train_epoch 50  <br>

- LR_path         (Training Low Resolution path)
- GT_path            (Training High Resolution path) 
- LR_path_val        (Validation Low Resolution path)
- GT_path_val        (Validation high Resolution path)
- res_num            
- batch_size         (Batch size)
- L2_coeff          (L2 weightage)
- adv_coeff          (Adversarial weightage)
- tv_loss_coeff      
- pre_train_epoch    (number of pre train epoch)
- fine_train_epoch   (number of fine train epoch)
- mode              (train/val/test)



Model files

