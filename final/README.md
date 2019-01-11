# ML2018FALL

## Toolkits
torchvision==0.2.1  
sklearn==1.8.0    
imgaug==0.2.6  
opencv-python==3.4.1.15  
pretrainedmodels==0.7.4

## How to Test
1. Change the and test_data (testing data path) in src/config.py, please use absolute path!

2. Testing  
$ bash test.sh

3. Get the inception_bestloss_submission.csv in the same directory

## How to Train
1. Change the and train_data (training data path) in src/config.py, please use absolute path!

2. Training and save model into checkpoints/best_models/  
$ bash train.sh

## Reference
https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/75691