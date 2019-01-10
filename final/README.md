# ML2018FALL

## Toolkits
pip3 install pretrainedmodels

## How to use
### Approach 1:
1. Change the train_data (training data path) and test_data path (testing data path) in src/config.py

2. Training and save model into checkpoints/best_models/  
$ bash train.sh

3. Testing  
$ bash test.sh

4. Get the inception_bestloss_submission.csv in the same directory

### Approach 2:
1. Download train.zip and test.zip from Human Protein Kaggle Website

2. Unzip two zip file   
$ unzip train.zip -d train/  
$ unzip test.zip -d test/

3. Training and save model into checkpoints/best_models/  
$ bash train.sh

4. Testing  
$ bash test.sh

5. Get the submission.csv in the same directory