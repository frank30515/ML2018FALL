import pandas as pd
import numpy as np
import os

#===================
# input  	: None
# output 	: 1.paths to train picture
#			: 2.one-hot labels (sample no * total class)
def getTrainDataset():

	path_to_train = 'train/'
	data = pd.read_csv('train.csv')

	paths = []
	labels = []
    
	for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
		y = np.zeros(28)
		for key in lbl:
			y[int(key)] = 1
		paths.append(os.path.join(path_to_train, name))
		labels.append(y)

	return np.array(paths), np.array(labels)
	
#===================
# input  	: None
# output 	: 1.paths to train picture
#			: 2.zeros
def getTestDataset():

	path_to_test = 'test/'
	data = pd.read_csv('sample_submission.csv')

	paths = []
	labels = []
    
	for name in data['Id']:
		y = np.zeros(28)
		paths.append(os.path.join(path_to_test, name))
		labels.append(y)
	return np.array(paths), np.array(labels)

#===================
# input  	: 	1.paths
#				2.labels
#				3.validation_ratio
#				4.SEED = None

# output 	: 	1.pathsTrain
#				2.labelsTrain
#				3.pathsVal
#				4.labelsVal

def train_test_split(paths,labels,validation_ratio,SEED=None):
	keys = np.arange(paths.shape[0], dtype=np.int)  
	np.random.seed(SEED)
	np.random.shuffle(keys)
	lastTrainIndex = int((1-validation_ratio) * paths.shape[0])
	
	pathsTrain = paths[0:lastTrainIndex]
	labelsTrain = labels[0:lastTrainIndex]
	pathsVal = paths[lastTrainIndex:]
	labelsVal = labels[lastTrainIndex:]
	
	return pathsTrain,labelsTrain,pathsVal,labelsVal
	