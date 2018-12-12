from keras.models import load_model
import sys
import pandas as pd
import numpy as np
from data_gen import *
from util import *
from model import f1
from sklearn.metrics import f1_score as off1

if __name__ == "__main__":
	
	#==============train th=============
	paths,labels =getTrainDataset()
	t_paths,t_labels,v_paths,v_labels = train_test_split(paths,labels,VAL_RATIO,SEED)
	fullValGen = ProteinDataGenerator(v_paths,v_labels,BATCH_SIZE,SHAPE)
	
	#load model
	bestModel = load_model('./base.model', custom_objects={'f1': f1}) #, 'f1_loss': f1_loss})
	
	rng = np.arange(0, 1, 0.001)
	f1s = np.zeros((rng.shape[0], 28))
	lastFullValPred = np.empty((0, 28))
	lastFullValLabels = np.empty((0, 28))
	for i in range(len(fullValGen)): 
		im, lbl = fullValGen[i]
		scores = bestModel.predict(im)
		lastFullValPred = np.append(lastFullValPred, scores, axis=0)
		lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)
		
	for j,t in enumerate(rng):
		for i in range(28):
			p = np.array(lastFullValPred[:,i]>t, dtype=np.int8)
			scoref1 = off1(lastFullValLabels[:,i], p, average='binary')
			f1s[j,i] = scoref1
	
	T = np.empty(28)
	for i in range(28):
		T[i] = rng[np.where(f1s[:,i] == np.max(f1s[:,i]))[0][0]]
	
	
	
	#get test path and label
	pathsTest, labelsTest = getTestDataset()
	
	
	
	#data gen
	testg = ProteinDataGenerator(pathsTest, labelsTest, BATCH_SIZE, SHAPE)
	
	
	submit = pd.read_csv('sample_submission.csv')
	P = np.zeros((pathsTest.shape[0], 28))
	for i in range(len(testg)):
		images, labels = testg[i]
		score = bestModel.predict(images)
		P[i*BATCH_SIZE:i*BATCH_SIZE+score.shape[0]] = score
	
	PP = np.array(P)
	
	prediction = []

	for row in range(submit.shape[0]):
		
		str_label = ''
		
		for col in range(PP.shape[1]):
			if(PP[row, col] < T[col]):
				str_label += ''
			else:
				str_label += str(col) + ' '
		prediction.append(str_label.strip())
		
	submit['Predicted'] = np.array(prediction)
	submit.to_csv('ans.csv', index=False)