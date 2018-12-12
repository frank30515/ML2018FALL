import pandas as pd
import numpy as np
import keras
from keras.utils import Sequence
from data_gen import *
from util import *
from model import *
from param import *

if __name__ == "__main__":
	paths,labels =getTrainDataset()
	
	#parameter which can tune
	#BATCH_SIZE 	= 128 
	#EPOCHS		= 100
	#SEED		= 777
	#SHAPE		= (192,192,4)
	#VAL_RATIO 	= 0.1
	#THRESHOLD	= 0.05
	
	#Split data into training and validation
	t_paths,t_labels,v_paths,v_labels = train_test_split(paths,labels,VAL_RATIO,SEED)
	
	#Data Generator 
	tg = ProteinDataGenerator(t_paths,t_labels,BATCH_SIZE,SHAPE)
	vg = ProteinDataGenerator(v_paths,v_labels,BATCH_SIZE,SHAPE)
	
	#model
	model = create_model(SHAPE)
	model.compile(
			loss='binary_crossentropy',
			optimizer=Adam(1e-03),
			metrics=['acc',f1])
	model.summary()

	checkpoint 		= ModelCheckpoint('./base.model', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
	reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')
	
	use_multiprocessing = False # DO NOT COMBINE MULTIPROCESSING WITH CACHE! 
	workers = 1 # DO NOT COMBINE MULTIPROCESSING WITH CACHE! 

	#fit data into model
	hist = model.fit_generator(
						tg,
						steps_per_epoch=len(tg),
						validation_data=vg,
						validation_steps=8,
						epochs=EPOCHS,
						use_multiprocessing=use_multiprocessing,
						workers=workers,
						verbose=1,
						callbacks=[checkpoint])
	