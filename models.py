from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.backend as K
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import logging
import Config

import seq2seq
from seq2seq.models import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq



class Seq2SeqModel(object):

	def __init__(self, batch_dim, hidden_dim, output_dim, output_len, encode_depth, decode_depth,
		teacher_force, file_name):
		self.batch_dim = Config.BATCH_DIM
		self.hidden_dim = Config.HIDDEN_DIM
		self.output_dim = Config.OUTPUT_DIM
		self.output_len = Config.OUTPUT_LEN
		self.encode_depth = Config.ENCODE_DEPTH
		self.decode_depth = Config.DECODE_DEPTH
		self.teacher_force = Config.TEACHER_FORCE
		logging.basicConfig(level=logging.INFO)
    	self.logger = logging.getLogger(Config.FILE_NAME)



	def build_model(self, ckpt_path, phase):
		if phase == 'train':
			self.teacher_force = False
		elif phase == 'test':
			self.teacher_force = True

		self.model = Seq2Seq(batch_input_shape=self.batch_dim, hidden_dim=self.hidden_dim, 
								output_length= self.output_len, output_dim=self.output_dim, 
								depth=(self.encode_depth, self.decode_depth),
								teacher_force=self.teacher_force,inner_broadcast_state=True, peek=False)

		self.model.compile(loss='mse', optimizer='adam')
		if ckpt_path not None:
			self.model.load_weights(ckpt_path)

		return self.model

	def train_model(self, X_train, Y_train, X_val, Y_val):

		history = LossHistory()
		csv_logger = CSVLogger(train_log)
		checkpointer = ModelCheckpoint(filepath="model_checkpoint.hdf5", verbose=1, save_best_only=True)

		hist = self.model.fit(X_train, Y_train, 
          batch_size=self.batch_size, nb_epoch=Config.NUM_EPOCHS,shuffle='batch',
          verbose=1,			
          callbacks=[csv_logger, checkpointer],validation_data=(X_val, Y_val)) 

		self.model.save('model_model.hdf5')


	def test_model(self):
		print("Loading the model for testing and prediction purposes")
		model = self.model
		model.load_weights(modelPath)
		predictions = model.predict(X_test, batch_size = 32, verbose = 1)
		return predictions


