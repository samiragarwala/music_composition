import numpy as np  
import matplotlib.pyplot as plt 
from scipy import fft
from intervaltree import Interval,IntervalTree
import os
import random


fs = 44100
seq_len = 128
stride = 512                         # 512 samples between windows
wps = fs/float(stride)               # ~86 windows/second
num_notes = 128
num_div = 4
frame_len = fs/num_div


class MusicNetDataset(object):

	def __init__(self):
		self.fs = 44100
		self.seq_len = 128
		self.stride = 512                         # 512 samples between windows
		self.wps = self.fs/float(self.stride)     # ~86 windows/second
		self.num_notes = 128
		self.num_div = 4
		self.frame_len = self.fs/self.num_div
		self.temp_name = '/diskhdd/musicData/FrameArray.mmap'
		self.final_name = '/diskhdd/musicData/RawAudioDataset'
		self.num_recs = 330
		self.cur_frame = 0
		self.batch_done = True

	def create_raw_audio_dataset(train_data):
		rec_ids = train_data.keys()
		cntr = 0
		frames_done = 0
		frame_arr = np.memmap(self.temp_name, dtype='float32', mode='w+', shape=(self.num_recs*3000, self.frame_len))

		for key in rec_ids:
			X,Y = train_data[key]
			x_len = len(X)

			for i in xrange(0, x_len, self.frame_len):
				cur_frame = X[i:i+self.frame_len]
				if i+self.frame_len >= x_len:
					padding = np.zeros(i+self.frame_len-x_len)
					cur_frame = np.append(cur_frame, padding)

				frame_arr[frames_done, :] = cur_frame
				frames_done += 1

			cntr += 1
			print("Finished {0} IDs with cur ID {1}".format(cntr, key))

		self.finalData = np.memmap(self.final_name + '_' + str(self.num_div) + '.mmap', dtype='float32', 
											mode='w+', shape=(frames_done, self.frame_len))
		self.final_shape = (frames_done, self.frame_len)

		self.finalData[:,:] = frame_arr[:frames_done, :] 
		del frame_arr
		os.remove(self.temp_name)
		del finalData

	def get_current_batch():
		if self.batch_done:
			self.cur_frame = 0
			shuff_indx = random.shuffle(np.arange(self.final_shape[0]))
			self.finalData[:,:] = self.finalData[shuff_indx,:]
			self.batch_done = False

		curInputData = self.finalData[self.cur_frame:self.cur_frame+self.seq_len, :]
		curTargetData = self.finalData[self.cur_frame+1:self.cur_frame+self.seq_len+1, :]

		self.cur_frame += self.seq_len
		if (self.cur_frame + self.seq_len+1) > self.final_shape[0]:
			self.batch_done = True
		return curInputData, curTargetData



def create_notes_dataset(train_data):
	rec_ids = train_data.keys()
	note_list = []
	cntr = 0
	for key in rec_ids:
		X,Y = train_data[key]
		x_len = len(X)
		y_len = len(Y)
		num_sec = x_len/fs

		cur_rec_data = []
		for window in range(int(num_sec*wps)):
			labels = Y[window*stride]
			cur_win_label = np.zeros((1, num_notes))
			for label in labels:
				if label.data[0] > 8:
					continue

				cur_win_label[0, label.data[1]] = 1

			cur_rec_data.append(cur_win_label)

		cur_rec_data = np.concatenate(cur_rec_data, axis=0)
		note_list.append(cur_rec_data)
		cntr += 1
		print("Completed {0} recordings with current key ID {1}".format(cntr, key))

	note_list = np.concatenate(note_list, axis=0)
	np.save('../data/notes_dataset_' + str(stride) + '.npy', note_list)
	return note_list


def create_raw_audio_dataset(train_data):
	rec_ids = train_data.keys()
	frame_list = []
	cntr = 0
	frames_done = 0
	frame_arr = np.memmap('/diskhdd/musicData/FrameArray.mmap', dtype='float32', mode='w+', shape=(330*3000, frame_len))

	for key in rec_ids:
		X,Y = train_data[key]
		x_len = len(X)

		for i in xrange(0, x_len, frame_len):
			cur_frame = X[i:i+frame_len]
			if i+frame_len >= x_len:
				padding = np.zeros(i+frame_len-x_len)
				cur_frame = np.append(cur_frame,padding)

			frame_arr[frames_done, :] = cur_frame
			frames_done += 1

		cntr += 1
		print("Finished {0} IDs with cur ID {1}".format(cntr, key))

	finalData = np.memmap('/diskhdd/musicData/RawAudioDataset_' + str(num_div) + '.mmap', dtype='float32', 
										mode='w+', shape=(frames_done, frame_len))
	finalData[:,:] = frame_arr[:frames_done, :] 
	del frame_arr
	os.remove('/diskhdd/musicData/FrameArray.mmap')
	del finalData
	




def main():
	train_data = np.load(open('../data/musicnet.npz','rb'))
	# create_notes_dataset(train_data)
	create_raw_audio_dataset(train_data)


if __name__ == '__main__':
	main()
