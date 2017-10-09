import keras
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from intervaltree import Interval,IntervalTree
from argparse import ArgumentParser
from multiprocessing import Pool

fs = 44100
clip_len = 10
# DATASET_DIR = 

def parseCommandLine():
	desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
	parser = ArgumentParser(description=desc)

	print("Parsing Command Line Arguments...")
	requiredTrain = parser.add_argument_group('Required Train/Test arguments')
	requiredTrain.add_argument('-p', choices = ["train", "test"], type = str,
						dest = 'phase', required = True, help = 'Training or Testing phase to be run')

	parser.add_argument('-ckpt', dest='ckpt_dir', default=None, 
									type=str, help='Set the checkpoint directory')
	parser.add_argument('-data', dest='data', default='/data/musicnet/', 
									type=str, help='Set the data directory')

	args = parser.parse_args()
	return args

def read_wav_as_np(filename):
	data = wav.read(filename)
	np_arr = data[1].astype('float32') / 32767.0 #Normalize 16-bit input to [-1, 1] range
	#np_arr = np.array(np_arr)
	return np_arr, data[0]

def time_blocks_to_fft_blocks(blocks_time_domain):
	fft_blocks = []
	for block in blocks_time_domain:
		fft_block = np.fft.fft(block)
		new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
		fft_blocks.append(new_block)
	return fft_blocks

def load_training_example(filename, block_size=2048, useTimeDomain=False):
	data, bitrate = read_wav_as_np(filename)
	x_t = convert_np_audio_to_sample_blocks(data, block_size)
	y_t = x_t[1:]
	y_t.append(np.zeros(block_size)) #Add special end block composed of all zeros
	if useTimeDomain:
		return x_t, y_t
	X = time_blocks_to_fft_blocks(x_t)
	Y = time_blocks_to_fft_blocks(y_t)
	return X, Y

def convert_np_audio_to_sample_blocks(song_np, block_size):
	block_lists = []
	total_samples = song_np.shape[0]
	num_samples_so_far = 0
	while(num_samples_so_far < total_samples):
		block = song_np[num_samples_so_far:num_samples_so_far+block_size]
		if(block.shape[0] < block_size):
			padding = np.zeros((block_size - block.shape[0],))
			block = np.concatenate((block, padding))
		block_lists.append(block)
		num_samples_so_far += block_size
	return block_lists

def convert_wav_files_to_nptensor(directory, block_size, max_seq_len, out_file, max_files=20, useTimeDomain=False):
	files = []
	for file in os.listdir(directory):
		if file.endswith('.wav'):
			files.append(directory+file)
	chunks_X = []
	chunks_Y = []
	num_files = len(files)
	if(num_files > max_files):
		num_files = max_files
	for file_idx in xrange(num_files):
		file = files[file_idx]
		print 'Processing: ', (file_idx+1),'/',num_files
		print 'Filename: ', file
		X, Y = load_training_example(file, block_size, useTimeDomain=useTimeDomain)
		cur_seq = 0
		total_seq = len(X)
		print total_seq
		print max_seq_len
		while cur_seq + max_seq_len < total_seq:
			chunks_X.append(X[cur_seq:cur_seq+max_seq_len])
			chunks_Y.append(Y[cur_seq:cur_seq+max_seq_len])
			cur_seq += max_seq_len
	num_examples = len(chunks_X)
	num_dims_out = block_size * 2
	if(useTimeDomain):
		num_dims_out = block_size
	out_shape = (num_examples, max_seq_len, num_dims_out)
	x_data = np.zeros(out_shape)
	y_data = np.zeros(out_shape)
	for n in xrange(num_examples):
		for i in xrange(max_seq_len):
			x_data[n][i] = chunks_X[n][i]
			y_data[n][i] = chunks_Y[n][i]
		print 'Saved example ', (n+1), ' / ',num_examples
	print 'Flushing to disk...'
	mean_x = np.mean(np.mean(x_data, axis=0), axis=0) #Mean across num examples and num timesteps
	std_x = np.sqrt(np.mean(np.mean(np.abs(x_data-mean_x)**2, axis=0), axis=0)) # STD across num examples and num timesteps
	std_x = np.maximum(1.0e-8, std_x) #Clamp variance if too tiny
	x_data[:][:] -= mean_x #Mean 0
	x_data[:][:] /= std_x #Variance 1
	y_data[:][:] -= mean_x #Mean 0
	y_data[:][:] /= std_x #Variance 1

	np.save(out_file+'_mean', mean_x)
	np.save(out_file+'_var', std_x)
	np.save(out_file+'_x', x_data)
	np.save(out_file+'_y', y_data)
	print 'Done!'

def parse_wav_files():
	block_size = freq / 4 #block sizes used for training - this defines the size of our input state
	max_seq_len = int(round((freq * clip_len) / block_size)) #Used later for zero-padding song sequences
	#Step 2 - convert WAVs to frequency domain with mean 0 and standard deviation of 1
	convert_wav_files_to_nptensor(new_directory, block_size, max_seq_len, output_filename)
	





