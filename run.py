import keras
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import parseCommandLine, load_data




def create_model(args):
	curModel = Seq2SeqModel()
	curModel.build_model(args.ckpt_dir, args.phase)

	return curModel


def main(args):
	dataset, keys, files = load_data(args.data)
	curModel = create_model(args)
	



if __name__ == '__main__':
	args = parseCommandLine()
	main(args)