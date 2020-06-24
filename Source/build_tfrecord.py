import argparse
import logging
import os
import random

import numpy as np
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json

from tfrecord_utils import wrap_bytes, convert

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='Data/Train_seis',
                    help="Directory containing inputs")
parser.add_argument('--output_dir', default='Data/Train_fac',
                    help="Directory containing outputs")
parser.add_argument('--record_path', default='Data/Train_fac',
                    help="Path to output file")# of the form '../../filename.tfrecords'
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")

if __name__ == '__main__':

	args = parser.parse_args()

	json_path = os.path.join(args.model_dir, 'params.json')
	assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
	print(json_path)
	params = Params(json_path)

	input_filenums=[int(f[len('seis_'):-len('.txt')]) for f in os.listdir(args.input_dir) if f.endswith('.txt') ]
	output_filenums=[int(f[len('facies_'):-len('.txt')]) for f in os.listdir(args.output_dir) if f.endswith('.txt') ]

	input_filenames = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.txt')]
	output_filenames = [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if f.endswith('.txt')]
	input_filenames=[input_filenames[ix] for ix in np.argsort(input_filenums)]
	output_filenames=[output_filenames[ix] for ix in np.argsort(output_filenums)]
	convert(input_filenames, output_filenames, args.record_path, params)




