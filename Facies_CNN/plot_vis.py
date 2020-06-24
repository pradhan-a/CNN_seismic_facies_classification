"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf
import numpy as np

from model.input_fn import input_fn
from model.model_fn_vis import model_fn
from model.evaluation import plot_vis
from model.utils import Params
from model.utils import set_logger
from model.build import build_testset
from tfrecord_utils import wrap_bytes, convert


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='Data',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirectory of model dir or file containing the weights")
parser.add_argument('--input_dir', default='Data/Test_seis',
                    help="Directory containing inputs")
parser.add_argument('--output_dir', default='Data/Test_fac',
                    help="Directory containing outputs")

if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'plot.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    num_plot=2 # Make sure not too big
    params.batch_size = num_plot # Each epoch has one step
    params.eval_size=5 # Added for TI90 cases
    #params.eval_size = 100
    # Load dataset
    input_filenums=[int(f[len('seis_'):-len('.txt')]) for f in os.listdir(args.input_dir) if f.endswith('.txt') ]
    output_filenums=[int(f[len('facies_'):-len('.txt')]) for f in os.listdir(args.output_dir) if f.endswith('.txt') ]

    input_filenames = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.txt')]
    output_filenames = [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if f.endswith('.txt')]

    input_filenames=[input_filenames[ix] for ix in np.argsort(input_filenums)]
    output_filenames=[output_filenames[ix] for ix in np.argsort(output_filenums)]

    rnd_idx=np.random.randint(params.eval_size,size=num_plot)
    rnd_idx[0]=2084
    rnd_idx[-1:]=765
    input_filenames=list(input_filenames[i] for i in rnd_idx)
    output_filenames=list(output_filenames[i] for i in rnd_idx)
    rnd_idx=rnd_idx+1

    plot_filename = os.path.join(args.data_dir, "plot.tfrecords")
    convert(input_filenames, output_filenames, plot_filename)

    #test_fac_filenames=test_fac_filenames[rnd_idx]
    # create the iterator over the dataset
    test_inputs = input_fn(False, plot_filename, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', test_inputs, params, reuse=False)

    logging.info("Plotting")
    plot_vis(model_spec, args.model_dir, params, args.restore_from, rnd_idx)
