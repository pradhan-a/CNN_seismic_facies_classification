"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate, refmod_evaluate, abc_evaluate
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
parser.add_argument('--refmod_dir', default='Data/refmod',
                    help="Subdirectory of model dir or file containing the weights")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'abc.log'))

    # Create the input data pipeline
    logging.info("Creating the reference dataset...")

    num_ref=1 # Make sure not too big
    batch_size = params.batch_size
    params.batch_size = num_ref # Each epoch has one step

    refip_filenames = [os.path.join(args.refmod_dir, f) for f in os.listdir(args.refmod_dir) if f.startswith('seis')]
    refop_filenames = [os.path.join(args.refmod_dir, f) for f in os.listdir(args.refmod_dir) if f.startswith('facies')]

    refmod_filename = os.path.join(args.data_dir, "refmod.tfrecords")
    convert(refip_filenames, refop_filenames, refmod_filename)

    refmod_inputs = input_fn(False, refmod_filename, params)

    # Define the model
    logging.info("Creating the reference model...")
    refmod_spec = model_fn('eval', refmod_inputs, params, reuse=False)

    logging.info("Starting reference evaluation")
    ref_logits = refmod_evaluate(refmod_spec, args.model_dir, params, args.restore_from)

    tf.reset_default_graph()

    data_dir = args.data_dir
    test_filenames = os.path.join(data_dir, "Train.tfrecords")#IF you change this, change the next line

    params.eval_size = params.train_size
    params.batch_size = batch_size

    # create the iterator over the dataset
    test_inputs = input_fn(False, test_filenames, params)

    # Define the model
    logging.info("Creating the abc model...")
    model_spec = model_fn('eval', test_inputs, params, reuse=False)

    logging.info("Starting abc evaluation")
    abc_evaluate(model_spec, args.model_dir, params, args.restore_from, ref_logits)
