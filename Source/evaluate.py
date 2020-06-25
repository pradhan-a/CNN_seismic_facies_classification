#######################
# Facies prediction from seismic data by CNN-based semantic segmentation 
# Author: Anshuman Pradhan 
# Email: pradhan1@stanford.edu; pradhan.a269@gmail.com
#######################

# Functon for evaluating trained model on test set
#########################
"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger
from model.build import build_testset


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--record_path', default='Data/Test.tfrecords',
                    help="Path to tfrecords file to be evaluated")
parser.add_argument('--restore_from', default='best_weights',
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
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    test_filenames = args.record_path

    params.eval_size = params.dev_size

    # create the iterator over the dataset
    test_inputs = input_fn(False, test_filenames, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', test_inputs, params, reuse=False)

    # Evaluate the model
    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_from)
