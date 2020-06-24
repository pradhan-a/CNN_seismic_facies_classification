#######################
# Facies prediction from seismic data by CNN-based semantic segmentation 
# Author: Anshuman Pradhan 
# Email: pradhan1@stanford.edu; pradhan.a269@gmail.com
#######################

# Main code for importing training data, defining the CNN architecture, training the CNN
# Code structure and organization adapted by Stanford CS230 starter project code
# Note: tf graph for CNN network is split over 2 GPUs 
#########################

import argparse
import logging
import os
import random

import tensorflow as tf
import numpy as np

from model.input_fn import input_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_fn import model_fn
from model.training import train_and_evaluate

# Command line arguments for specifying the path to data and model directories
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='Data',
                    help="Directory containing tfrecords files for training and validation set")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

if __name__ == '__main__':

    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    print(json_path)
    params = Params(json_path)

    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_filenames = os.path.join(data_dir, "Train.tfrecords")
    eval_filenames = os.path.join(data_dir, "Eval.tfrecords")


    # Create the two iterators over the two datasets
    train_inputs = input_fn(True, train_filenames, params)
    eval_inputs = input_fn(False, eval_filenames, params)


    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
