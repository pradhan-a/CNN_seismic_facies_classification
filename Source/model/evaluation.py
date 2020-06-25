#######################
# Facies prediction from seismic data by CNN-based semantic segmentation 
# Author: Anshuman Pradhan 
# Email: pradhan1@stanford.edu; pradhan.a269@gmail.com
#######################

# Utility functions for evaluation
#########################

import logging
import os

#from tqdm import trange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model.utils import save_dict_to_json


def evaluate_sess(sess, model_dir, model_spec, num_steps, writer=None, params=None):
    """Evauate metrics without Monte-Carlo dropout

    Args:
        sess: (tf.Session) current session
        model_dir: directory containing params.json
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    # compute metrics over the dataset
    for _ in range(num_steps):
        sess.run(update_metrics)

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)

    return metrics_val


def evaluate_sess_epistemic(sess, model_dir, model_spec, num_steps, num_samples, writer=None, params=None):
    """Evauate metrics with Monte-Carlo dropout (MCD)
    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        num_samples: Number of Monte-Carlo dropout samples
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """

    facies = model_spec['Facies']
    logits = model_spec['logits']


    facies_list=[]
    prob_list=[]


    # Loop over each MCD sample
    for i in range(num_samples):
        print('Working on sample #'+str(i))

        # Load the evaluation dataset into the pipeline and initialize the metrics init op
        sess.run(model_spec['iterator_init_op'])
        sess.run(model_spec['metrics_init_op'])

        # compute metrics over the dataset
        for j in range(num_steps):
                fac, logitss = sess.run([facies,logits])
                logitss=np.exp(logitss)/np.sum(np.exp(logitss),-1,keepdims=True)
                if i==0:
                    facies_list.append(fac)
                    prob_list.append(logitss)
                else:
                    prob_list[j]=prob_list[j]+logitss

    # Average probabilities over number of MC samples
    prob_list=[p/num_samples for p in prob_list]
    for i in range(len(prob_list)):
        if i==0:
            probabilities=prob_list[i]
            facies=facies_list[i]
        else:
            probabilities=np.concatenate((probabilities,prob_list[i]),axis=0)
            facies=np.concatenate((facies,facies_list[i]),axis=0)
    # Most likely facies predictions
    predictions=np.argmax(probabilities,axis=-1)
    facies=np.squeeze(facies)
    accuracy = np.mean(np.equal(predictions,facies))

    eval_metrics={'accuracy': accuracy}

    # Get the values of the metrics
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in eval_metrics.items())
    logging.info("- Eval metrics: " + metrics_string)

    #np.savetxt(model_dir+'/lccdftestmodel.txt',np.squeeze(prob_list2[:,:4,:,:,1]).flatten(order='F'),delimiter=',')
    #np.savetxt(model_dir+'/true_testmodel.txt',np.squeeze(facies[:4,:,:]).flatten(order='F'),delimiter=',')
    #np.savetxt(model_dir+'/predicted_testmodel.txt',np.squeeze(predictions[:4,:,:]).flatten(order='F'),delimiter=',')
    return None


def evaluate(model_spec, model_dir, params, restore_from):
    """Evaluate the model

    Args:
        model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        # Evaluate
        num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size

        if params.use_epistemic:
            metrics = evaluate_sess_epistemic(sess, model_dir, model_spec, num_steps, params.num_samples_epistemic)
        else:
            metrics = evaluate_sess(sess, model_dir, model_spec, num_steps)

        metrics_name = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
        save_dict_to_json(metrics, save_path)