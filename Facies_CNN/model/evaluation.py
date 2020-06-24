"""Tensorflow utility functions for evaluation"""

import logging
import os

from tqdm import trange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model.utils import save_dict_to_json


def evaluate_sess(sess, model_spec, num_steps, writer=None, params=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
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

    """
    # Add summaries manually to writer at global_step_val
    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_step_val)
            """

    return metrics_val


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
        metrics = evaluate_sess(sess, model_spec, num_steps)
        metrics_name = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir, "metrics_test_{}.json".format(metrics_name))
        save_dict_to_json(metrics, save_path)



def plot_pred(model_spec, model_dir, params, restore_from,rnd_idx):
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
            print(save_path)
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        sess.run(model_spec['iterator_init_op'])
        sess.run(model_spec['metrics_init_op'])
        # Plot
        pred = model_spec['predictions']
        facies = model_spec['Facies']
        logits = model_spec['logits']
        predd, fac, logitss = sess.run([pred,facies,logits])
        logitss=np.exp(logitss)/np.sum(np.exp(logitss),3,keepdims=True)

        for i in range(predd.shape[0]):
            plt.figure()
            plt.subplot(121)
            plt.imshow(np.squeeze(predd[i,:,:]))
            plt.title('Predicted')
            plt.subplot(122)
            plt.imshow(np.squeeze(fac[i,:,:,:]))
            plt.title('True model#'+str(rnd_idx[i]))
            plt.figure()
            plt.imshow(np.squeeze(logitss[i,:,:,1]))
            plt.title('For model#'+str(rnd_idx[i]))
            plt.colorbar()
            plt.figure()
            plt.imshow(np.squeeze(logitss[i,:,:,2]))
            plt.title('For model#'+str(rnd_idx[i]))
            plt.colorbar()
            #np.savetxt('lccdf_testmodel#'+str(rnd_idx[i]+1)+'fac1'+'.txt',np.squeeze(logitss[i,:,:,1]),delimiter=',')
            #np.savetxt('lccdf_testmodel#'+str(rnd_idx[i]+1)+'fac2'+'.txt',np.squeeze(logitss[i,:,:,2]),delimiter=',')
            #np.savetxt('predicted_testmodel#'+str(rnd_idx[i])+'.txt',np.squeeze(predd[i,:,:,:]),delimiter=',')
        plt.show()



def plot_vis(model_spec, model_dir, params, restore_from,rnd_idx):
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

        sess.run(model_spec['iterator_init_op'])
        sess.run(model_spec['metrics_init_op'])
        # Plot
        pred = model_spec['predictions']
        facies = model_spec['Facies']
        logits = model_spec['logits']
        activatns = model_spec['activatns']
        predd, fac, logitss, activatnss = sess.run([pred,facies,logits,activatns])
        logitss=np.exp(logitss)/np.sum(np.exp(logitss),3,keepdims=True)

        for i in range(predd.shape[0]):
            plt.figure()
            plt.subplot(121)
            plt.imshow(np.squeeze(predd[i,:,:]))
            plt.title('Predicted')
            plt.subplot(122)
            plt.imshow(np.squeeze(fac[i,:,:,:]))
            plt.title('True model#'+str(rnd_idx[i]))
            """
            plt.figure()
            plt.imshow(np.squeeze(logitss[i,:,:,1]))
            plt.title('For model#'+str(rnd_idx[i]))
            plt.colorbar()
            plt.figure()
            plt.imshow(np.squeeze(logitss[i,:,:,2]))
            plt.title('For model#'+str(rnd_idx[i]))
            plt.colorbar()
            """
            for k in range(activatnss.shape[0]):
                plt.figure()
                for j in np.arange(16):
                    plt.subplot(4,4,j+1)
                    plt.imshow(np.squeeze(activatnss[k,i,:,:,j+16]))
                for j in np.arange(activatnss.shape[-1]):
                    np.savetxt('./Results/model#'+str(rnd_idx[i])+'L'+str(k+1)+'act'+str(j+1)+'.txt',np.squeeze(activatnss[k,i,:,:,j]),delimiter=',')


            #np.savetxt('lccdf_testmodel#'+str(rnd_idx[i])+'.txt',np.squeeze(logitss[i,:,:,:]),delimiter=',')
            #np.savetxt('predicted_testmodel#'+str(rnd_idx[i])+'.txt',np.squeeze(predd[i,:,:,:]),delimiter=',')
        plt.show()



def refmod_evaluate(model_spec, model_dir, params, restore_from):
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

        sess.run(model_spec['iterator_init_op'])
        sess.run(model_spec['metrics_init_op'])
        pred = model_spec['predictions']
        logits = model_spec['logits']
        logitss = sess.run(logits)
        logitss=1/(1+np.exp(-logitss))

    return logitss #predd



def abc_evaluate(model_spec, model_dir, params, restore_from, ref_logits):
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

        num_post=50
        post_dist=1e20*np.ones((num_post,))
        post_samp=[np.zeros((params.output_size_x,params.output_size_y)) for _ in range(num_post)]
        #post_samp=np.zeros((num_post,params.output_size_x,params.output_size_y))


        logits = model_spec['logits']
        pred = model_spec['predictions']
        facies = model_spec['Facies']
        global_step = tf.train.get_global_step()
        # Load the evaluation dataset into the pipeline and initialize the metrics init op
        sess.run(model_spec['iterator_init_op'])
        sess.run(model_spec['metrics_init_op'])
        # compute metrics over the dataset
        for iter1 in range(num_steps):
            logitss, fac = sess.run([logits,facies])
            logitss=1/(1+np.exp(-logitss))
            #dist_fac=list(np.sum(np.absolute(predd-ref_pred),(1,2)))
            epsilon=1e-6
            logitss[logitss==1]=logitss[logitss==1]-epsilon
            logitss[logitss==0]=logitss[logitss==0]+epsilon
            ref_logits[ref_logits==1]=ref_logits[ref_logits==1]-epsilon
            ref_logits[ref_logits==0]=ref_logits[ref_logits==0]+epsilon
            dist_kl=(ref_logits*np.log(ref_logits/logitss))+((1-ref_logits)*np.log((1-ref_logits)/(1-logitss)))
            dist_fac=np.sum(dist_kl,(1,2))

            for i, d in enumerate(dist_fac):
                ix_arr=np.argwhere(post_dist>d)
                if ix_arr.shape[0]!=0:
                    #print(ix_arr.shape,post_dist.shape,d)
                    ix=int(ix_arr[0])
                    post_dist=list(post_dist)
                    post_dist.insert(ix,d)
                    post_dist.pop()
                    assert len(post_dist)==num_post
                    post_dist=np.reshape(np.array(post_dist),(num_post,))

                    post_samp.insert(ix,np.squeeze(fac[i,:,:]))
                    post_samp.pop()
                    assert len(post_samp)==num_post

                    """
                    if ix==0:
                        post_samp=np.concatenate((np.reshape(fac[i,:,:],(1,params.output_size_x,params.output_size_y)),post_samp[:-1,:,:]))
                    elif ix==1:
                        a1=np.concatenate((np.reshape(post_samp[0,:,:],(1,params.output_size_x,params.output_size_y)),np.reshape(fac[i,:,:],(1,params.output_size_x,params.output_size_y))))
                        a1=np.concatenate((a1,post_samp[ix:-1,:,:]))
                        post_samp=a1
                    else:
                        a1=np.concatenate((post_samp[0:ix,:,:],np.reshape(fac[i,:,:],(1,params.output_size_x,params.output_size_y))))
                        a1=np.concatenate((a1,post_samp[ix:-1,:,:]))
                        post_samp=a1
                        """
                        
                    #print(ix,post_samp.shape)
                    #assert post_samp.shape == (num_post,params.output_size_x,params.output_size_y)
                    #iter_string = " ; ".join("{}: {:05.2f}".format(iter1))
            logging.info("Step# " + str(iter1+1)+ " out of " + str(num_steps))

        np.savetxt('post_dist_KLdiv1.txt',post_dist,delimiter=',')
        np.savetxt('post_samp_KLdiv1.txt',np.reshape(np.array(post_samp),(num_post,params.output_size_x*params.output_size_y)),delimiter=',')
        etype=np.mean(np.array(post_samp),axis=0)
        plt.figure()
        plt.imshow(np.squeeze(etype))
        plt.figure()
        plt.imshow(post_samp[0])
        plt.figure()
        plt.scatter(np.arange(num_post),post_dist)
        plt.show()




