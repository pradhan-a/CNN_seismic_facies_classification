#######################
# Facies prediction from seismic data by CNN-based semantic segmentation 
# Author: Anshuman Pradhan 
# Email: pradhan1@stanford.edu; pradhan.a269@gmail.com
#######################

# Model function defining the CNN network architecture for semantic segmentation
#########################


import tensorflow as tf
import numpy as np

def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (seismic, facies labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    facies = tf.cast(tf.squeeze(inputs['Facies']),tf.int64)

    # -----------------------------------------------------------
    # BUILD THE MODEL
    with tf.variable_scope('model', reuse=reuse):
        # Compute logits from network with seismic as input
        logits = build_dilatedconv_model_gradual(is_training, inputs, params)
        # Compute predicted (most likely) facies labels
        predictions = tf.argmax(logits, -1)
        assert predictions.get_shape().as_list() == [None, params.output_size_z, params.output_size_x, params.output_size_y]

    # Define loss and accuracy
    loss = tf.reduce_sum(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = facies),0))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions,facies), tf.float32))
    
    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            #
            # Add a dependency to update the moving mean and variance for batch normalization
            # Set colocate_gradients_with_ops=True so that network graph may be split seamlessly over multiple GPUs
            #
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step,colocate_gradients_with_ops=True)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step,colocate_gradients_with_ops=True)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=facies, predictions=predictions),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['Seis'])

    
    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec["logits"] = logits
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec



def build_dilatedconv_model_gradual(is_training, inputs, params):
    """Function defining the network graph
        Computes logits from the model given input seismic

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    seis = inputs['Seis']
    assert seis.get_shape().as_list() == [None, params.input_size_z, params.input_size_x, params.input_size_y, params.input_size_c]
    out = seis

    bn_momentum = params.bn_momentum

    # Network divided into blocks for easily splitting over the two GPUs being used
    # Note some empirical experimentation required to decide optimal memory splitting over the two GPUs

    num_conv_layers_block1=2
    dilate_factors_block1=[1,1] # No dilation of conv filters for first two layers

    num_conv_layers_block2=2
    dilate_factors_block2=[2,4] # Increase dilation exponentially

    num_conv_layers_block3=0
    dilate_factors_block3=[8]

    num_conv_layers_block4=3
    dilate_factors_block4=[8,16,32]

    #Number of channels kept constant in each layer
    num_channels=params.num_channels


    # Graph for first gpu
    with tf.device('/gpu:0'):

        # conv3d layers in first block
        for j in range(num_conv_layers_block1):
            d=dilate_factors_block1[j]
            out = tf.layers.conv3d(out, num_channels, 3, 1, padding='same',dilation_rate=(d,d,d))
            if params.use_batch_norm: # Flag for using batch-norm
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            if params.use_epistemic: # Flag for using Monte-Carlo dropout
                out=tf.nn.dropout(out,keep_prob=params.dropout_keep_prob)

        # first conv3d_transpose layer to go upsample time dimension of seismic to depth dimension of facies volumes
        out=tf.layers.conv3d_transpose(out, num_channels, (33,1,1), 1, padding='valid')#30 for nt=66, 26 for nt=75
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        if params.use_epistemic: # Flag for using Monte-Carlo dropout
            out=tf.nn.dropout(out,keep_prob=params.dropout_keep_prob)

        # conv3d layers in second block
        for j in range(num_conv_layers_block2):
            d=dilate_factors_block2[j]
            out = tf.layers.conv3d(out, num_channels, 3, 1, padding='same',dilation_rate=(d,d,d))
            if params.use_batch_norm: # Flag for using batch-norm
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            if params.use_epistemic: # Flag for using Monte-Carlo dropout
                out=tf.nn.dropout(out,keep_prob=params.dropout_keep_prob)

    # Graph for second gpu
    with tf.device('/gpu:1'):

        # conv3d layers in third block
        for j in range(num_conv_layers_block3):
            d=dilate_factors_block3[j]
            out = tf.layers.conv3d(out, num_channels, 3, 1, padding='same',dilation_rate=(d,d,d))
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            if params.use_epistemic:
                out=tf.nn.dropout(out,keep_prob=params.dropout_keep_prob)

        # second conv3d_transpose layer to go upsample time dimension of seismic to depth dimension of facies volumes
        out=tf.layers.conv3d_transpose(out, num_channels, (68,1,1), (2,1,1), padding='valid')#62 for nt=66, 52 for nt=71
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        if params.use_epistemic:
            out=tf.nn.dropout(out,keep_prob=params.dropout_keep_prob)

        # conv3d layers in fourth block
        for j in range(num_conv_layers_block4):
            d=dilate_factors_block4[j]
            out = tf.layers.conv3d(out, num_channels, 3, 1, padding='same',dilation_rate=(d,d,d))
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            if params.use_epistemic:
                out=tf.nn.dropout(out,keep_prob=params.dropout_keep_prob)

        # Final conv3D layer to predict logts for each facies class
        out = tf.layers.conv3d(out, params.num_labels, 3, 1, padding='same')
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)

        logits=out

    return logits
