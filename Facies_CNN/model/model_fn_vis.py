"""Define the model."""

import tensorflow as tf
import numpy as np

def build_baseline_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    seis = inputs['Seis']

    assert seis.get_shape().as_list() == [None, params.input_size_x, params.input_size_y,1]

    out = seis
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_layers = params.num_layers
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels_encoder = [num_channels*(2**i) for i in np.arange(num_layers)]
    channels_decoder=[32,16,16,1]

    for i, c in enumerate(channels_encoder):
        with tf.variable_scope('encoder_block{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, 1, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            if i==0:
                out = tf.layers.max_pooling2d(out, [2,1], [2,1])
            else:
                out = tf.layers.max_pooling2d(out, 2, 2)

    assert out.get_shape().as_list() == [None, 30, 37, num_channels * (2**(num_layers-1))]
    """
    for i, c in enumerate(channels_decoder):
        with tf.variable_scope('decoder{}'.format(i+1)):
            
            if i%2==0:
                out = tf.layers.conv2d_transpose(out, c, 2, 2, padding='same')
                if params.use_batch_norm:
                    out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
                out = tf.nn.relu(out)
            else:
                out = tf.layers.conv2d_transpose(out, c, 3, 1, padding='same')
                if params.use_batch_norm:
                    out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            """
    out = tf.layers.conv2d_transpose(out, 32, (2,3), 2, padding='valid')
    if params.use_batch_norm:
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
    out = tf.nn.relu(out)
    out = tf.layers.conv2d_transpose(out, 32, 3, 1, padding='same')
    if params.use_batch_norm:
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
    out = tf.layers.conv2d_transpose(out, 16, 2, 2, padding='valid')
    if params.use_batch_norm:
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
    out = tf.nn.relu(out)
    out = tf.layers.conv2d_transpose(out, params.num_labels, 3, 1, padding='same')
    if params.use_batch_norm:
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)


    logits=out

    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    facies = tf.cast(tf.squeeze(inputs['Facies']),tf.int64)
    #facies = tf.cast(facies, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        if params.model_version=='baseline':
            logits = build_baseline_model(is_training, inputs, params)
        elif params.model_version=='deeperconv':
            logits = build_deeperconv_model(is_training, inputs, params)
        elif params.model_version=='dilatedconv':
            logits, activatns = build_dilatedconv_model(is_training, inputs, params)
        predictions = tf.argmax(logits, -1)
        assert predictions.get_shape().as_list() == [None, params.output_size_x, params.input_size_y]

    # Define loss and accuracy
    loss = tf.reduce_sum(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = facies),0))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions,facies), tf.float32))
    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


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

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    
    mask = tf.not_equal(facies, predictions)

    # Add a different summary to know how they were misclassified
    """
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['Seis'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)
        """
    
    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec["logits"] = logits
    model_spec["activatns"] = activatns
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec



def build_deeperconv_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    seis = inputs['Seis']

    assert seis.get_shape().as_list() == [None, params.input_size_x, params.input_size_y,1]

    out = seis
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_layers = params.num_layers
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels_encoder = [num_channels*(2**i) for i in np.arange(num_layers)]
    #channels_decoder=[]
    #for i in channels_encoder[::-1]:
    #    channels_decoder.extend([i,i])
    channels_decoder=[32,16,16,1]
    for i, c in enumerate(channels_encoder):
        with tf.variable_scope('encoder_block{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, 1, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, c, 3, 1, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            if i==0:
                out = tf.layers.max_pooling2d(out, [2,1], [2,1])
            else:
                out = tf.layers.max_pooling2d(out, 2, 2)

    assert out.get_shape().as_list() == [None, 30, 37, num_channels * (2**(num_layers-1))]
    """
    for i, c in enumerate(channels_decoder):
        with tf.variable_scope('decoder{}'.format(i+1)):
            
            if i%2==0:
                out = tf.layers.conv2d_transpose(out, c, 2, 2, padding='same')
                if params.use_batch_norm:
                    out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
                out = tf.nn.relu(out)
            else:
                out = tf.layers.conv2d_transpose(out, c, 3, 1, padding='same')
                if params.use_batch_norm:
                    out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            """
    out = tf.layers.conv2d_transpose(out, 32, (2,3), 2, padding='valid')
    if params.use_batch_norm:
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
    out = tf.nn.relu(out)
    out = tf.layers.conv2d_transpose(out, 32, 3, 1, padding='same')
    if params.use_batch_norm:
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
    out = tf.layers.conv2d_transpose(out, 16, 3, 1, padding='same')
    if params.use_batch_norm:
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
    out = tf.layers.conv2d_transpose(out, 16, 2, 2, padding='valid')
    if params.use_batch_norm:
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
    out = tf.nn.relu(out)
    out = tf.layers.conv2d_transpose(out, 16, 3, 1, padding='same')
    if params.use_batch_norm:
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
    out = tf.layers.conv2d_transpose(out, params.num_labels, 3, 1, padding='same')
    if params.use_batch_norm:
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)


    logits=out

    return logits


def build_dilatedconv_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    seis = inputs['Seis']

    assert seis.get_shape().as_list() == [None, params.input_size_x, params.input_size_y,1]

    #activatns_list=[]
    out = seis
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_layers = params.num_layers
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels_encoder = [num_channels*(2**i) for i in np.arange(num_layers)]

    for i, c in enumerate(channels_encoder):
        with tf.variable_scope('encoder_block{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, 1, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            if i==num_layers-1:
                out = tf.layers.max_pooling2d(out, [2,1], [2,1])
    #activatns_list.append(out[:,:,:,0])
    activatns1=out

    num_dilate_layers=params.num_dilate_layers
    channels_dilate=[channels_encoder[-1] for i in np.arange(num_dilate_layers)]
    dilate_factors=[2**(i+1) for i in np.arange(num_dilate_layers)]

    for i, c in enumerate(channels_dilate):
        with tf.variable_scope('dilate_block{}'.format(i+1)):
            d=dilate_factors[i]
            out = tf.layers.conv2d(out, c, 3, 1, padding='same',dilation_rate=(d,d))
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            if i==2:
                #activatns_list.append(out[:,:,:,0])
                activatns2=out
            if i==3:
                #activatns_list.append(out[:,:,:,0])
                activatns3=out

    #activatns_list.append(out[:,:,:,0])
    #activatns=tf.stack(activatns_list,axis=3)
    activatns4=out
    activatns=tf.stack([activatns1,activatns2])



    out = tf.layers.conv2d(out, 16, 1, 1, padding='same')
    if params.use_batch_norm:
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
    out = tf.nn.relu(out)

    out = tf.layers.conv2d(out, params.num_labels, 1, 1, padding='same')
    if params.use_batch_norm:
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)

    logits=out

    return logits, activatns

