#######################
# Facies prediction from seismic data by CNN-based semantic segmentation 
# Author: Anshuman Pradhan 
# Email: pradhan1@stanford.edu; pradhan.a269@gmail.com
#######################
# Utility functions for building tfrecords files from 
#########################
import numpy as np
import tensorflow as tf

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(seis_paths, fac_paths, out_path, params):
    # Args:
    # seis_paths   List of file-paths for the seismic data.
    # fac_paths        Facies labels for the seismic data.
    # out_path      File-path for the TFRecords output file.
    
    print("Converting: " + out_path)
    
    # Number of examples. Used when printing the progress.
    num_images = len(seis_paths)
    
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        
        # Iterate over all the seis_paths and fac_paths
        for i, (spath, fpath) in enumerate(zip(seis_paths, fac_paths)):
            print("Working on example#"+str(i+1))

            # Load the data and reshape it into 3D volumes
            seis = np.loadtxt(spath,delimiter=',')
            seis=np.reshape(seis,(params.input_size_z,params.input_size_x,params.input_size_y),order='F')

            fac = np.loadtxt(fpath,delimiter=',')
            fac=np.reshape(fac,(params.output_size_z,params.output_size_x,params.output_size_y),order='F')
            
            # Convert the data to raw bytes.
            seis_bytes = seis.tostring()
            fac_bytes = fac.tostring()
            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'seis': wrap_bytes(seis_bytes),
                    'fac': wrap_bytes(fac_bytes)
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()
            
            # Write the serialized data to the TFRecords file.
            writer.write(serialized)