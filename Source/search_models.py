"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys
import numpy as np
from model.utils import Params


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir1', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--parent_dir2', default='experiments/deeperconv_model',
                    help="Directory containing params.json")
parser.add_argument('--parent_dir3', default='experiments/dilatedconv_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='Data',
                    help="Directory containing the dataset")


def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir {model_dir} --data_dir {data_dir}".format(python=PYTHON,
            model_dir=model_dir, data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    
    json_path1 = os.path.join(args.parent_dir1, 'params.json')
    assert os.path.isfile(json_path1), "No json configuration file found at {}".format(json_path1)
    params1 = Params(json_path1)

    json_path2 = os.path.join(args.parent_dir2, 'params.json')
    assert os.path.isfile(json_path2), "No json configuration file found at {}".format(json_path2)
    params2 = Params(json_path2)

    json_path3 = os.path.join(args.parent_dir3, 'params.json')
    assert os.path.isfile(json_path3), "No json configuration file found at {}".format(json_path3)
    params3 = Params(json_path3)

    # Perform hypersearch over one parameter
    learning_rates = (10**(-3.5+(-1.5+3.5)*np.random.rand(10))).tolist()
    #learning_rates = [1e-3, 1e-2]
    #learning_rates = [256, 512]


    for learning_rate in learning_rates:
        job_name = "learning_rate_{}".format(learning_rate)
        #job_name = "batch_size_{}".format(learning_rate)
        
        #if learning_rate != 1e-3:
            # Launch job (name has to be unique)
            # Modify the relevant parameter in params
        #params1.learning_rate = learning_rate
        #launch_training_job(args.parent_dir1, args.data_dir, job_name, params1)  

        #params2.learning_rate = learning_rate
        #params2.batch_size = learning_rate
        #launch_training_job(args.parent_dir2, args.data_dir, job_name, params2)

        params3.learning_rate = learning_rate
        launch_training_job(args.parent_dir3, args.data_dir, job_name, params3)


