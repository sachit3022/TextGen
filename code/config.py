# config.py

import re
import os
import json
import argparse
import datetime
import configparser
from ast import literal_eval as make_tuple

from utils import str2bool

def parse_args():
    
    result_path = "results/"
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_path = os.path.join(result_path, now)

    # parse the arguments
    parser = argparse.ArgumentParser(description="Train Seq2Seq")
    
    # the following two parameters can only be provided at the command line.
    parser.add_argument('--result-path', type=str, default=result_path, metavar='', help='full path to store the results')
    parser.add_argument("-c", "--config", "--args-file", dest="config_file", default="args.txt", help="Specify a config file", metavar="FILE")
    args, remaining_argv = parser.parse_known_args()
    
    # ======================= Experiment Settings ===========================
    parser.add_argument('--project_name', default='Seq2Seq', help="Name of the Project")
    parser.add_argument('--save_dir', default='checkpoints', help="directory to save results")
    parser.add_argument('--logs_dir', default='logs', help="directory to save results")
    parser.add_argument('--checkpoint-max-history', type=int, default=10, metavar='', help='max checkpopint history')
    parser.add_argument('-s', '--save', '--save-results', type=str2bool, dest="save_results",default='Yes', metavar='', help='save the arguments and the results')
    
    # ======================= Dataset Settings ============================
    parser.add_argument('--dataset', choices=["pig_latin"], help="dataset to run")
    parser.add_argument('--dataset-options', type=json.loads, default={}, help="options for loading dataset")
    
    # ======================= Network Model Settings ============================
    parser.add_argument('--model-type', choices=["Transformer", "S4", "Mamba"], type=str, default=None, help='type of network')
    parser.add_argument('--model-options', type=json.loads, default={}, metavar='', help='additional model-specific parameters, i.e. \'{"hidden_size": 1}\'')
    parser.add_argument('--loss-type', type=str, default=None, help='loss method')
    parser.add_argument('--loss-options', type=json.loads, default={}, metavar='', help='loss-specific parameters, i.e. \'{"wsigma": 1}\'')
    parser.add_argument('--evaluation-type', type=str, default=None, help='evaluation method')
    parser.add_argument('--evaluation-options', type=json.loads, default={}, metavar='', help='evaluation-specific parameters, i.e. \'{"topk": 1}\'')
    parser.add_argument('--precision', default=32, help="Precision for training. Options are 32 or 16")
    
    # Mamba specific arguments
    parser.add_argument('--expansion_factor', default=2, help="Mamba expansion factor")
    parser.add_argument('--dt_rank', default="auto", help="Mamba expansion factor")
    parser.add_argument('--kernel_size', default=4, help="Mamba 1D convolution kernel size")
    
    # ======================= Training Settings ================================
    parser.add_argument('--batch-size_test', type=int, default=None, help='batch size for testing')
    parser.add_argument('--batch-size_train', type=int, default=None, help='batch size for training')
    parser.add_argument('--nepochs', type=int, default=None, help='number of epochs to train')
    parser.add_argument('--niters', type=int, default=None, help='number of iterations at test time')
    parser.add_argument('--epoch-number', type=int, default=None, help='epoch number')
    parser.add_argument('--nthreads', type=int, default=None, help='number of threads for data loading')
    parser.add_argument('--manual-seed', type=int, default=None, help='manual seed for randomness')
    parser.add_argument('--check-val-every-n-epochs', type=int, default=1, help='validation every n epochs')
    parser.add_argument("--local_rank", default=0, type=int)

    # ======================= Hyperparameter Settings ===========================
    parser.add_argument('--learning-rate', type=float, default=None, help='learning rate')
    parser.add_argument('--optim-method', type=str, default=None, help='the optimization routine ')
    parser.add_argument('--optim-options', type=json.loads, default={}, metavar='', help='optimizer-specific parameters, i.e. \'{"lr": 0.001}\'')
    parser.add_argument('--scheduler-method', type=str, default=None, help='cosine, step, exponential, plateau')
    parser.add_argument('--scheduler-options', type=json.loads, default={}, metavar='', help='optimizer-specific parameters')
    parser.add_argument('--swa', type=bool, default=False, help='Stochastic Weight Averaging')

    if os.path.exists(args.config_file):
        config = configparser.ConfigParser()
        config.read([args.config_file])
        defaults = dict(config.items("Arguments"))
        parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_argv)
    
    # refine tuple arguments: this section converts tuples that are
    #                         passed as string back to actual tuples.
    pattern = re.compile('^\(.+\)')

    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        if isinstance(arg_value, str) and pattern.match(arg_value):
            setattr(args, arg_name, make_tuple(arg_value))
            print(arg_name, arg_value)
        elif isinstance(arg_value, dict):
            dict_changed = False
            for key, value in arg_value.items():
                if isinstance(value, str) and pattern.match(value):
                    dict_changed = True
                    arg_value[key] = make_tuple(value)
            if dict_changed:
                setattr(args, arg_name, arg_value)

    return args