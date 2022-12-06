"""This script runs main.py multiple times changing the seed values.
"""
import argparse
import configparser
import os

from main import main
from evaluate import evaluate

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-cm', '--config_main_file', default='config_main.ini')
    parser.add_argument('-ce', '--config_eval_file', default='config_eval.ini')
    parser.add_argument('-n', '--num_experiments', default=5)

    args = parser.parse_args()

    if os.path.isfile(args.config_main_file):
        config = configparser.ConfigParser()
        config.read(args.config_main_file)

        num_exp = int(args.num_experiments)

        if os.path.isfile(args.config_eval_file):
            config_eval = configparser.ConfigParser()
            config_eval.read(args.config_eval_file)
            run_eval = True
        else:
            run_eval = False

        for idx in range(num_exp):
            print(f'**********************************************************************************')
            print(f'*********************Starting experiment {idx+1} of {num_exp}*********************')
            print(f'**********************************************************************************')

            # configure experiment folder:
            # (config elements are all strings)
            prev_dir_out = str(config['io']['dir_out'])
            config['io']['dir_out'] = config['io']['dir_out']+config['datamodule']['seed']

            # run experiment:
            model = main(config)

            # pytorch lightning "does not consider" script continues after .fit, so we need to check model's rank
            # https://stackoverflow.com/questions/66261729/pytorch-lightning-duplicates-main-script-in-ddp-mode
            if model.global_rank == 0:

                if run_eval:
                    config_eval['io']['dir_out'] = config['io']['dir_out']
                    config_eval['io']['model_path'] = os.path.join(config_eval['io']['dir_out'], 'model.pt')
                    evaluate(config_eval)

                # update seed for next experiment:
                config['datamodule']['seed'] = str(int(config['datamodule']['seed']) + 1)

                # restore dir_out
                config['io']['dir_out'] = prev_dir_out
    
    else:
        print('Please provide a valid configuration file.')

#python multi_experiments.py -cm configs/SA-all.ini -ce configs/eval_SA-all.ini & python multi_experiments.py -cm configs/SA-freeze.ini -ce configs/eval_SA-freeze.ini & python multi_experiments.py -cm configs/SA-melt.ini -ce configs/eval_SA-melt.ini & python multi_experiments.py -cm configs/poly_type-all.ini -ce configs/eval_poly_type-all.ini & python multi_experiments.py -cm configs/poly_type-freeze.ini -ce configs/eval_poly_type-freeze.ini & python multi_experiments.py -cm configs/poly_type-melt.ini -ce configs/eval_poly_type-melt.ini
#python multi_experiments.py -cm configs/SD-all.ini -ce configs/eval_SD-all.ini & python multi_experiments.py -cm configs/SD-freeze.ini -ce configs/eval_SD-freeze.ini & python multi_experiments.py -cm configs/SD-melt.ini -ce configs/eval_SD-melt.ini