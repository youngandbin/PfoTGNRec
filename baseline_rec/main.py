import yaml
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
SEED = 2022

import wandb
import argparse

from logging import getLogger
from recbole.config import Config
from recbole.utils.utils import init_seed
from recbole.utils.logger import init_logger
from recbole.data.utils import create_dataset, data_preparation
from recbole.utils.utils import get_model
from recbole.utils.utils import get_trainer

# from recbole.trainer.hyper_tuning import HyperTuning
from recbole.quick_start.quick_start import objective_function
from recbole.data.dataset import Dataset
from recbole.trainer.hyper_tuning import HyperTuning

# random seed
SEED = 2023
np.random.seed(SEED)
torch.manual_seed(SEED)

# set the gpu device as cuda:1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"""
arg parser
"""
parser = argparse.ArgumentParser(description='recbole baseline')
parser.add_argument('--model', type=str, default='BPR')
parser.add_argument('--period', type=str, default='period_1')
parser.add_argument('--dataset', type=str, default='transaction')
parser.add_argument('--config', type=str, default='general')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

"""
arg parser -> variables
"""
MODEL = args.model
DATASET = args.dataset
CONFIG = f'./config/fixed_config_{args.config}.yaml'

"""
main functions
"""

def objective_function(config_dict=None, config_file_list=None):
    
    config = Config(model=MODEL, dataset=DATASET, config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model_name = config['model']
    model = get_model(model_name)(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    """ (1) training """
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False)
    """ (2) testing """
    test_result = trainer.evaluate(test_data)

    return {
        'model': model_name,
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result, 
        'test_result': test_result
    }

def main_HPO():
    best_saved_path = f'/workspace/best_saved/{DATASET}/'
    if not os.path.exists(best_saved_path):
        os.makedirs(best_saved_path)
        
    with open(CONFIG, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['checkpoint_dir'] = f'/workspace/best_saved/{DATASET}/'
    with open(CONFIG, 'w') as f:
        yaml.dump(config, f)

    hp = HyperTuning(objective_function=objective_function, algo="exhaustive",
                     max_evals=50, params_file=f'hyper_result/{DATASET}/{MODEL}.hyper', fixed_config_file_list=[CONFIG]) # f'hyper_result/{DATASET}/{MODEL}.hyper'

    # run
    hp.run()
    torch.cuda.empty_cache()
    # export result to the file
    if not os.path.exists(f'./hyper_result/{DATASET}'):
        os.makedirs(f'./hyper_result/{DATASET}')
    hp.export_result(
        output_file=f'hyper_result/{DATASET}/{MODEL}.result')
    # print best parameters
    print('best params: ', hp.best_params)
    # save best parameters
    if not os.path.exists(f'./hyper_result/{DATASET}'):
        os.makedirs(f'./hyper_result/{DATASET}')
    with open(f'./hyper_result/{DATASET}/{MODEL}.best_params', 'w') as file:
        documents = yaml.dump(hp.best_params, file)
    
    # print best result
    best_result = hp.params2result[hp.params2str(hp.best_params)]

    best_result_df = pd.DataFrame.from_dict(
        best_result['test_result'], orient='index', columns=[f'{DATASET}'])
    if not os.path.exists(result_path+f'{DATASET}'):
        os.makedirs(result_path+f'{DATASET}')
    best_result_df.to_csv(
        result_path + f'{DATASET}/{MODEL}.csv', index=True)

def main():

    config = Config(model=MODEL, dataset=DATASET, config_file_list=[CONFIG])
    
    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # dataset creating and filtering # convert atomic files -> Dataset
    dataset = create_dataset(config)

    # dataset splitting # convert Dataset -> Dataloader
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    torch.cuda.empty_cache()
    
    """ (1) training """
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    """ (2) testing """
    trainer.eval_collector.data_collect(train_data)
    test_result = trainer.evaluate(test_data)
    # save result
    result_df = pd.DataFrame.from_dict(test_result, orient='index', columns=[f'{DATASET}'])
    result_df.to_csv(result_path + f'{MODEL}-{DATASET}.csv', index=True)


"""
main
"""
if __name__ == '__main__':
    
    # wandb
    # wandb.login()
    
    # wandb.init(project="stock_rec", name=f'{MODEL}_{DATASET}', entity="yejining_99")
    # wandb.config.update(args)
    
    # result path
    
    print('START--Model: ',MODEL,'Dataset: ', DATASET)
    result_path = './result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    if MODEL=='Pop':
        main()
    else:
        main_HPO() # hyper parameter optimization하려면 이거
        
    print('DONE--Model: ',MODEL,'Dataset: ', DATASET)
    
