import os
import sys
sys.path.insert(0, os.getcwd())
from utils import load_yaml


def get_full_data_name(conf):
    conf_full_data = conf["full_data"]
    return f'{conf_full_data["data_type"]}-{conf_full_data["data_name"]}'

def get_target_data_name(conf):
    conf_target_data = conf["target_data"]
    return f'{conf_target_data["data_type"]}-{conf_target_data["data_name"]}'

def get_proxy_data_name(conf):
    conf_proxy_data = conf["proxy_data"]
    return f'{conf_proxy_data["data_type"]}-{conf_proxy_data["data_name"]}'

def get_token_model_name(conf):
    conf_token_model = conf["full_data"]
    return f'{conf_token_model["model_type"]}-{conf_token_model["model_name"]}'

def get_refer_model_name(conf):
    conf_refer_model = conf["scorer_data_infer"]
    return f'{conf_refer_model["model_type"]}'

###
def s1_1_full_target_token_model(conf):
    return f'token-{get_token_model_name(conf)}'

def s1_2_full_token_data(conf):
    return f'{get_full_data_name(conf)}_token-{get_token_model_name(conf)}'

def s2_1_target_data(conf, file_name=False):
    if file_name:
        return os.path.join(str(get_target_data_name(conf)), str(get_target_data_name(conf))+'.jsonl')
    else:
        return f'{get_target_data_name(conf)}'

def s2_2_target_token_data(conf):
    return f'{get_target_data_name(conf)}_token-{get_token_model_name(conf)}'

def s3_proxy_token_data(conf):
    return f'{get_proxy_data_name(conf)}_token-{get_token_model_name(conf)}'

def s4_1_annotated_proxy_token_data(conf):
    return f'{get_proxy_data_name(conf)}_{get_target_data_name(conf)}_annotate-{get_token_model_name(conf)}'
    
def s4_2_prepare_scorer_model(conf):
    return f'scorer-{get_refer_model_name(conf)}'

def s4_3_prepare_scorer_data(conf):
    return f'scorer-trainset_{get_proxy_data_name(conf)}'

def s5_scorer_model(conf):
    return f'scorer-{get_refer_model_name(conf)}_{get_proxy_data_name(conf)}_{get_target_data_name(conf)}'

def s6_1_convert_scorer_token_data(conf):
    return f'{get_proxy_data_name(conf)}_token-{get_refer_model_name(conf)}'

def s6_2_infer_full_data(conf):
    return f'scorer-infer_{get_full_data_name(conf)}'
