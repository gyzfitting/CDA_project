import os
import random
import numpy as np
import torch
import utils.preprocess as preprocess
import yaml

def set_seed(seed=3):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    print("Loaded config from {}".format(file_path))
    print("config: {}".format(config))
    return config

def build_config(config):
    config = load_config(config)
    config["DEVICE"] = torch.device(config["DEVICE"])

    config_data = config["dataset"]
    config_data["name"] = config_data["data_label"].split("/")[-1].lower()
    #  "******" is the database name
    if "******" in config_data["name"] and config_data["method"] == "UDA":
        config["Architecture"]["class_num"] = 2
        config_data['data_prefix'] = {'train': '_train.txt', 'test': '_test.txt'}
    elif "******" in config_data["name"] and config_data["method"] == "UDA":
        config["Architecture"]["class_num"] = 2
        config_data['data_prefix'] = {'train': '_train.txt', 'test': '_test.txt'}
    elif "******" in config_data["name"] and config_data["method"] == "UDA":
        config["Architecture"]["class_num"] = 2
        config_data['data_prefix'] = {'train': '_train.txt', 'test': '_test.txt'}
    else:
        raise ValueError(
            "dataset cannot be recognized. Please define your own dataset here."
        )   
    source_name = config_data["source"]["name"]
    target_name = config_data["target"]["name"]
    target_shot = config_data["target_shot"]
    method = (
        config_data["method"]
        if config_data["method"] == "UDA"
        else f"{config_data['method']}_{target_shot}"
    )
    
    ### Config info for Adaptation stage ###
    config["output_path"] = os.path.join(
        config["output_path"],
        config_data["name"],
        f"{source_name}_to_{target_name}_{method}_{config['output_name']}",       
    )
  
    if not os.path.exists(config["output_path"]):
        os.system("mkdir -p " + config["output_path"])  
    config["out_file"] = open(os.path.join(config["output_path"], "log.txt"), "w")  
    ### Get config model ###
    config_architecture = config["Architecture"]
    backbone_setting = config_architecture["Backbone"]  
    classifier_setting = config_architecture["Classifier"]   

    ### Set Pretrained model path ###   
    if config["warmup"]:
        backbone_setting["pretrained_1"] = ''
        backbone_setting["pretrained_2"] = ''
        classifier_setting["pretrained_F1"] = ''
        classifier_setting["pretrained_F2"] = ''

    elif backbone_setting["pretrained_1"] and classifier_setting["pretrained_F1"]:
        pass

    elif config["pretrained_models"]:
        pretrained_name = f"{source_name}_to_{target_name}_{method}_pretrained_warmup"
        pretrained_path = os.path.join(
            config["pretrained_models"], config_data["name"], pretrained_name
        )
        backbone_setting["pretrained_1"] = (
            f"{pretrained_path}/the_best_G1_pretrained.pth.tar"
        )
        backbone_setting["pretrained_2"] = (
            f"{pretrained_path}/the_best_G2_pretrained.pth.tar"
        )
        classifier_setting["pretrained_F1"] = (
            f"{pretrained_path}/the_best_F1_pretrained.pth.tar"
        )
        classifier_setting["pretrained_F2"] = (
            f"{pretrained_path}/the_best_F2_pretrained.pth.tar"
        ) 
    return config

def write_logs(out_file, log_str, colors=False):
    with open(out_file, "a") as f:
        f.write(log_str + "\n")
    if colors:
        print("\033[92m" + log_str + "\033[0m")  
    else:
        print(log_str)



