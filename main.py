import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import shutil
import trainer
import trainer_warmup as trainer_warmup
from log_utils.utils import ReDirectSTD
from model.basenet import build_model
from utils.dataloader import build_data
from utils.utils import build_config, set_seed, write_logs
import warnings
import numpy as np  
from pathlib import Path
from datetime import datetime
from trainer import evaluate  
import torch
import gc

def parse_opt(known=False):
    parser = argparse.ArgumentParser(description="The proposed method")
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/train.yaml",
        help="config.yaml path",
    )
    return parser.parse_known_args()[0] if known else parser.parse_args()
def main(args):
    config = build_config(args.cfg)
    set_seed(config["seed"])
      
    acc_list_cnn, auc_list_cnn, sen_list_cnn, spe_list_cnn, f1s_list_cnn = [], [], [], [], []
    acc_list_vit, auc_list_vit, sen_list_vit, spe_list_vit, f1s_list_vit = [], [], [], [], []

    for fold in range(1,6):  # 5-fold cross-validation
        print(f"\n===== Starting Fold {fold}/5 =====")
        
        config_data = config["dataset"]
        dsets, dset_loaders = build_data(config_data, fold, debug=True)

        config_architecture = config["Architecture"]
        G1, G2, F1, F2 = build_model(config_architecture, DEVICE=config["DEVICE"])
       
        data_label_path = config_data["data_label"]
        data_subdir = Path(data_label_path).name.lower()

      
        task_name = f"{config_data['source']['name']}_to_{config_data['target']['name']}_{config['output_name']}"
        result_root = Path("results") / data_subdir / task_name
        if fold == 1:
            result_root.mkdir(parents=True, exist_ok=True)
            shutil.copy("trainer.py", result_root / "trainer.py")
       
        fold_dir = result_root / f"fold_{fold}"
        eval_dir = fold_dir / "eval_results"
        model_dir = fold_dir / "models"

        
        fold_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"==> Pre-evaluating on source domain for fold {fold}...")
        eval_result_source = evaluate(
            G1, G2, F1, F2,
            dset_loaders,
            device=config["DEVICE"],
            source_name=config["dataset"]["source"]["name"],
            target_name=config["dataset"]["target"]["name"],
            out_file=str(fold_dir / "log.txt"),
            save_dir=str(eval_dir),
            save_cnn_feature_name="features_cnn_source_only.npy",  
            return_pseduo=False
        )      
        data_label_path = config_data["data_label"]
        data_subdir = Path(data_label_path).name.lower()

        task_name = f"{config_data['source']['name']}_to_{config_data['target']['name']}_{config['output_name']}"
        result_root = Path("results")/ data_subdir / task_name
        if fold == 1:
            result_root.mkdir(parents=True, exist_ok=True)
            shutil.copy("trainer.py", result_root / "trainer.py")

        fold_dir = result_root / f"fold_{fold}"
        eval_dir = fold_dir / "eval_results"
        model_dir = fold_dir / "models"

        fold_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
    
        config["output_path"] = str(fold_dir)
        config["output_path_eval"] = str(eval_dir)
        config["output_path_model"] = str(model_dir)
        config["out_file"] = str(fold_dir / "log.txt")     
    
        out_file = config["out_file"]
        # write_logs(out_file, str(config))

        log_file_name = os.path.join("./logs", config_data["name"], config["output_path"].split("/")[-1] + f"_fold{fold}.txt")
        ReDirectSTD(log_file_name, "stdout", True, True)

        log_str = f"==> Starting the adaptation for Fold {fold}/5"
        write_logs(out_file, log_str, colors=True)
        G1, G2, F1, F2, eval_result, best_auc_cnn, best_acc_cnn, best_spe_cnn, best_sen_cnn, best_f1s_cnn, best_auc_vit, best_acc_vit, best_spe_vit, best_sen_vit, best_f1s_vit = trainer.train(
            config, G1, G2, F1, F2, dset_loaders)

        log_str = "Finished training and evaluation!"
        write_logs(out_file, log_str, colors=True)

        auc_list_cnn.append(eval_result.get("best_auc_cnn", 0.0)) #auc_cnn_target_test
        acc_list_cnn.append(eval_result.get("best_acc_cnn", 0.0))
        spe_list_cnn.append(eval_result.get("best_spe_cnn", 0.0))
        sen_list_cnn.append(eval_result.get("best_sen_cnn", 0.0))
        f1s_list_cnn.append(eval_result.get("best_f1s_cnn", 0.0))

        auc_list_vit.append(eval_result.get("best_auc_vit", 0.0)) #auc_vit_target_test
        acc_list_vit.append(eval_result.get("best_acc_vit", 0.0))
        spe_list_vit.append(eval_result.get("best_spe_vit", 0.0))
        sen_list_vit.append(eval_result.get("best_sen_vit", 0.0))
        f1s_list_vit.append(eval_result.get("best_f1s_vit", 0.0))
    
        gc.collect()

    avg_auc_cnn, std_auc_cnn = np.mean(auc_list_cnn), np.std(auc_list_cnn)
    avg_acc_cnn, std_acc_cnn = np.mean(acc_list_cnn), np.std(acc_list_cnn)
    avg_spe_cnn, std_spe_cnn = np.mean(spe_list_cnn), np.std(spe_list_cnn)
    avg_sen_cnn, std_sen_cnn = np.mean(sen_list_cnn), np.std(sen_list_cnn)
    avg_f1s_cnn, std_f1s_cnn = np.mean(f1s_list_cnn), np.std(f1s_list_cnn)

    avg_auc_vit, std_auc_vit = np.mean(auc_list_vit), np.std(auc_list_vit)
    avg_acc_vit, std_acc_vit = np.mean(acc_list_vit), np.std(acc_list_vit)
    avg_spe_vit, std_spe_vit = np.mean(spe_list_vit), np.std(spe_list_vit)
    avg_sen_vit, std_sen_vit = np.mean(sen_list_vit), np.std(sen_list_vit)
    avg_f1s_vit, std_f1s_vit = np.mean(f1s_list_vit), np.std(f1s_list_vit)

    final_log_str_cnn = (
        "\n==== 5-Fold G2_Cross-Validation Results ====\n"  
       
        f"Average AUC: {avg_auc_cnn:.4f} ± {std_auc_cnn:.4f}\n"
        f"Average ACC: {avg_acc_cnn:.4f}% ± {std_acc_cnn:.4f}\n"       
        f"Average SPE: {avg_spe_cnn:.4f} ± {std_spe_cnn:.4f}\n"
        f"Average SEN: {avg_sen_cnn:.4f} ± {std_sen_cnn:.4f}\n"
        f"Average F1-Score: {avg_f1s_cnn:.4f} ± {std_f1s_cnn:.4f}\n"
    )
    final_log_str_vit = (
        "\n==== 5-Fold G1_Cross-Validation Results ====\n"       
        f"Average AUC: {avg_auc_vit:.4f} ± {std_auc_vit:.4f}\n"
        f"Average ACC: {avg_acc_vit:.4f}% ± {std_acc_vit:.4f}\n"        
        f"Average SPE: {avg_spe_vit:.4f} ± {std_spe_vit:.4f}\n"
        f"Average SEN: {avg_sen_vit:.4f} ± {std_sen_vit:.4f}\n"
        f"Average F1-Score: {avg_f1s_vit:.4f} ± {std_f1s_vit:.4f}\n"
    )
    print(final_log_str_cnn)
    write_logs(out_file, final_log_str_cnn, colors=True)
    print(final_log_str_vit)
    write_logs(out_file, final_log_str_vit, colors=True)
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_opt()
    main(args)


