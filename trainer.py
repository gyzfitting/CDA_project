import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils.loss import jsd_divergence
from utils.optimizer import inv_lr_scheduler
from utils.utils import write_logs
import torchio as tio
import copy
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd

def evaluate(G1, G2, F1, F2, dset_loaders, **kwargs):
    ### Get some variables in kwargs ###
    method = kwargs.get("method", "UDA")
    source_name = kwargs.get("source_name", "ncode")
    target_name = kwargs.get("target_name", "nbold")
    out_file = kwargs.get("out_file", None)
   
    log_str = f"  -- Domain task [{source_name} --> {target_name}] \n"
   
    target_test_result = eval_domain(
        G1, G2, F1, F2, dset_loaders["target_test"], **kwargs
    )
  
    kwargs["return_pseduo"] = False
    
    log_str += "\t-- CNN's Accuracy Target Test = {:<05.4f}%  ViT's Accuracy Target Test = {:.4f}% \n".format(
        target_test_result["cnn_accuracy"], target_test_result["vit_accuracy"]
    )
   
    log_str += "\t-- AUC CNN Target Test = {:<05.4f}  AUC ViT Target Test = {:.4f} \n".format(
        target_test_result["auc_cnn"], target_test_result["auc_vit"]
    )
    log_str += "\t-- SEN CNN Target Test = {:<05.4f}  SEN ViT Target Test = {:.4f} \n".format(
        target_test_result["sen_cnn"], target_test_result["sen_vit"]
    )
    log_str += "\t-- SPE CNN Target Test = {:<05.4f}  SPE ViT Target Test = {:.4f} \n".format(
        target_test_result["spe_cnn"], target_test_result["spe_vit"]
    )
    log_str += "\t-- F1s CNN Target Test = {:<05.4f}  F1s ViT Target Test = {:.4f} \n".format(
        target_test_result["f1s_cnn"], target_test_result["f1s_vit"]
    )
  
    write_logs(out_file, log_str)
        
    G1.train()
    G2.train()
    F1.train()
    F2.train()
    return {
        "cnn_acc_target_test": target_test_result.get("cnn_accuracy", 0.0),
        "vit_acc_target_test": target_test_result.get("vit_accuracy", 0.0),
        "auc_cnn_target_test": target_test_result.get("auc_cnn", 0.0),
        "auc_vit_target_test": target_test_result.get("auc_vit", 0.0),
        "sen_cnn_target_test": target_test_result.get("sen_cnn", 0.0),
        "sen_vit_target_test": target_test_result.get("sen_vit", 0.0),
        "spe_cnn_target_test": target_test_result.get("spe_cnn", 0.0),
        "spe_vit_target_test": target_test_result.get("spe_vit", 0.0),
        "f1s_cnn_target_test": target_test_result.get("f1s_cnn", 0.0),
        "f1s_vit_target_test": target_test_result.get("f1s_vit", 0.0),
        "best_acc_cnn": target_test_result.get("best_acc_cnn", 0.0),
        "best_spe_cnn": target_test_result.get("best_spe_cnn", 0.0),
        "best_sen_cnn": target_test_result.get("best_sen_cnn", 0.0),
        "best_f1s_cnn": target_test_result.get("best_f1s_cnn", 0.0),
        "best_acc_vit": target_test_result.get("best_acc_vit", 0.0),
        "best_spe_vit": target_test_result.get("best_spe_vit", 0.0),
        "best_sen_vit": target_test_result.get("best_sen_vit", 0.0),
        "best_f1s_vit": target_test_result.get("best_f1s_vit", 0.0),
        "pl_acc_cnn": target_test_result.get("pl_acc_cnn", 0.0),
        "correct_pl_cnn": target_test_result.get("correct_pl_cnn", 0.0),
        "total_pl_cnn": target_test_result.get("total_pl_cnn", 0.0),
        "pl_acc_vit": target_test_result.get("pl_acc_vit", 0.0),
        "correct_pl_vit": target_test_result.get("correct_pl_vit", 0.0),
        "total_pl_vit": target_test_result.get("total_pl_vit", 0.0),
    }

def eval_domain(G1, G2, F1, F2, test_loader, **kwargs):  
    device = kwargs.get("device", "cpu")
    return_pseduo = kwargs.get("return_pseduo", False)
    thresh_vit = kwargs.get("thresh_vit", "0.5")
    thresh_cnn = kwargs.get("thresh_cnn", "0.5")
    save_dir = kwargs.get("save_dir", "./eval_results")  
  
    G1.eval()
    G2.eval()
    F1.eval()
    F2.eval()
    os.makedirs(save_dir, exist_ok=True)
    logits_cnn_all, logits_vit_all, labels_all, confidences_cnn_all, confidences_vit_all= [], [], [], [], []
    pl_acc_cnn, correct_pl_cnn, total_pl_cnn, pl_acc_vit, correct_pl_vit, total_pl_vit= 0, 0, 0, 0, 0, 0
    probs_cnn_all, probs_vit_all = [], [] 
    features_cnn_all = [] 
    features_vit_all = []     
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs = data['mri'][tio.DATA].to(device).float()          
            feat_cnn = G2(inputs)
            feat_vit = G1(inputs)
            features_cnn_all.append(feat_cnn.cpu())
            features_vit_all.append(feat_vit.cpu())         
            vit_logits = F1(feat_vit)
            cnn_logits = F2(feat_cnn)

            logits_cnn_all.append(cnn_logits.cpu())
            logits_vit_all.append(vit_logits.cpu())
            labels_all.append(data["labels"])

            confidences_cnn_all.append(nn.Softmax(dim=1)(cnn_logits).max(1)[0].cpu())
            confidences_vit_all.append(nn.Softmax(dim=1)(vit_logits).max(1)[0].cpu())

    labels = torch.cat(labels_all, dim=0)
    labels_np = labels.numpy()
    print("===True labels", labels)  
    logits_cnn = torch.cat(logits_cnn_all, dim=0)
    logits_vit = torch.cat(logits_vit_all, dim=0)
  
    probs_cnn = nn.Softmax(dim=1)(logits_cnn).numpy()
    pos_probs_cnn = probs_cnn[:, 1]
    fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(labels_np, pos_probs_cnn)
    auc_cnn = auc(fpr_cnn, tpr_cnn)
    youden_cnn = tpr_cnn - fpr_cnn
    best_idx_cnn = np.argmax(youden_cnn)
    best_thresh_cnn = thresholds_cnn[best_idx_cnn]
    predict_cnn = (pos_probs_cnn >= best_thresh_cnn).astype(int)
    tp_cnn = np.sum((predict_cnn == 1) & (labels_np == 1))
    tn_cnn = np.sum((predict_cnn == 0) & (labels_np == 0))
    fp_cnn = np.sum((predict_cnn == 1) & (labels_np == 0))
    fn_cnn = np.sum((predict_cnn == 0) & (labels_np == 1))
    sen_cnn = tp_cnn / (tp_cnn + fn_cnn + 1e-10)
    spe_cnn = tn_cnn / (tn_cnn + fp_cnn + 1e-10)
    precision_cnn = tp_cnn / (tp_cnn + fp_cnn + 1e-10)
    f1s_cnn = 2 * (precision_cnn * sen_cnn) / (precision_cnn + sen_cnn + 1e-10)
    cnn_accuracy = (tp_cnn + tn_cnn) / len(labels_np)

    probs_vit = nn.Softmax(dim=1)(logits_vit).numpy()
    pos_probs_vit = probs_vit[:, 1]
    fpr_vit, tpr_vit, thresholds_vit = roc_curve(labels_np, pos_probs_vit)
    auc_vit = auc(fpr_vit, tpr_vit)
    youden_vit = tpr_vit - fpr_vit
    best_idx_vit = np.argmax(youden_vit)
    best_thresh_vit = thresholds_vit[best_idx_vit]
    predict_vit = (pos_probs_vit >= best_thresh_vit).astype(int)
    tp_vit = np.sum((predict_vit == 1) & (labels_np == 1))
    tn_vit = np.sum((predict_vit == 0) & (labels_np == 0))
    fp_vit = np.sum((predict_vit == 1) & (labels_np == 0))
    fn_vit = np.sum((predict_vit == 0) & (labels_np == 1))
    sen_vit = tp_vit / (tp_vit + fn_vit + 1e-10)
    spe_vit = tn_vit / (tn_vit + fp_vit + 1e-10)
    precision_vit = tp_vit / (tp_vit + fp_vit + 1e-10)
    f1s_vit = 2 * (precision_vit * sen_vit) / (precision_vit + sen_vit + 1e-10)
    vit_accuracy = (tp_vit + tn_vit) / len(labels_np)

    print("===predict_cnn", predict_cnn)
    print(len(labels))
    print("===predict_vit", predict_vit)
    print(len(labels))

    features_cnn = torch.cat(features_cnn_all, dim=0).numpy()
    features_vit = torch.cat(features_vit_all, dim=0).numpy()
    # np.save(os.path.join(save_dir, "features_cnn.npy"), features_cnn)
    save_cnn_feature_name = kwargs.get("save_cnn_feature_name", "features_cnn.npy")  
    np.save(os.path.join(save_dir, save_cnn_feature_name), features_cnn) 
    np.save(os.path.join(save_dir, "features_vit.npy"), features_vit)

    if return_pseduo:
        def get_pseduo_label_info(confidences_all, predict, labels, thresh):
            confidences = torch.cat(confidences_all, dim=0)         
            if isinstance(thresh, str):
                try:
                    thresh = float(thresh)
                except ValueError:
                    raise TypeError(f"Invalid threshold value: {thresh}. It must be a number.")          
            masks_bool = confidences > thresh
            total_pl = masks_bool.sum().item()  
            if total_pl > 0:
                masked_predict = predict[masks_bool]
                masked_labels = labels[masks_bool]              
                correct_pl = (masked_predict == masked_labels).clone().detach().sum().item()               
                pl_acc = correct_pl / total_pl
            else:
                correct_pl = -1.0
                pl_acc = -1.0
            return correct_pl, pl_acc, total_pl
        correct_pl_cnn, pl_acc_cnn, total_pl_cnn = get_pseduo_label_info(
            confidences_cnn_all, predict_cnn, labels, thresh_cnn
        )       
        correct_pl_vit, pl_acc_vit, total_pl_vit = get_pseduo_label_info(
            confidences_vit_all, predict_vit, labels, thresh_vit
        )
    # 1. outputs.npy (包含兩個分支的輸出)
    np.save(os.path.join(save_dir, "outputs.npy"), {
        "cnn_probs": probs_cnn,
        "vit_probs": probs_vit,
        "cnn_logits": logits_cnn.numpy(),
        "vit_logits": logits_vit.numpy()
    })
    # 2. targets.npy
    np.save(os.path.join(save_dir, "targets.npy"), labels.numpy())
    # 3. predictions.csv
    pd.DataFrame({
        "True_Label": labels_np,
        "CNN_Prob_0": probs_cnn[:, 0],
        "CNN_Prob_1": probs_cnn[:, 1],
        "CNN_Pred": predict_cnn,
        "ViT_Prob_0": probs_vit[:, 0],
        "ViT_Prob_1": probs_vit[:, 1],
        "ViT_Pred": predict_vit,
        "CNN_Confidence": torch.cat(confidences_cnn_all).numpy(),
        "ViT_Confidence": torch.cat(confidences_vit_all).numpy()
    }).to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
    # 4. metrics.npy
    np.save(os.path.join(save_dir, "metrics.npy"), {
        "CNN": {
            "Accuracy": cnn_accuracy,
            "AUC": auc_cnn,
            "Sensitivity": sen_cnn,
            "Specificity": spe_cnn,
            "F1": f1s_cnn
        },
        "ViT": {"Accuracy": vit_accuracy,
            "AUC": auc_vit,
            "Sensitivity": sen_vit,
            "Specificity": spe_vit,
            "F1": f1s_vit}  

    })
    return {
        "cnn_accuracy": cnn_accuracy * 100,
        "vit_accuracy": vit_accuracy * 100,
        "auc_cnn": auc_cnn,
        "sen_cnn": sen_cnn,
        "spe_cnn": spe_cnn,#
        "f1s_cnn": f1s_cnn,
        "auc_vit": auc_vit,
        "sen_vit": sen_vit,
        "spe_vit": spe_vit,
        "f1s_vit": f1s_vit,       
        "best_acc_cnn": cnn_accuracy,
        "best_spe_cnn": spe_cnn,
        "best_sen_cnn": sen_cnn,
        "best_f1s_cnn": f1s_cnn,
        "best_acc_vit": vit_accuracy,
        "best_spe_vit": spe_vit,
        "best_sen_vit": sen_vit,
        "best_f1s_vit": f1s_vit,
        "pl_acc_cnn": pl_acc_cnn,
        "correct_pl_cnn": correct_pl_cnn,
        "total_pl_cnn": total_pl_cnn,
        "pl_acc_vit": pl_acc_vit,
        "correct_pl_vit": correct_pl_vit,
        "total_pl_vit": total_pl_vit,
        }
def train(config, G1, G2, F1, F2, dset_loaders):#dset_loaders：
    method = config["dataset"]["method"]
    out_file = config["out_file"]
    config_op = config["optimizer"]
    log_str = f"Method: {method} - File: trainer.py"
    write_logs(out_file, log_str, colors=True)
    DEVICE = config["DEVICE"]
    source_name = config["dataset"]["source"]["name"]
    target_name = config["dataset"]["target"]["name"]
    eval_result = {}   
    def get_param(base_network, multi=0.1, weight_decay=0.0005): #multi=0.1
        param = []
        for key, value in dict(base_network.named_parameters()).items():
            if value.requires_grad:
                if "classifier" not in key:
                    param += [{"params": [value], "lr": multi, "weight_decay": weight_decay}]
                else:
                    param += [{"params": [value],"lr": multi * 10, #10,"weight_decay": weight_decay,}]
        return param
    params1 = get_param(G1, multi=0.1, weight_decay=0.0005) #multi=0.1
    params2 = get_param(G2, multi=0.1, weight_decay=0.0005)
  
    G1.train(), G2.train(),F1.train(), F2.train()  
    optimizer_g1 = optim.SGD(params1,momentum=config_op["momentum"],weight_decay=config_op["weight_decay"],nesterov=config_op["nesterov"], )
    optimizer_g2 = optim.SGD(params2,momentum=config_op["momentum"],weight_decay=config_op["weight_decay"],nesterov=config_op["nesterov"], )
    optimizer_f1 = optim.SGD(list(F1.parameters()),lr=config_op["lr"],momentum=config_op["momentum"],weight_decay=config_op["weight_decay"],nesterov=config_op["nesterov"], )
    optimizer_f2 = optim.SGD(list(F2.parameters()),lr=config_op["lr"],momentum=config_op["momentum"],weight_decay=config_op["weight_decay"],nesterov=config_op["nesterov"], )

    def zero_grad_all():
        optimizer_g1.zero_grad()
        optimizer_g2.zero_grad()
        optimizer_f1.zero_grad()
        optimizer_f2.zero_grad()  
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE) 
    len_source_labeled = len(dset_loaders["source_train"])
    len_target_unlabeled = len(dset_loaders["target_train"]) 
    the_best_acc_vit_test = 0.0   
    The_best_AUC_CNN_Target_Test = 0
    best_acc_cnn = 0
    best_spe_cnn = 0
    best_sen_cnn = 0
    best_f1s_cnn = 0
    The_best_AUC_ViT_Target_Test = 0
    best_acc_vit = 0
    best_spe_vit = 0
    best_sen_vit = 0
    best_f1s_vit = 0  
    param_lr_g1 = []
    for param_group in optimizer_g1.param_groups:
        param_lr_g1.append(param_group["lr"])
    param_lr_g2 = []
    for param_group in optimizer_g2.param_groups:
        param_lr_g2.append(param_group["lr"])
    param_lr_f1 = []
    for param_group in optimizer_f1.param_groups:
        param_lr_f1.append(param_group["lr"])
    param_lr_f2 = []
    for param_group in optimizer_f2.param_groups:
        param_lr_f2.append(param_group["lr"])

    source_only_path = os.path.join(config["output_path_eval"], "features_cnn_source_only.npy")
    if config.get("save_source_only_features", True) and not os.path.exists(source_only_path):
        eval_kwargs_source_only = {
            "method": method,
            "device": DEVICE,
            "source_name": source_name,
            "target_name": target_name,
            "out_file": config["out_file"],
            "return_pseduo": False,
            "thresh_cnn": config["thresh_CNN"],
            "thresh_ViT": config["thresh_ViT"],
            "save_dir": config["output_path_eval"],
            "save_cnn_feature_name": "features_cnn_source_only.npy",
        }
        print("🔹 Saving CNN source-only features before domain adaptation...")
        evaluate(G1, G2, F1, F2, dset_loaders, **eval_kwargs_source_only)
  
    for step in range(config["adapt_iters"]):

        optimizer_g1 = inv_lr_scheduler(
            param_lr_g1, optimizer_g1, step, init_lr=config_op["lr_vit"],
        )
        optimizer_g2 = inv_lr_scheduler(
            param_lr_g2, optimizer_g2, step, init_lr=config_op["lr_cnn"],
        )
        optimizer_f1 = inv_lr_scheduler(param_lr_f1, optimizer_f1, step, init_lr=config_op["lr_cnn"])
        optimizer_f2 = inv_lr_scheduler(param_lr_f2, optimizer_f2, step, init_lr=config_op["lr_cnn"])
        lr_g1 = optimizer_g1.param_groups[0]["lr"]
        lr_g2 = optimizer_g2.param_groups[0]["lr"]
       
        if step % len_source_labeled == 0:
            iter_source = iter(dset_loaders["source_train"])
        if step % len_target_unlabeled == 0:
            iter_unlabeled_target = iter(dset_loaders["target_train"])
        batch_source = next(iter_source)
        batch_target_unlabeled = next(iter_unlabeled_target)
      
        source_w, source_labeled = (
            batch_source["mri"][tio.DATA].to(DEVICE),
            batch_source["labels"].to(DEVICE),) 
        inputs_target_w, inputs_target_str = (
            batch_target_unlabeled["mri"][tio.DATA].to(DEVICE),
            batch_target_unlabeled["mri2"][tio.DATA].to(DEVICE),)  
    
        labeled_targetw_tuple = [source_w]
        labeled_gt = [source_labeled]
        nl = source_w.size(0)
       
        labeled_targetw_input = torch.cat(labeled_targetw_tuple + [inputs_target_w], 0).float()
        labeled_targetstr_input = torch.cat(labeled_targetw_tuple + [inputs_target_str], 0).float()
        unlabeled_target_input = torch.cat((inputs_target_w, inputs_target_str), 0).float()
        labeled_input = torch.cat(labeled_targetw_tuple, 0).float()
        labeled_gt = torch.cat(labeled_gt, 0)
        zero_grad_all()
        ####Supervised Learning#####
        vit_logits = F1(G1(labeled_input))
        vit_loss = ce_criterion(vit_logits, labeled_gt)
        vit_loss.backward()
        optimizer_g1.step()
        optimizer_f1.step()
        zero_grad_all()

        cnn_logits = F2(G2(labeled_targetw_input))
        cnn_loss = ce_criterion(cnn_logits[:nl], labeled_gt)
        cnn_loss.backward()
        optimizer_g2.step()
        optimizer_f2.step()
        zero_grad_all()
        torch.cuda.empty_cache()
        if step < 3000:
            continue
        if step > 3450:
            torch.nn.utils.clip_grad_norm_(G1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(G2.parameters(), max_norm=1.0)
        F1_s = copy.deepcopy(F1)
        F2_s = copy.deepcopy(F2)
        ###Minimax Classifier Discrepancy####
        output_f1 = F1(G1(labeled_input))
        output_f2 = F2(G2(labeled_targetw_input))
        loss_f1 = ce_criterion(output_f1, labeled_gt)
        loss_f2 = ce_criterion(output_f2[:nl], labeled_gt)
        loss_s = loss_f1 + loss_f2
        vit_features_t = G1(unlabeled_target_input)
        
        output_t1 = F1(vit_features_t)
        output_t2 = F2(vit_features_t)
        loss_dis = torch.mean(torch.abs(F.softmax(output_t1, dim=1) - F.softmax(output_t2, dim=1)))
        loss = loss_s - loss_dis
        loss.backward()
        optimizer_f1.step()
        optimizer_f2.step()
        zero_grad_all()
        ##### Conquering #####
        for j in range(6):
            feat_t = G2(unlabeled_target_input)
            # feat_t = G1(unlabeled_target_input)
            out_t1 = F1(feat_t)
            out_t2 = F2(feat_t)
            loss_dis = torch.mean(torch.abs(F.softmax(out_t1, dim=1) - F.softmax(out_t2, dim=1)))
            loss_dis.backward()
            torch.nn.utils.clip_grad_norm_(G2.parameters(), max_norm=1.0)  # 仅G
            optimizer_g2.step()
            zero_grad_all()

        ##Collaborative training with JSD Thresholding** #####
        ####### 1. ViT Branch with JSD Thresholding ######
        jsd_threshold = max(0.05, 0.68 * (1.0 - step / config["adapt_iters"]))
        with torch.no_grad():
            vit_logits = F1(G1(inputs_target_w))
            vit_logits_1 = F1_s(G1(inputs_target_w))
        jsd_value = jsd_divergence(vit_logits, vit_logits_1)
        if (jsd_value < jsd_threshold).any():
            averaged_vit_logits = F.softmax((vit_logits + vit_logits_1) / 2.0, dim=1)
            cnn_logits = F.softmax(F2(G2(labeled_targetstr_input)), dim=1)           
            confidence_threshold = 0.5 + 0.2 * (1.0 - step / config["adapt_iters"])
            confidence_mask = torch.max(averaged_vit_logits, dim=1).values > confidence_threshold

            filtered_indices = confidence_mask.nonzero(as_tuple=True)[0]
            filtered_logits = averaged_vit_logits[filtered_indices]
            filtered_cnn_logits = cnn_logits[nl:][filtered_indices]
            if filtered_cnn_logits.size(0) < 10:
                continue           
            loss_vit_to_cnn = F.kl_div(
                filtered_cnn_logits.log(), filtered_logits.detach(), reduction="batchmean"
            )
            loss_vit_to_cnn.backward()
            optimizer_g2.step()
            optimizer_f2.step()
        zero_grad_all()
        ####### 2. CNN Branch with JSD Thresholding ######
        with torch.no_grad():
            cnn_logits = F2(G2(labeled_targetw_input))
            cnn_logits_1 = F2_s(G2(labeled_targetw_input))
        jsd_value = jsd_divergence(cnn_logits, cnn_logits_1)
        if (jsd_value < jsd_threshold).any():
            averaged_cnn_logits = F.softmax((cnn_logits + cnn_logits_1) / 2.0, dim=1)
            vit_logits = F.softmax(F1(G1(inputs_target_str)), dim=1)
            if vit_logits.size(0) < 8:
                continue
            loss_cnn_to_vit = F.kl_div(
                vit_logits.log(), averaged_cnn_logits[nl:].detach(), reduction="batchmean"
            )
            loss_cnn_to_vit.backward()
            optimizer_g1.step()
            optimizer_f1.step()
        zero_grad_all()

        if step % 20 == 0 or step == config["adapt_iters"] - 1:
            log_str = (
                "Iters: ({}/{}) \t  lr_g1 = {:<10.6f} lr_g2 = {:<10.6f} "   
                "cnn_loss = {:<10.6f}  vit_loss = {:<10.6f}" .format(
                # "loss_s = {:<10.6f}  loss = {:<10.6f}".format(
                    step,
                    config["adapt_iters"],
                    lr_g1,
                    lr_g2,                   
                    cnn_loss.item(),
                    vit_loss.item(),
                    # loss_s.item(),
                    # loss.item(),

                )
            )
            write_logs(out_file, log_str)
    
        if step % config["test_interval"] == config["test_interval"] - 1:           
            eval_kwargs = {                
                "method": method,
                "device": DEVICE,
                "source_name": source_name,
                "target_name": target_name,
                "out_file": config["out_file"],               
                "return_pseduo": True,
                "thresh_cnn": config["thresh_CNN"],
                "thresh_ViT": config["thresh_ViT"],
                "save_dir": config["output_path_eval"],
            }         
            eval_kwargs_acc = {
                **eval_kwargs,
                "save_cnn_feature_name": "features_cnn_acc_best.npy"
            }
            eval_kwargs_auc = {
                **eval_kwargs,
                "save_cnn_feature_name": "features_cnn_auc_best.npy"
            }
            eval_result = evaluate(G1, G2, F1, F2, dset_loaders, **eval_kwargs)          
            cnn_acc_test = eval_result.get("cnn_acc_target_test", 0.0)
            vit_acc_test = eval_result.get("vit_acc_target_test", 0.0)

            auc_cnn_target_test = eval_result.get("auc_cnn_target_test", 0.0)
            auc_vit_target_test = eval_result.get("auc_vit_target_test", 0.0)

            if cnn_acc_test > the_best_acc_cnn_test:
                the_best_acc_cnn_test = cnn_acc_test
                # Save model
                if config["save_models"]:
                    torch.save(
                        G2.state_dict(),
                        os.path.join(config["output_path_model"], "the_best_G2.pth.tar"),
                    )
                    torch.save(
                        F2.state_dict(),
                        os.path.join(config["output_path_model"], "the_best_F2.pth.tar"),
                    )
                  
                    evaluate(G1, G2, F1, F2, dset_loaders, **eval_kwargs_acc)
            if vit_acc_test > the_best_acc_vit_test:
                the_best_acc_vit_test = vit_acc_test
                if config["save_models"]:
                    torch.save(
                        G1.state_dict(),
                        os.path.join(config["output_path_model"], "the_best_G1.pth.tar"),
                    )
                    torch.save(
                        F1.state_dict(),
                        os.path.join(config["output_path_model"], "the_best_F1.pth.tar"),
                    )
            if auc_cnn_target_test > The_best_AUC_CNN_Target_Test:
                The_best_AUC_CNN_Target_Test = auc_cnn_target_test
                best_acc_cnn = eval_result["cnn_acc_target_test"]
                best_spe_cnn = eval_result["spe_cnn_target_test"]
                best_sen_cnn = eval_result["sen_cnn_target_test"]
                best_f1s_cnn = eval_result["f1s_cnn_target_test"]
                best_auc_cnn = The_best_AUC_CNN_Target_Test
                torch.save(
                    {
                        "G1": G1.state_dict(),  
                        "F1": F1.state_dict(), 
                        "G2": G2.state_dict(),
                        "F2": F2.state_dict(),
                        "step": step,
                        "best_auc_vit": The_best_AUC_CNN_Target_Test,
                    },
                    os.path.join(config["output_path_model"], "model_checkpoint_cnn.pth")
                )
                evaluate(G1, G2, F1, F2, dset_loaders, **eval_kwargs_auc)

            if auc_vit_target_test > The_best_AUC_ViT_Target_Test:
                The_best_AUC_ViT_Target_Test = auc_vit_target_test
                best_acc_vit = eval_result["vit_acc_target_test"]
                best_spe_vit = eval_result["spe_vit_target_test"]
                best_sen_vit = eval_result["sen_vit_target_test"]
                best_f1s_vit = eval_result["f1s_vit_target_test"]
                best_auc_vit = The_best_AUC_ViT_Target_Test
                torch.save(
                        {
                            "G1": G1.state_dict(),
                            "F1": F1.state_dict(),
                            "G2": G2.state_dict(),
                            "F2": F2.state_dict(),
                            "step": step,
                            "best_auc_vit":The_best_AUC_ViT_Target_Test,
                        },
                        os.path.join(config["output_path_model"], "model_checkpoint_vit.pth")
                )
            log_str += (
                "\t-- The best CNN's Acc Target Test = {:<05.4f}% The best ViT's Acc Target Test = {:<05.4f}% \n"
                "\t-- Acc_Pseudo_Labels_CNN = {:<05.4f} Correct_Pseudo_Labels_CNN = {} Total_Pseudo_Labels_CNN = {:<10} \n"
                "\t-- Acc_Pseudo_Labels_ViT = {:<05.4f} Correct_Pseudo_Labels_ViT = {} Total_Pseudo_Labels_ViT = {:<10} \n"
                "\t--The_best_AUC_CNN_Target_Test = {:<05.4f}  The_best_AUC_ViT_Target_Test= {:.4f} \n" #.format(
                "\t--best_acc_cnn={:<05.4f}  best_spe_cnn={:<05.4f}  best_sen_cnn={:<05.4f}  best_f1s_cnn={:<05.4f} \n"
                "\t--best_acc_vit={:<05.4f}  best_spe_vit={:<05.4f}  best_sen_vit={:<05.4f}  best_f1s_vit={:<05.4f} \n"
                ).format(
                the_best_acc_cnn_test,
                the_best_acc_vit_test,
                eval_result["pl_acc_cnn"],
                eval_result["correct_pl_cnn"],
                eval_result["total_pl_cnn"],
                eval_result["pl_acc_vit"],
                eval_result["correct_pl_vit"],
                eval_result["total_pl_vit"],
              
                The_best_AUC_CNN_Target_Test,
                The_best_AUC_ViT_Target_Test,
                eval_result["best_acc_cnn"],
                eval_result["best_spe_cnn"],
                eval_result["best_sen_cnn"],
                eval_result["best_f1s_cnn"],
                eval_result["best_acc_vit"],
                eval_result["best_spe_vit"],
                eval_result["best_sen_vit"],
                eval_result["best_f1s_vit"],
            )
            write_logs(out_file, log_str, colors=True)

    return G1, G2, F1, F2, {
        **eval_result,
        "best_auc_cnn": The_best_AUC_CNN_Target_Test,
        "best_auc_vit": The_best_AUC_ViT_Target_Test,
    }, The_best_AUC_CNN_Target_Test, best_acc_cnn, best_spe_cnn, best_sen_cnn, best_f1s_cnn, The_best_AUC_ViT_Target_Test, best_acc_vit, best_spe_vit, best_sen_vit, best_f1s_vit
