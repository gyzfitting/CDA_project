import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_mmd_loss(source_features, target_features): 
    def gaussian_kernel(x, y, sigma=1.0):
        beta = 1.0 / (2 * sigma ** 2)
        dist = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-beta * dist)

    k_ss = gaussian_kernel(source_features, source_features)
    k_tt = gaussian_kernel(target_features, target_features)
    k_st = gaussian_kernel(source_features, target_features)
    return k_ss.mean() + k_tt.mean() - 2 * k_st.mean()
def compute_ka_loss(source_features, target_features):
    """ 计算核矩匹配 (Kernel Alignment, KA) 损失 """
    k_ss = torch.mm(source_features, source_features.T)
    k_tt = torch.mm(target_features, target_features.T)
    k_st = torch.mm(source_features, target_features.T)
    return torch.mean(k_ss + k_tt - 2 * k_st)

class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.25, 0.75], gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).to("cuda")
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss) 
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def jsd_divergence(p, q):
    m = 0.5 * (p + q)
    kl_p_m = F.kl_div(p.log(), m, reduction='none').sum(dim=1)
    kl_q_m = F.kl_div(q.log(), m, reduction='none').sum(dim=1)
    return 0.5 * (kl_p_m + kl_q_m)

def ce_loss(logits, targets, reduction="none"):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
  
    if logits.shape == targets.shape:
      
        log_pred = F.log_softmax(logits, dim=-1)
   
        nll_loss = torch.sum(-targets * log_pred, dim=1)
      
        if reduction == "none":
            return nll_loss
        else:
            return nll_loss.mean()
  
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


def consistency_loss(logits_u_str, logits_u_w, threshold=0.5): #0.6
    """
    Consistency regularization for fixed threshold loss in semi-supervised learning.
    Args:
        logits_u_str: logits of strong augmented unlabeled samples
        logits_u_w: logits of weak augmented unlabeled samples
        threshold: fixed threshold
    Returns:
        loss: consistency regularization loss
    """
    pseudo_label = torch.softmax(logits_u_w, dim=1)
    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(threshold).float()
    loss = (ce_loss(logits_u_str, targets_u, reduction="none") * mask).mean()
    return loss
