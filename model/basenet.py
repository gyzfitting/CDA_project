import copy
import timm
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models
from model.resnet import resnet34, resnet50
from model.vit3d import ViT3D
from vit3d import Vision_Transformer3D, DeiT_Transformer3D
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import monai
from monai.networks.nets import vit,resnet,SwinUNETR
import timm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)

class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=512, temp=0.05):
        super(Predictor_deep2, self).__init__()
        self.fc1 = nn.Linear(inc, 2)

    def forward(self, x):
        x = self.fc1(x)
        return x

def load_pretrain(model, pretrain_path, device='cpu', init_weight=True):
    if pretrain_path:
        print('Loading:', pretrain_path)
        model.load_state_dict(torch.load(pretrain_path, map_location=device), strict=True) 
        print(f'Pretrain weights {model.__class__.__name__} loaded.')
    elif init_weight:
        print(f"Weight Init: {model.__class__.__name__}")
        weights_init(model)
    else:
        print(f"Loading pretrain ImageNet: {model.__class__.__name__}")
    return model

def build_model(config, DEVICE, pretrain = True):
    config = copy.deepcopy(config)   
    backbone_setting = config['Backbone']   
    classifier_setting = config['Classifier']
    inc_1  = 256 
    inc_2 = 256    
    if backbone_setting['name_1'] == 'vit':
        G1 = timm.create_model(model_name='vit_base_patch16_224', pretrained=True)
    elif backbone_setting['name_1'] == '3dvit':
        imagesize = (176, 208, 176)
        G1 = ViT3D(
            image_size=imagesize,
            patch_size=8,
            num_classes=2,
            dim=256,
            depth=6,
            heads=6,
            mlp_dim=512,
            pool='mean',
            dropout=0.2,
            emb_dropout=0.1 
        )
        pretrained_dict = torch.load('./pretrained_models/ViT_B_pretrained_mae75_MICCAI.pth.tar', map_location=DEVICE)
        model_dict = G1.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        G1.load_state_dict(model_dict)

    elif backbone_setting['name_1'] == 'resnet18':
        G1 = monai.networks.nets.resnet18(pretrained=False, n_input_channels=1, num_classes=2,
                                             spatial_dims=3, feed_forward=False)
        G1.load_state_dict(torch.load('./pretrained_models/Pretext_Seg_epoch30.pth'), strict=False)
    elif backbone_setting['name_1'] == 'resnet34':
        G1 = monai.networks.nets.resnet34(pretrained=False, n_input_channels=1, num_classes=2,
                                             spatial_dims=3, feed_forward=False)
        G1.load_state_dict(torch.load('./pretrained_models/Pretext_Seg_epoch30.pth'), strict=False)
    elif backbone_setting['name_1'] == 'resnet50':
        G1 = monai.networks.nets.resnet50(pretrained=False, n_input_channels=1, num_classes=2,
                                             spatial_dims=3, feed_forward=False)
        G1.load_state_dict(torch.load('./pretrained_models/Pretext_Seg_epoch30.pth'), strict=False)  

    elif backbone_setting['name_1'] == 'resnet3d':

        G1 = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
     
        raise ValueError('Model cannot be recognized.')

    G1 = G1.to(DEVICE)
      
    if backbone_setting['name_2'] == '3dvit':
        imagesize = (176, 208, 176)
        G2 = ViT3D(
            image_size=imagesize,
            patch_size=8,
            num_classes=2,
            dim=256,
            depth=6,
            heads=6,
            mlp_dim=512,
            pool='mean',
            dropout=0.2,
            emb_dropout=0.1 
        )
        pretrained_dict = torch.load('./pretrained_models/ViT_B_pretrained_mae75_MICCAI.pth.tar', map_location=DEVICE)
        model_dict = G1.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        G2.load_state_dict(model_dict)
    elif backbone_setting['name_2'] == 'resnet18':
        G2 = monai.networks.nets.resnet18(pretrained=False, n_input_channels=1, num_classes=2,
                                      spatial_dims=3, feed_forward=False)
        G2.load_state_dict(torch.load('./pretrained_models/Pretext_Seg_epoch30.pth'), strict=False)
    elif backbone_setting['name_2'] == 'resnet34':
        G2 = monai.networks.nets.resnet34(pretrained=False, n_input_channels=1, num_classes=2,
                                             spatial_dims=3, feed_forward=False)
        G2.load_state_dict(torch.load('./pretrained_models/Pretext_Seg_epoch30.pth'), strict=False)
    elif backbone_setting['name_2'] == 'resnet50':
        G2 = monai.networks.nets.resnet50(pretrained=False, n_input_channels=1, num_classes=2,
                                             spatial_dims=3, feed_forward=False)
        G2.load_state_dict(torch.load('./pretrained_models/Pretext_Seg_epoch30.pth'), strict=False)   

    elif backbone_setting['name_2'] == "resnet3d":
        G2 = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    else:
        raise ValueError('Model cannot be recognized.')
    G2 = G2.to(DEVICE)   
    F1 = Predictor_deep(num_class=config['class_num'], inc=inc_1).to(DEVICE) 
    F2 = Predictor_deep(num_class=config['class_num'], inc=inc_2).to(DEVICE)

    if pretrain:       
        print("======== LOAD PRETRAIN ========")
        G1 = load_pretrain(G1, backbone_setting['pretrained_1'], device=DEVICE, init_weight=False) #改为了nit_weight=True
        G2 = load_pretrain(G2, backbone_setting['pretrained_2'], device=DEVICE, init_weight=False)
        F1 = load_pretrain(F1, classifier_setting['pretrained_F1'], device=DEVICE)
        F2 = load_pretrain(F2, classifier_setting['pretrained_F2'], device=DEVICE)
        print("===============================")   


    return G1, G2, F1, F2
