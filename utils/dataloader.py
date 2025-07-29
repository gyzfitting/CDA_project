import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os, argparse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchio as tio

transform = tio.transforms.Compose([
    tio.transforms.RandomFlip(axes=('LR', 'AP', 'IS')),   
    tio.transforms.RandomAffine(scales=(0.9, 1.1), degrees=(10, 10, 10)),  
    tio.transforms.RandomNoise(mean=0.0, std=(0, 0.05)), 
    tio.transforms.RandomBiasField(coefficients=0.5),     
    tio.transforms.RandomGamma(log_gamma=(0.9, 1.1)),      
    #tio.transforms.CropOrPad((128, 128, 128)),             
    #tio.transforms.RescaleIntensity((0, 1)),              
])
samplespace = 2
class MDD_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, augment_type="none" ):
        self.subjects = []
        # image_paths 和 labels
        for (image_path, label) in zip(image_paths, labels):
            subject = tio.Subject(
                # mri=tio.ScalarImage(image_path),
                mri=tio.ScalarImage(image_path, check_nifti=False),
                #labels=int(label),
                labels=int(float(label))  
            )
            self.subjects.append(subject)
        self.build_transforms(augment_type)

        self.dataset = tio.SubjectsDataset(self.subjects, transform=self.transform_pipeline_weak, transform2=self.transform_pipeline_strong) #load_getitem=False,preload=True,

    def build_transforms(self,augment_type):
        get_foreground = tio.ZNormalization.mean
        LI_LANDMARKS = "0 8.06305571158 15.5085721044 18.7007018006 21.5032879029 26.1413278906 29.9862059045 33.8384058795 38.1891334787 40.7217966068 44.0109152758 58.3906435207 100.0"
        LI_LANDMARKS = np.array([float(n) for n in LI_LANDMARKS.split()])
        landmarks_dict = {'mri': LI_LANDMARKS}
        preprocess = tio.Compose([
            tio.ToCanonical(),
            tio.RescaleIntensity(out_min_max=(0.0, 1.0)),
            tio.CropOrPad((176, 208, 176)),
            tio.Resample((samplespace, samplespace, samplespace)),           
            tio.ZNormalization(masking_method=get_foreground),                          
            
        ])
        weak_augment = tio.Compose([
            preprocess,          
            tio.transforms.RandomFlip(axes=(0, 1, 2)),
            tio.transforms.RandomAffine(scales=(0.9, 1.1), degrees=(5, 5, 5)), 
            tio.transforms.RandomBiasField(coefficients=0.2), 
            tio.transforms.RescaleIntensity((0, 1)), 

        ])

        strong_augment = tio.Compose([
            preprocess,
            tio.RandomAffine(scales=0.2, degrees=20, translation=10, isotropic=True, center='image'),  
            tio.RandomElasticDeformation(max_displacement=7, num_control_points=5), 
            tio.transforms.RandomNoise(mean=0.0, std=(0, 0.1)),  
            tio.transforms.RandomGamma(log_gamma=(0.7, 1.5)),  
            tio.RandomAffine(scales=(0.95, 1.05), degrees=(5, 5, 5)),
            tio.RandomNoise(mean=0.0, std=0.02),
        ])
      
        self.transform_pipeline_weak = weak_augment      
        self.transform_pipeline_strong = strong_augment       
        self.transform_pipeline = preprocess

    def __len__(self):
        return len(self.subjects)

def build_data(config,fold,debug=True):
    dsets = {}
    dset_loaders = {}
    source_bs = config["source"]["batch_size"]
    target_bs = config["target"]["batch_size"]

    source_name = config["source"]["name"]
    target_name = config["target"]["name"]

    method = config["method"]
    data_root = config["data_root"]
    data_label = f"./dataset/Data_text/Fold{fold}"

    if method == "UDA":
        prefix = config.get("data_prefix", {'train':'.txt', 'test': '.txt'})
        image_set_file_s = os.path.join(data_label, source_name + prefix['train'])
        image_set_file_s_test = os.path.join(data_label, source_name + prefix['test'])
        image_set_file_tu = os.path.join(data_label, target_name + prefix['train'])
        image_set_file_tu_test = os.path.join(data_label, target_name + prefix['test'])
    if debug:       
        print("="*10, 'DATA PATH', "="*10)
        print(f"source train: {image_set_file_s}")
        print(f"source test: {image_set_file_s_test}")
        print(f"target train: {image_set_file_tu}")
        print(f"target test: {image_set_file_tu_test}")
        print("="*31)
        print(f"Using Fold {fold} for data loading.")   
    filename1 = data_label+'_'+ source_name +'_train.txt'
    labels0 = np.genfromtxt(filename1, dtype=str)
    imagepaths = labels0[..., 0].tolist()
    label = labels0[..., 1].tolist()
    dsets["source_train"] = MDD_Dataset(image_paths=imagepaths, labels=label, augment_type="none" )
    
    filename1_test = data_label+'_'+ source_name +'_test.txt'
    labels0 = np.genfromtxt(filename1_test, dtype=str)
    imagepaths = labels0[..., 0].tolist()
    label = labels0[..., 1].tolist()
    dsets["source_test"] = MDD_Dataset(image_paths=imagepaths, labels=label, augment_type="weak")
  
    dset_loaders["source_train"] = DataLoader(
        dsets["source_train"].dataset,
        batch_size=source_bs,
        shuffle=True, #True
        num_workers=config["num_workers"],
        drop_last=True,
        pin_memory=False
    )

    dset_loaders["source_test"] = DataLoader(
        dsets["source_test"].dataset,
        batch_size=source_bs,
        num_workers=config["num_workers"],
        pin_memory=False
    )
   
    filename2=data_label+'_'+ target_name +'_train.txt'
    labels0 = np.genfromtxt(filename2, dtype=str)
    imagepaths = labels0[..., 0].tolist()
    label = labels0[..., 1].tolist()
    dsets["target_train"] = MDD_Dataset(image_paths=imagepaths, labels=label, augment_type="weak")

    filename2_test = data_label+'_'+ target_name +'_test.txt'   
    labels0 = np.genfromtxt(filename2_test, dtype=str)
    imagepaths = labels0[..., 0].tolist()
    label = labels0[..., 1].tolist()
    dsets["target_test"] = MDD_Dataset(image_paths=imagepaths, labels=label, augment_type="none")

    dset_loaders["target_train"] = DataLoader(
        dataset=dsets["target_train"].dataset,
        batch_size=target_bs,
        shuffle=True, #True
        num_workers=config["num_workers"],
        drop_last=True,
        pin_memory=False
    )

    dset_loaders["target_test"] = DataLoader(
        dataset=dsets["target_test"].dataset,
        batch_size=target_bs,
        num_workers=config["num_workers"],
        pin_memory=False
    ) 

    return dsets, dset_loaders

