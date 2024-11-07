import logging
import glob
import torch
import torch
import os
from typing import List, Union
import pickle
import pandas as pd
import torch
import cv2
import json
from omegaconf import DictConfig
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from timm.data import create_transform
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import yaml
from skimage.transform import rescale, resize, downscale_local_mean

import h5py
import torchvision
from PIL import Image
import scipy.io as io
from torchvision import transforms, datasets
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    DistributedSampler,
    SequentialSampler,
)
from torchvision import transforms, datasets
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    DistributedSampler,
    SequentialSampler,
)


logger = logging.getLogger(__name__)

# with open("/DATA/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/config.yaml", "r") as f:
#     my_dict = yaml.load(f, Loader=yaml.Loader)

train_transformations = transforms.Compose(
    [  # Training Transform
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ]
)



class ImageNet(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, split)

        if split == "train":
            # syn_id = entry
            # target = self.syn_to_class[syn_id.split('_')[0]]
            # syn_folder = os.path.join(samples_dir, syn_id)
            # for sample in os.listdir(syn_folder):
            #     sample_path = os.path.join(syn_folder, sample)
            #     self.samples.append(sample_path)
            #     self.targets.append(target)
            for train_path in glob.glob(samples_dir + "/*.JPEG"):
                syn_id = os.path.basename(train_path).split(".")[0]
                target = self.syn_to_class[syn_id.split("_")[0]]
                self.samples.append(train_path)
                self.targets.append(target)
        elif split == "val":
            for val_path in glob.glob(samples_dir + "/*.JPEG"):
                syn_id = os.path.basename(val_path).split(".")[0]
                target = self.syn_to_class[syn_id.split("_")[-1].split(".")[0]]
                self.samples.append(val_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


class Cub2011(Dataset):

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        # transform_cfg: DictConfig,
        channels: List[int] = [0, 1, 2],  # use all rgb channels
        scale: float = 1,
    ):

        self.train = train
        self.root = root
        self.scale = scale
        self.transform = transform
        self.channels = torch.tensor([c for c in channels])
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(
                self.root, "CUB_200_2011", "CUB_200_2011", "image_class_labels.txt"
            ),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(
                self.root, "CUB_200_2011", "CUB_200_2011", "train_test_split.txt"
            ),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        class_names = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "CUB_200_2011", "classes.txt"),
            sep=" ",
            names=["class_name"],
            usecols=[1],
        )
        self.class_names = class_names["class_name"].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(
            self.root, "CUB_200_2011", "CUB_200_2011", "images", sample.filepath
        )
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = cv2.imread(path)
        # to PIL.Image
        img = Image.fromarray(img)

        img_chw = self.transform(img)

        if type(img_chw) is list:
            img_chw = [img[self.channels, :, :] for img in img_chw]
        else:
            img_chw = img_chw[self.channels, :, :]

        channels = self.channels
        if self.scale != 1:
            if type(img_chw) is list:
                # multi crop for DINO training
                img_chw = [c * self.scale for c in img_chw]
            else:
                # single view for linear probing
                img_chw *= self.scale
            # img_chw, {"label": target, "channels": channels}
        return img_chw, target


class LAEData_Train:
    def __init__(self, train_path, label_path, hd5_path, transform=None):
        self.annotations = np.load(
            train_path,
            allow_pickle=True,
        )
        self.Label = np.load(
            label_path,
            allow_pickle=True,
        )
        self.hd5_path = hd5_path
        self.Label = np.array(self.Label)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        key = self.annotations[index]
        with h5py.File(
            self.hd5_path, "r"
        ) as f:  # Database for Left Channel Test Spectrogram
            SG_Data = f[key][()]
            SG_Data = np.array(SG_Data)
            SG_Label = torch.from_numpy(np.array((self.Label[index])))
            r1 = 0
            r2 = 127
            ES_Data = SG_Data[r1:r2, :]
            # ES_Data=np.stack((ES_Data, ES_Data,ES_Data))
            # print(ES_Data.shape)
            ES_Data = Image.fromarray(ES_Data)

            if self.transform:
                ES_Data = self.transform(ES_Data)
        return ES_Data, SG_Label

class SignalSet(Dataset):
    
    def __init__(self, root="", mode="train", n_class=8, n_snr=10, transform=None):
        self.input_sig_dict = dict()
        self.n_class = n_class
        self.n_snr = n_snr
        self.list_snr = list(28 - 2 * np.arange(self.n_snr))
        self.list_snr.reverse()
        self.transform = transform
        ref_point_1 = int(288 * 0.8)
        ref_point_2 = int(288 * 0.9)
        if mode == "train":
            start, end = 0, ref_point_1
        elif mode == "valid":
            start, end = ref_point_1, ref_point_2
        else:
            start, end = ref_point_2, 288
        self.n_inst = end - start

        mat_list = glob.glob(root + "/*.mat")
        for mat in mat_list:
            mod_type = mat.split("/")[-1].split("_")[0]
            snr = int(mat.split("/")[-1].split("_")[1][:2])
            if snr in self.list_snr:
                tot_arr = io.loadmat(mat)["dataset"]
                data_arr = tot_arr[start:end, :]
                self.input_sig_dict[(mod_type, snr)] = data_arr
    
    def class2num(self):
        CLASS2NUM = {
            "BPSK": 0,
            "QPSK": 1,
            "8PSK": 2,
            "16QAM": 3,
            "32QAM": 4,
            "64QAM": 5,
            "128QAM": 6,
            "256QAM": 7,
        }
        return CLASS2NUM

    def num2class(self):
        old_dict = self.class2num()
        NUM2CLASS = dict([(value, key) for key, value in old_dict.items()])
        return NUM2CLASS

    def __getitem__(self, index):
        ind_mod_type = index // (self.n_snr * self.n_inst)
        ind_snr = (index % (self.n_snr * self.n_inst)) // self.n_inst
        ind_inst = (index % (self.n_snr * self.n_inst)) % self.n_inst

        mod_type = self.num2class()[ind_mod_type]
        snr = self.list_snr[ind_snr]
        input_sig_array = self.input_sig_dict[(mod_type, snr)]
        input_ = input_sig_array[ind_inst, :]
        in_i = np.expand_dims(input_.real, axis=-1)
        in_q = np.expand_dims(input_.imag, axis=-1) 
        input_ = np.concatenate((in_i, in_q), axis=-1)
        input_ = resize(input_, (224, 224))
        input_ = np.expand_dims(input_, axis=0)
        input_ = torch.from_numpy(input_)
        input_ = torch.repeat_interleave(input_, 3, dim=0)
        return {
            "input": input_,
            "modtype": mod_type,
            "snr": snr,
        }

    def __len__(self):
        return self.n_class * self.n_snr * self.n_inst

    

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (args.img_size, args.img_size), scale=(0.05, 1.0)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(
            root="/DATA/nfsshare/Amartya/EMNLP-WACV/ViT-pytorch",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = (
            datasets.CIFAR10(
                root="/DATA/nfsshare/Amartya/EMNLP-WACV/ViT-pytorch",
                train=False,
                download=True,
                transform=transform_test,
            )
            if args.local_rank in [-1, 0]
            else None
        )

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(
            root="/DATA/nfsshare/Amartya/EMNLP-WACV/ViT-pytorch",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = (
            datasets.CIFAR100(
                root="/DATA/nfsshare/Amartya/EMNLP-WACV/ViT-pytorch",
                train=False,
                download=True,
                transform=transform_test,
            )
            if args.local_rank in [-1, 0]
            else None
        )
    elif args.dataset == "cub2011":
        trainset = Cub2011(
            root="/DATA/nfsshare/Amartya/EMNLP-WACV/channel_mamba",
            train=True,
            transform=transform_train,
        )
        testset = (
            Cub2011(
                root="/DATA/nfsshare/Amartya/EMNLP-WACV/channel_mamba",
                train=False,
                transform=transform_test,
            )
            if args.local_rank in [-1, 0]
            else None
        )
    elif args.dataset == "imagenet200":
        trainset = torchvision.datasets.ImageFolder(
            root="/DATA/nfsshare/Amartya/EMNLP-WACV/ViT-pytorch/tiny-imagenet-200/train",
            transform=transform_train,
        )
        testset = (
            torchvision.datasets.ImageFolder(
                root="/DATA/nfsshare/Amartya/EMNLP-WACV/ViT-pytorch/tiny-imagenet-200/val",
                transform=transform_test,
            )
            if args.local_rank in [-1, 0]
            else None
        )
    elif args.dataset == "imagenet1k":
        trainset = torchvision.datasets.ImageFolder(
            root="/nfsshare/Amartya/EMNLP-WACV/imagenet/train",
            transform=transform_train,
        )
        testset = (
            torchvision.datasets.ImageFolder(
                root="/nfsshare/Amartya/EMNLP-WACV/imagenet/val",
                transform=transform_test,
            )
            if args.local_rank in [-1, 0]
            else None
        )
    elif args.dataset == "stanfordcars":
        trainset = torchvision.datasets.StanfordCars(
            root=os.path.join(args.data_path, "train"),
            transform=transform_train,
            download=True,
        )
        testset = torchvision.datasets.StanfordCars(
            root=os.path.join(args.data_path, "test"),
            transform=transform_test,
            download=True,
        )
    elif args.dataset == "aircraft":
        trainset = torchvision.datasets.FGVCAircraft(
            root=os.path.join(args.data_path, "train"),
            transform=transform_train,
            download=True,
        )
        testset = torchvision.datasets.FGVCAircraft(
            root=os.path.join(args.data_path, "val"),
            transform=transform_test,
            download=True,
        )
    elif args.dataset == "dtd":
        trainset = torchvision.datasets.DTD(
            root=os.path.join(args.data_path, "train"),
            transform=transform_train,
            download=True,
        )
        testset = torchvision.datasets.DTD(
            root=os.path.join(args.data_path, "val"),
            transform=transform_test,
            download=True,
        )
    elif args.dataset == "oxford":
        trainset = torchvision.datasets.OxfordIIITPet(
            root=os.path.join(args.data_path, "trainval"),
            transform=transform_train,
            download=True,
        )
        testset = torchvision.datasets.OxfordIIITPet(
            root=os.path.join(args.data_path, "test"),
            transform=transform_test,
            download=True,
        )
    elif args.dataset == "flowers":
        trainset = torchvision.datasets.Flowers102(
            root=os.path.join(args.data_path, "train"),
            transform=transform_train,
            download=True,
        )
        testset = torchvision.datasets.Flowers102(
            root=os.path.join(args.data_path, "val"),
            transform=transform_test,
            download=True,
        )
    elif args.dataset == "stl":
        trainset = torchvision.datasets.STL10(
            root=os.path.join(args.data_path, "train"),
            transform=transform_train,
            download=True,
        )
        testset = torchvision.datasets.STL10(
            root=os.path.join(args.data_path, "test"),
            transform=transform_test,
            download=True,
        )
    elif args.dataset == "communication":
        trainset = SignalSet(
            root=os.path.join(args.data_path),
            mode="train",
            transform=transform_train,
        )
        testset = SignalSet(
            root=os.path.join(args.data_path),
            mode="train",
            transform=transform_test,
        )
    elif args.dataset == "fsc2022":
        Train_Dataset = LAEData_Train(transform=train_transformations)
        train_size = int(0.8 * len(Train_Dataset))
        valid_size = len(Train_Dataset) - train_size
        trainset, testset = torch.utils.data.random_split(
            Train_Dataset, [train_size, valid_size]
        )
    elif args.dataset == "esc50":
        
        Train_Dataset = LAEData_Train(
            train_path="/DATA/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/ESC50TrainData.npy",
            label_path="/DATA/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/ESC50TrainLabel.npy",
            hd5_path="/DATA/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/ESC50.hdf5",
            transform=train_transformations,
        )
        train_size = int(0.8 * len(Train_Dataset))
        valid_size = len(Train_Dataset) - train_size
        trainset, testset = torch.utils.data.random_split(
            Train_Dataset, [train_size, valid_size]
        )
    
    elif args.dataset == "dcase":
        
        trainset = LAEData_Train(
            train_path="/DATA/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/DCASETrainData.npy",
            label_path="/DATA/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/DCASETrainLabel.npy",
            hd5_path="/DATA/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/DCASE19.hdf5",
            transform=train_transformations,
        )
        testset = LAEData_Train(
            train_path="/DATA/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/DCASETestData.npy",
            label_path="/DATA/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/DCASETestLabel.npy",
            hd5_path="/DATA/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/DCASE19.hdf5",
            transform=train_transformations,
        )
        # train_size = int(0.8 * len(Train_Dataset))
        # valid_size = len(Train_Dataset) - train_size
        # trainset, testset = torch.utils.data.random_split(
        #     Train_Dataset, [train_size, valid_size]
        # )
        
    # if args.local_rank == 0:
    #     torch.distributed.barrier()

    # train_sampler = (
    #     RandomSampler(trainset)
    #     if args.local_rank == -1
    # train_loader#     else DistributedSampler(trainset)
    # )
    # test_sampler = SequentialSampler(testset)
    # sampler=train_sampler,
    train_loader = DataLoader(
        trainset,
        batch_size=args.train_batch_size,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = (
        DataLoader(
            testset,
            batch_size=args.eval_batch_size,
            num_workers=8,
            drop_last=True,
            pin_memory=True,
        )
        if testset is not None
        else None
    )

    return train_loader, test_loader
