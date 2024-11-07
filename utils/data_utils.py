import logging

import torch
import torch
import os
from typing import List, Union
import pickle
import pandas as pd
import torch
import cv2
from omegaconf import DictConfig
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from timm.data import create_transform
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import yaml
import h5py
from PIL import Image
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

with open("/DATA/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/config.yaml", "r") as f:
    my_dict = yaml.load(f, Loader=yaml.Loader)

train_transformations = transforms.Compose(
    [  # Training Transform
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ]
)

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
    def __init__(self, transform=None):
        self.annotations = np.load(
            "/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/FSC22TrainData.npy",
            allow_pickle=True,
        )  # Read The names of Test Signals
        self.Label = np.load(
            "/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/FSC22TrainLabel.npy",
            allow_pickle=True,
        )
        self.Label = np.array(self.Label)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        key = self.annotations[index]
        with h5py.File(
            "/nfsshare/Amartya/EMNLP-WACV/vit-reinforcement/DCASE19.hdf5", "r"
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
            root="/nfsshare/Amartya/EMNLP-WACV/ViT-pytorch",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = (
            datasets.CIFAR10(
                root="/nfsshare/Amartya/EMNLP-WACV/ViT-pytorch",
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
        # trainset = torchvision.datasets.ImageFolder(
        #     root="/DATA/nfsshare/Amartya/EMNLP-WACV/ViT-pytorch/tiny-imagenet-200/train",
        #     transform=transform_train,
        # )
        # testset = (
        #     torchvision.datasets.ImageFolder(
        #         root="/DATA/nfsshare/Amartya/EMNLP-WACV/ViT-pytorch/tiny-imagenet-200/val",
        #         transform=transform_test,
        #     )
        #     if args.local_rank in [-1, 0]
        #     else None
        # )
        trainset = torchvision.datasets.ImageNet(root='/export/home/vivian/imagenet', split='train', transform=transform_train)
        testset = (
            torchvision.datasets.ImageFolder(
                root='/export/home/vivian/imagenet',
                split='val'
                transform=transform_test,
            )
            if args.local_rank in [-1, 0]
            else None
        )
    elif args.dataset == "fsc2022":
        Train_Dataset = LAEData_Train(transform=train_transformations)
        train_size = int(0.8 * len(Train_Dataset))
        valid_size = len(Train_Dataset) - train_size
        trainset, testset = torch.utils.data.random_split(
            Train_Dataset, [train_size, valid_size]
        )
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = (
        RandomSampler(trainset)
        if args.local_rank == -1
        else DistributedSampler(trainset)
    )
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=12,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = (
        DataLoader(
            testset,
            sampler=test_sampler,
            batch_size=args.eval_batch_size,
            num_workers=12,
            drop_last=True,
            pin_memory=True,
        )
        if testset is not None
        else None
    )

    return train_loader, test_loader
