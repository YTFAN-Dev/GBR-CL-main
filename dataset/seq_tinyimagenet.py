# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
from utils.conf import data_path
from PIL import Image
import os
from dataset.utils.validation import get_train_val
from dataset.utils.continual_dataset import ContinualDataset, store_masked_loaders
from dataset.utils.continual_dataset import get_previous_train_loader
from dataset.transforms.denormalization import DeNormalize


class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                import googledrivedownloader as gdd

                # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
                print('Downloading dataset')
                gdd.download_file_from_google_drive(
                    file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',
                    dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                    unzip=True)

        suffix = 'train' if self.train else 'val'

        # Load labels (small, keep in memory)
        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(root, 'processed/y_%s_%02d.npy' % (suffix, num+1))))
        self.targets = np.concatenate(self.targets)

        # Memory-map image data to avoid loading 1.14 GiB per instance
        cache_x = os.path.join(root, 'processed', 'x_%s_cache.dat' % suffix)
        if not os.path.exists(cache_x):
            parts = []
            for num in range(20):
                parts.append(np.load(os.path.join(root, 'processed/x_%s_%02d.npy' % (suffix, num+1))))
            concat = np.concatenate(parts)
            mm = np.memmap(cache_x, dtype=np.uint8, mode='w+', shape=concat.shape)
            mm[:] = concat[:]
            mm.flush()
            del concat, parts, mm
        self.data = np.memmap(cache_x, dtype=np.uint8, mode='r',
                              shape=(len(self.targets), 64, 64, 3))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class MyTinyImagenet(TinyImagenet):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        super(MyTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
          return img, target, not_aug_img, self.logits[index]

        return img, target,  not_aug_img


class SequentialTinyImagenet(ContinualDataset):

    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(64, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4802, 0.4480, 0.3975),
                                  (0.2770, 0.2691, 0.2821))])
    # TRANSFORM = transforms.Compose([transforms.RandomCrop(64, padding=4),
    #                                 transforms.RandAugment(1, 14),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.4802, 0.4480, 0.3975),
    #                                                      (0.2770, 0.2691, 0.2821))])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyTinyImagenet(data_path() + 'TINYIMG', train=True, download=False, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset, test_transform, self.NAME)
        else:
            test_dataset = TinyImagenet(data_path() + 'TINYIMG', train=False, download=False, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_denormalization_transform()])

        train_dataset = MyTinyImagenet(data_path() + 'TINYIMG', train=True, download=False, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_backbone(backbone_name='resnet18'):
        from backbone.ResNet import resnet18, resnet34, resnet50, resnet101, resnet152
        from backbone.ResNet_meta import resnet18_meta, resnet34_meta, resnet50_meta, resnet101_meta, resnet152_meta

        n_classes = SequentialTinyImagenet.N_CLASSES_PER_TASK * SequentialTinyImagenet.N_TASKS
        backbone_dict = {
            # ResNet
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152,
            'resnet18-meta': resnet18_meta,
            'resnet34-meta': resnet34_meta,
            'resnet50-meta': resnet50_meta,
            'resnet101-meta': resnet101_meta,
            'resnet152-meta': resnet152_meta,
        }
        assert backbone_name in backbone_dict.keys()
        backbone = backbone_dict[backbone_name](n_classes)
        return backbone

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None