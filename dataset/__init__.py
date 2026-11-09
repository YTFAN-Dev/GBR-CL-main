# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from dataset.utils.continual_dataset import ContinualDataset
from dataset.utils.gcl_dataset import GCLDataset

from dataset.seq_mnist import SequentialMNIST
from dataset.seq_cifar10 import SequentialCIFAR10
from dataset.seq_cifar100 import SequentialCIFAR100
from dataset.seq_tinyimagenet import SequentialTinyImagenet
from dataset.rot_mnist import RotatedMNIST
from dataset.perm_mnist import PermutedMNIST
from dataset.mnist_360 import MNIST360
from dataset.seq_cifar10_blurry import SequentialCIFAR10Blurry


NAMES = {
    SequentialMNIST.NAME: SequentialMNIST,
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
    RotatedMNIST.NAME: RotatedMNIST,
    PermutedMNIST.NAME: PermutedMNIST,
    MNIST360.NAME: MNIST360,
    SequentialCIFAR10Blurry.NAME: SequentialCIFAR10Blurry,
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)
    