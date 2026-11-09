# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import datasets
import numpy as np
import torch.optim


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args
        
        # Flexible splitting
        if self.N_TASKS is not None and self.N_CLASSES_PER_TASK is not None:
            self.total_classes = self.N_TASKS * self.N_CLASSES_PER_TASK
        else:
            self.total_classes = 0  # specific datasets might override this

        # 1. Class Order (Permutation)
        if hasattr(args, 'rand_split') and args.rand_split and self.total_classes > 0:
            self.class_order = np.random.permutation(self.total_classes)
        elif self.total_classes > 0:
            self.class_order = np.arange(self.total_classes)
        else:
            self.class_order = None

        # 2. Task Schedule (Number of classes per task)
        if hasattr(args, 'tasks_schedule') and args.tasks_schedule is not None:
            self.tasks_schedule = args.tasks_schedule
            # Update N_TASKS if schedule dictates
            self.N_TASKS = len(self.tasks_schedule)
        elif self.N_TASKS is not None and self.N_CLASSES_PER_TASK is not None:
            self.tasks_schedule = [self.N_CLASSES_PER_TASK] * self.N_TASKS
        else:
            self.tasks_schedule = None

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass

    @staticmethod
    def get_minibatch_size():
        pass


def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    task_idx = len(setting.test_loaders)
    
    # Determine the classes for the current task
    if setting.class_order is not None and setting.tasks_schedule is not None:
        # Calculate offset
        start_idx = sum(setting.tasks_schedule[:task_idx])
        num_classes_current = setting.tasks_schedule[task_idx]
        end_idx = start_idx + num_classes_current
        
        current_classes = setting.class_order[start_idx:end_idx]
        
        # Advance the counter (setting.i) for legacy reasons or display
        setting.i = end_idx 
    else:
        # Fallback to original logic if not properly initialized
        # This assumes sequential classes
        current_classes = np.arange(setting.i, setting.i + setting.N_CLASSES_PER_TASK)
        setting.i += setting.N_CLASSES_PER_TASK

    train_mask = np.isin(np.array(train_dataset.targets), current_classes)
    test_mask = np.isin(np.array(test_dataset.targets), current_classes)

    try:
        train_dataset.data = train_dataset.data[train_mask]
        test_dataset.data = test_dataset.data[test_mask]
    except TypeError:
        train_dataset.data = [train_dataset.data[d] for d in np.where(train_mask > 0)[0]]
        test_dataset.data = [test_dataset.data[d] for d in np.where(test_mask > 0)[0]]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    print(f'Number of seen classes={setting.i}\tCurrent classes={current_classes}')
    train_loader = DataLoader(train_dataset, batch_size=setting.args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=0)
    
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    return train_loader, test_loader


def store_blurry_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                                setting: ContinualDataset, current_task) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    # print(setting.train_idx_all[current_task].max(), len(np.array(train_dataset.targets)))
    train_dataset.data = train_dataset.data[setting.train_idx_all[current_task]]
    test_dataset.data = test_dataset.data[setting.test_idx_all[current_task]]

    train_dataset.targets = np.array(train_dataset.targets)[setting.train_idx_all[current_task]]
    test_dataset.targets = np.array(test_dataset.targets)[setting.test_idx_all[current_task]]

    train_loader = DataLoader(train_dataset, batch_size=setting.args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=0)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    # NOTE: This function attempts to get the 'previous' task. 
    # With flexible scheduling, we need to recalculate based on tasks_schedule.
    
    # Current state: setting.i points to the END of the current task (or next start).
    # We want task t-1.
    task_idx = len(setting.test_loaders) # This is current task index (0-based) during training?
    # Usually get_previous_train_loader is called during the current task training, so we want the PREVIOUS one.
    
    if setting.class_order is not None and setting.tasks_schedule is not None:
        # Determine index of previous task
        prev_task_idx = task_idx - 1
        if prev_task_idx < 0:
            # Should not happen if called correctly
            return None
            
        start_idx = sum(setting.tasks_schedule[:prev_task_idx])
        num_classes_prev = setting.tasks_schedule[prev_task_idx]
        end_idx = start_idx + num_classes_prev
        
        prev_classes = setting.class_order[start_idx:end_idx]
    else:
        # Fallback
        prev_classes = np.arange(setting.i - setting.N_CLASSES_PER_TASK * 2, setting.i - setting.N_CLASSES_PER_TASK)

    train_mask = np.isin(np.array(train_dataset.targets), prev_classes)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
