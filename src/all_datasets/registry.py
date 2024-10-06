import sys
import inspect
import random
import torch
import copy
import os

from torch.utils.data.dataset import random_split

from all_datasets.cars import Cars
from all_datasets.cifar10 import CIFAR10
from all_datasets.cifar100 import CIFAR100
from all_datasets.dtd import DTD
from all_datasets.eurosat import EuroSAT, EuroSATVal
from all_datasets.gtsrb import GTSRB
from all_datasets.imagenet import ImageNet
from all_datasets.mnist import MNIST
from all_datasets.resisc45 import RESISC45
from all_datasets.stl10 import STL10
from all_datasets.svhn import SVHN
from all_datasets.sun397 import SUN397

registry = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}

class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None

def split_train_into_train_val_for_finetuning(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0):
    print("Splitting finetune train data into train/validation")
    assert val_fraction > 0. and val_fraction < 1.

    total_size = len(dataset.finetune_set)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(
        dataset.finetune_set,
        lengths,
        generator=torch.Generator().manual_seed(seed)
    )
    #if new_dataset_class_name == 'MNISTVal':
        #assert trainset.indices[0] == 36044

    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset, ), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def get_dataset(dataset_name, preprocess, location, batch_size=64, num_workers=8, val_fraction=0.3, max_val_samples=5000):
    # This function returns a dataset based on the provided name and parameters.
    # If the dataset_name ends with 'Val', it will handle splitting a base dataset into train/validation subsets.
    # Otherwise, it retrieves the dataset directly from a registry of predefined datasets.
    if dataset_name.endswith('Val'):
        # Check if the dataset name ends with 'Val', indicating we are asking for a validation split.
        
        # Handle the case where the 'Val' dataset is not directly in the registry.
        # Remove the 'Val' suffix from the dataset name to get the base dataset name.
        base_dataset_name = dataset_name.split('Val')[0]
        
        # Recursively call the get_dataset function to fetch the base dataset (without 'Val').
        base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers)
        
        # Split the base dataset into training and validation subsets.
        dataset = split_train_into_train_val_for_finetuning(
            base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples
        )
        # Return the validation split of the dataset.
        return dataset
    else:
        # If the dataset name does not end with 'Val', directly retrieve the dataset class.
        
        # Ensure the dataset exists in the registry, otherwise raise an assertion error with a message.
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        
        # Get the dataset class from the registry using the dataset_name as the key.
        dataset_class = registry[dataset_name]
        print(dataset_class)

    # Instantiate the dataset class by passing required parameters like preprocessing, location, batch size, and workers.
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, num_workers=num_workers
    )

    print("saving finetune, distillation and merging train and test subsets ...")
    dataset.save_subsets(location)
    
    # Return the instantiated dataset.
    return dataset