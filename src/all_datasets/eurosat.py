import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader, ConcatDataset

class EuroSAT:
    def __init__(self, preprocess, location: str = os.path.expanduser('~/data'), batch_size=64, num_workers=16):
        # Load the EuroSAT dataset from the specified location
        full_dataset = ImageFolder(root='/mnt/lts4/scratch/data/EuroSAT_splits', transform=preprocess)
        
        # Store class_to_idx mapping from the original dataset before splitting
        self.class_to_idx = full_dataset.class_to_idx
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]

        # Split the full dataset into train and test sets
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # Split the train dataset into three subsets
        finetune_train_size = int(0.04 * len(train_dataset))
        distillation_train_size = int(0.04 * len(train_dataset))
        merging_train_size = len(train_dataset) - finetune_train_size - distillation_train_size

        # Split the test dataset into three subsets
        finetune_test_size = int(0.04 * len(test_dataset))
        distillation_test_size = int(0.04 * len(test_dataset))
        merging_test_size = len(test_dataset) - finetune_test_size - distillation_test_size

        finetune_train_set, self.distillation_train_set, self.merging_train_set = random_split(
            train_dataset, [finetune_train_size, distillation_train_size, merging_train_size])
        
        finetune_test_set, self.distillation_test_set, self.merging_test_set = random_split(
            test_dataset, [finetune_test_size, distillation_test_size, merging_test_size])
        
        self.finetune_set = ConcatDataset([finetune_train_set, finetune_test_set])

        # Initialize DataLoaders
        self.train_loader = DataLoader(self.finetune_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.distillation_train_loader = DataLoader(self.distillation_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.merging_train_loader = DataLoader(self.merging_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.distillation_test_loader = DataLoader(self.distillation_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.merging_test_loader = DataLoader(self.merging_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def save_subsets(self, location):
        # Create dataset directories
        split_save_dir = os.path.join(location, 'Subsets', 'EuroSAT')
        os.makedirs(split_save_dir, exist_ok=True)

        # Save the split datasets
        torch.save(self.finetune_set, os.path.join(split_save_dir, 'finetune_set.pt'))
        torch.save(self.distillation_train_set, os.path.join(split_save_dir, 'distillation_train_set.pt'))
        torch.save(self.merging_train_set, os.path.join(split_save_dir, 'merging_train_set.pt'))
        torch.save(self.distillation_test_set, os.path.join(split_save_dir, 'distillation_test_set.pt'))
        torch.save(self.merging_test_set, os.path.join(split_save_dir, 'merging_test_set.pt'))