import os
import torch
import torchvision.datasets as datasets
from torch.utils.data import random_split

class MNIST:
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), batch_size=64, num_workers=16):

        train_dataset = datasets.MNIST(root=location, download=True, train=True, transform=preprocess)
        test_dataset = datasets.MNIST(root=location, download=True, train=False, transform=preprocess)

        # Split the train dataset into three subsets
        finetune_train_size = int(0.4 * len(train_dataset)) 
        distillation_train_size = int(0.4 * len(train_dataset)) 
        merging_train_size = len(train_dataset) - finetune_train_size - distillation_train_size

        # Split the test dataset into three subsets
        finetune_test_size = int(0.4 * len(test_dataset))
        distillation_test_size = int(0.4 * len(test_dataset)) 
        merging_test_size = len(test_dataset) - finetune_test_size - distillation_test_size

        finetune_train_set, self.distillation_train_set, self.merging_train_set = random_split(
            train_dataset, [finetune_train_size, distillation_train_size, merging_train_size])

        finetune_test_set, self.distillation_test_set, self.merging_test_set = random_split(
            test_dataset, [finetune_test_size, distillation_test_size, merging_test_size])
        
        self.finetune_set = torch.utils.data.ConcatDataset([finetune_train_set, finetune_test_set])
        
        self.train_loader = torch.utils.data.DataLoader(self.finetune_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.distillation_train_loader = torch.utils.data.DataLoader(self.distillation_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.merging_train_loader = torch.utils.data.DataLoader(self.merging_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.distillation_test_loader = torch.utils.data.DataLoader(self.distillation_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.merging_test_loader = torch.utils.data.DataLoader(self.merging_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def save_subsets(self,location):
        # Create dataset directories
        split_save_dir = os.path.join(location, 'Subsets', 'MNIST')
        os.makedirs(split_save_dir, exist_ok=True)

        # Save the split datasets
        torch.save(self.finetune_set, os.path.join(split_save_dir, 'finetune_set.pt'))
        torch.save(self.distillation_train_set, os.path.join(split_save_dir, 'distillation_train_set.pt'))
        torch.save(self.merging_train_set, os.path.join(split_save_dir, 'merging_train_set.pt'))
        torch.save(self.distillation_test_set, os.path.join(split_save_dir, 'distillation_test_set.pt'))
        torch.save(self.merging_test_set, os.path.join(split_save_dir, 'merging_test_set.pt'))

        