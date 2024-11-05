import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from typing import Callable, Optional

class RESISC45Dataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Traverse the directory to gather all image paths and labels
        for class_folder in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_folder)
            if os.path.isdir(class_path):
                # Assign an index to each class
                class_index = len(self.class_to_idx)
                self.class_to_idx[class_folder] = class_index

                # Collect all image paths in the class folder
                for image_name in os.listdir(class_path):
                    if image_name.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(class_path, image_name)
                        self.samples.append((image_path, class_index))

        # Store class names for future use
        self.classes = list(self.class_to_idx.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label

class RESISC45:
    def __init__(self, preprocess: Callable, location: str = os.path.expanduser('~/data'), batch_size: int = 32, num_workers: int = 16):
        # Load the full dataset
        full_dataset = RESISC45Dataset( '/mnt/lts4/scratch/data/resisc45/NWPU-RESISC45', transform=preprocess)

        # Split the dataset into train (80%) and test (20%)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # Further split the train dataset into finetune (40%), distillation (40%), and merging (20%) subsets
        finetune_train_size = int(0.04 * len(train_dataset))
        distillation_train_size = int(0.04 * len(train_dataset))
        merging_train_size = len(train_dataset) - finetune_train_size - distillation_train_size
        finetune_train_set, self.distillation_train_set, self.merging_train_set = random_split(
            train_dataset, [finetune_train_size, distillation_train_size, merging_train_size])

        # Further split the test dataset into finetune (40%), distillation (40%), and merging (20%) subsets
        finetune_test_size = int(0.04 * len(test_dataset))
        distillation_test_size = int(0.04 * len(test_dataset))
        merging_test_size = len(test_dataset) - finetune_test_size - distillation_test_size
        finetune_test_set, self.distillation_test_set, self.merging_test_set = random_split(
            test_dataset, [finetune_test_size, distillation_test_size, merging_test_size])
        
        # Combine the finetune train and test sets
        self.finetune_set = ConcatDataset([finetune_train_set, finetune_test_set])
        
        # Initialize DataLoaders for each subset
        self.train_loader = DataLoader(self.finetune_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.distillation_train_loader = DataLoader(self.distillation_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.merging_train_loader = DataLoader(self.merging_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.distillation_test_loader = DataLoader(self.distillation_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.merging_test_loader = DataLoader(self.merging_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # Class names from dataset
        self.classnames = [c.replace('_', ' ') for c in train_dataset.dataset.classes]

    def save_subsets(self, location):
        # Save subsets to specified location
        split_save_dir = os.path.join(location, 'Subsets', 'RESISC45')
        os.makedirs(split_save_dir, exist_ok=True)

        torch.save(self.finetune_set, os.path.join(split_save_dir, 'finetune_set.pt'))
        torch.save(self.distillation_train_set, os.path.join(split_save_dir, 'distillation_train_set.pt'))
        torch.save(self.merging_train_set, os.path.join(split_save_dir, 'merging_train_set.pt'))
        torch.save(self.distillation_test_set, os.path.join(split_save_dir, 'distillation_test_set.pt'))
        torch.save(self.merging_test_set, os.path.join(split_save_dir, 'merging_test_set.pt'))