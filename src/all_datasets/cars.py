import os
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torch.utils.data import random_split, DataLoader, ConcatDataset
from typing import Callable, Optional

class StanfordCarsDataset(VisionDataset):
    def __init__(self, root_dir: str, split: str = "train", transform: Optional[Callable] = None):
        super().__init__(root_dir, transform=transform)
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Load the class labels from the devkit
        self.load_class_labels()

        # Load image paths and their associated class index
        data_dir = os.path.join(self.root_dir, f'cars_{self.split}')
        for image_name in os.listdir(data_dir):
            if image_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(data_dir, image_name)
                label = self.get_class_idx_from_name(image_name)
                self.samples.append((image_path, label))

    def load_class_labels(self):
        # Load class names and create class_to_idx mapping from the devkit
        class_labels_file = os.path.join(self.root_dir, 'devkit', 'cars_meta.mat')
        import scipy.io
        cars_meta = scipy.io.loadmat(class_labels_file)
        class_names = [c[0] for c in cars_meta['class_names'][0]]
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.classes = list(self.class_to_idx.keys())

    def get_class_idx_from_name(self, image_name):
        # Map image name to class index using a lookup
        # (Assumes you have a way to determine class from image name or use annotations file)
        # Replace this with the appropriate label loading for your dataset
        # For example, you might use a .mat file or CSV to map image_name -> label index
        return 0  # Placeholder; replace with actual label extraction logic

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class Cars:
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), batch_size=64, num_workers=16):
        # Load the train and test datasets
        train_dataset = StanfordCarsDataset('/mnt/lts4/scratch/data/stanford_cars', 'train', transform=preprocess)
        test_dataset = StanfordCarsDataset('/mnt/lts4/scratch/data/stanford_cars', 'test', transform=preprocess)

        # Split train and test datasets into subsets
        finetune_train_size = int(0.4 * len(train_dataset))
        distillation_train_size = int(0.4 * len(train_dataset))
        merging_train_size = len(train_dataset) - finetune_train_size - distillation_train_size
        finetune_train_set, self.distillation_train_set, self.merging_train_set = random_split(
            train_dataset, [finetune_train_size, distillation_train_size, merging_train_size])
        
        finetune_test_size = int(0.4 * len(test_dataset))
        distillation_test_size = int(0.4 * len(test_dataset))
        merging_test_size = len(test_dataset) - finetune_test_size - distillation_test_size
        finetune_test_set, self.distillation_test_set, self.merging_test_set = random_split(
            test_dataset, [finetune_test_size, distillation_test_size, merging_test_size])
        
        self.finetune_set = ConcatDataset([finetune_train_set, finetune_test_set])

        # DataLoaders
        self.train_loader = DataLoader(self.finetune_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.distillation_train_loader = DataLoader(self.distillation_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.merging_train_loader = DataLoader(self.merging_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.distillation_test_loader = DataLoader(self.distillation_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.merging_test_loader = DataLoader(self.merging_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.classnames = train_dataset.classes

    def save_subsets(self, location):
        split_save_dir = os.path.join(location, 'Subsets', 'Cars')
        os.makedirs(split_save_dir, exist_ok=True)

        torch.save(self.finetune_set, os.path.join(split_save_dir, 'finetune_set.pt'))
        torch.save(self.distillation_train_set, os.path.join(split_save_dir, 'distillation_train_set.pt'))
        torch.save(self.merging_train_set, os.path.join(split_save_dir, 'merging_train_set.pt'))
        torch.save(self.distillation_test_set, os.path.join(split_save_dir, 'distillation_test_set.pt'))
        torch.save(self.merging_test_set, os.path.join(split_save_dir, 'merging_test_set.pt'))