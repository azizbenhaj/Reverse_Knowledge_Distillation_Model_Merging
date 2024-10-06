import os
import torch
import torchvision.datasets as datasets
import deeplake
from torchvision.datasets import VisionDataset
from PIL import Image
from typing import Any, Callable, Optional, Tuple
import pathlib

from torchvision.datasets.utils import download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import random_split


class PytorchStanfordCars(VisionDataset):
    """Stanford Cars dataset using DeepLake.
    
    Args:
        root (string): Root directory of dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = split
        self.dataset_path = f"hub://activeloop/stanford-cars-{self._split}"
        
        # Load the dataset from DeepLake
        self.ds = deeplake.load(self.dataset_path)

        # Check if the dataset was loaded successfully
        if not self.ds:
            raise RuntimeError(f"Failed to load the {split} dataset from DeepLake")

        # Number of samples in the dataset
        self._num_samples = len(self.ds)

        # Extract class names and create class_to_idx
        self.classes = self.ds['car_models'].info.class_names
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns PIL image and class_id for a given index."""

        # Access the image and label from DeepLake
        image = self.ds['images'][idx].numpy()  # Get image as NumPy array
        label = self.ds['car_models'][idx].numpy()  # Get label as NumPy array

        # Convert the NumPy array to a PIL image
        pil_image = Image.fromarray(image)

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return pil_image, label

class Cars:
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), batch_size=64, num_workers=16):
        # Data loading code

        train_dataset = PytorchStanfordCars(location, 'train', preprocess)
        test_dataset = PytorchStanfordCars(location, 'test', preprocess)

        # Split the train dataset into three subsets
        finetune_train_size = int(0.4 * len(train_dataset)) #0.4
        distillation_train_size = int(0.4 * len(train_dataset))
        merging_train_size = len(train_dataset) - finetune_train_size - distillation_train_size
        self.finetune_train_set, self.distillation_train_set, self.merging_train_set = random_split(
            train_dataset, [finetune_train_size, distillation_train_size, merging_train_size])
        
        # Split the test dataset into three subsets
        finetune_test_size = int(0.4 * len(test_dataset)) #0.4 
        distillation_test_size = int(0.4 * len(test_dataset))
        merging_test_size = len(test_dataset) - finetune_test_size - distillation_test_size
        self.finetune_test_set, self.distillation_test_set, self.merging_test_set = random_split(
            test_dataset, [finetune_test_size, distillation_test_size, merging_test_size])
        
        self.train_loader = torch.utils.data.DataLoader(self.finetune_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.distillation_train_loader = torch.utils.data.DataLoader(self.distillation_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.merging_train_loader = torch.utils.data.DataLoader(self.merging_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.finetune_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.distillation_test_loader = torch.utils.data.DataLoader(self.distillation_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.merging_test_loader = torch.utils.data.DataLoader(self.merging_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        idx_to_class = dict((v, k) for k, v in train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]

    def save_subsets(self,location):
        # Create dataset directories
        split_save_dir = os.path.join(location, 'Subsets', 'Cars')
        os.makedirs(split_save_dir, exist_ok=True)

        # Save the split train datasets
        torch.save(self.finetune_train_set, os.path.join(split_save_dir, 'finetune_train_set.pt'))
        torch.save(self.distillation_train_set, os.path.join(split_save_dir, 'distillation_train_set.pt'))
        torch.save(self.merging_train_set, os.path.join(split_save_dir, 'merging_train_set.pt'))

        # Save the split test datasets
        torch.save(self.finetune_test_set, os.path.join(split_save_dir, 'finetune_test_set.pt'))
        torch.save(self.distillation_test_set, os.path.join(split_save_dir, 'distillation_test_set.pt'))
        torch.save(self.merging_test_set, os.path.join(split_save_dir, 'merging_test_set.pt'))