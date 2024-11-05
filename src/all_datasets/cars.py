import os
import torch
import pathlib
import deeplake
import numpy as np
from PIL import Image
import scipy.io as sio
import torchvision.datasets as datasets
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import random_split, DataLoader, ConcatDataset, Dataset
from torchvision.datasets.utils import download_and_extract_archive, download_url, verify_str_arg


class PytorchStanfordCars(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        # Skip download if dataset files are already available
        if download and not self._check_exists():
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can set download=True to download it.")

        # Load dataset
        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        # Check if all necessary files and directories are present
        return (
            self._annotations_mat_path.exists() and
            self._images_base_path.is_dir()
        )
    
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target


    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )

class Cars:
    def __init__(self, preprocess, location="/mnt/lts4/scratch/data/stanford_cars", batch_size=64, num_workers=16):
        # Data loading code

        train_dataset = PytorchStanfordCars(location, 'train', preprocess)
        test_dataset = PytorchStanfordCars(location, 'test', preprocess)

        # Split the train dataset into three subsets
        finetune_train_size = int(0.4 * len(train_dataset))
        distillation_train_size = int(0.4 * len(train_dataset))
        merging_train_size = len(train_dataset) - finetune_train_size - distillation_train_size
        finetune_train_set, self.distillation_train_set, self.merging_train_set = random_split(
            train_dataset, [finetune_train_size, distillation_train_size, merging_train_size])
        
        # Split the test dataset into three subsets
        finetune_test_size = int(0.4 * len(test_dataset))
        distillation_test_size = int(0.4 * len(test_dataset))
        merging_test_size = len(test_dataset) - finetune_test_size - distillation_test_size
        finetune_test_set, self.distillation_test_set, self.merging_test_set = random_split(
            test_dataset, [finetune_test_size, distillation_test_size, merging_test_size])
        
        self.finetune_set = ConcatDataset([finetune_train_set, finetune_test_set])
        
        self.train_loader = DataLoader(self.finetune_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.distillation_train_loader = DataLoader(self.distillation_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.merging_train_loader = DataLoader(self.merging_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.distillation_test_loader = DataLoader(self.distillation_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.merging_test_loader = DataLoader(self.merging_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        idx_to_class = dict((v, k) for k, v in train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]

    def save_subsets(self,location):
        # Create dataset directories
        split_save_dir = os.path.join(location, 'Subsets', 'Cars')
        os.makedirs(split_save_dir, exist_ok=True)

        # Save the split datasets
        torch.save(self.finetune_set, os.path.join(split_save_dir, 'finetune_set.pt'))
        torch.save(self.distillation_train_set, os.path.join(split_save_dir, 'distillation_train_set.pt'))
        torch.save(self.merging_train_set, os.path.join(split_save_dir, 'merging_train_set.pt'))
        torch.save(self.distillation_test_set, os.path.join(split_save_dir, 'distillation_test_set.pt'))
        torch.save(self.merging_test_set, os.path.join(split_save_dir, 'merging_test_set.pt'))