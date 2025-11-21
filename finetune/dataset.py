"""NYU2 Dataset Loader for MiDaS Fine-tuning"""
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class NYU2Dataset(Dataset):
    """
    NYU2 Depth Estimation Dataset

    Following MiDaS paper's data augmentation strategy:
    - Horizontal flip (50% probability)
    - Random crop and resize to 384x384
    - NO color jittering or other complex augmentations
    """

    def __init__(self, csv_file, img_size=384, do_random_flip=True):
        """
        Args:
            csv_file: CSV file path, format: image_path,depth_path
            img_size: image size (default: 384 as in paper)
            do_random_flip: apply random horizontal flip (default: True)
        """
        self.data = pd.read_csv(csv_file, header=None, names=['image', 'depth'])
        self.img_size = img_size
        self.do_random_flip = do_random_flip

        # CSV directory for resolving relative paths
        self.csv_dir = os.path.dirname(os.path.abspath(csv_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Read paths
        img_path = self.data.iloc[idx]['image']
        depth_path = self.data.iloc[idx]['depth']

        # Resolve relative paths
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.csv_dir, img_path)
        if not os.path.isabs(depth_path):
            depth_path = os.path.join(self.csv_dir, depth_path)

        # Load RGB image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Cannot load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load depth map
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Cannot load depth: {depth_path}")

        # Convert to disparity space (disparity = 1/depth)
        # Following MiDaS paper: all ground truth is converted to disparity
        depth = depth.astype(np.float32)
        depth_valid = depth > 0
        disparity = np.zeros_like(depth)
        disparity[depth_valid] = 1.0 / (depth[depth_valid] + 1e-6)

        # Normalize to [0, 1] (as mentioned in paper)
        if disparity.max() > 0:
            disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)

        # Random horizontal flip (50% probability, as in paper)
        if self.do_random_flip and np.random.random() > 0.5:
            image = np.fliplr(image).copy()
            disparity = np.fliplr(disparity).copy()

        # Resize to 384x384 (paper's input size)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        disparity = cv2.resize(disparity, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        disparity = torch.from_numpy(disparity).float()

        # Normalize image (MiDaS normalization)
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        image = (image - mean) / std

        return {
            'image': image,
            'disparity': disparity,
            'image_path': img_path
        }
