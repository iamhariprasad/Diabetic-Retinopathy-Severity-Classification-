import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import cv2
import numpy as np

from preprocessing import get_train_transforms, get_val_transforms, normalize_4_channel

class APTOSDataset(Dataset):
    """
    Dataset for APTOS 2019 Blindness Detection (Diabetic Retinopathy).
    Expects a CSV with 'id_code' and 'diagnosis' columns.
    If a segmenter function is provided, this Dataset will return a 4-channel tensor.
    """
    def __init__(self, csv_file, img_dir, segmenter_fn=None, mask_dir=None, transform=None, is_train=True):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.segmenter_fn = segmenter_fn
        self.mask_dir = mask_dir
        
        if transform is None:
            self.transform = get_train_transforms() if is_train else get_val_transforms()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, f"{self.data_frame.iloc[idx, 0]}.png")
        image_bgr = cv2.imread(img_name)
        
        # APTOS 2019 usually contains .png, but maybe .tiff or .jpeg
        if image_bgr is None:
           # fallback to trying .jpeg
           img_name = os.path.join(self.img_dir, f"{self.data_frame.iloc[idx, 0]}.jpeg")
           image_bgr = cv2.imread(img_name)
        
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot find image: {self.data_frame.iloc[idx, 0]}")
            
        label = self.data_frame.iloc[idx, 1]

        # Generate or load vessel mask (1 channel)
        mask_2d = None
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, f"{self.data_frame.iloc[idx, 0]}.png")
            # fallback to trying .jpeg
            if not os.path.exists(mask_path):
                 mask_path = os.path.join(self.mask_dir, f"{self.data_frame.iloc[idx, 0]}.jpeg")
            if os.path.exists(mask_path):
                 mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                 if mask_img is not None:
                      mask_2d = mask_img / 255.0

        if mask_2d is None and self.segmenter_fn is not None:
            # segmenter_fn should return a binary mask (H, W) in range [0, 1] or [0, 255]
            mask_2d = self.segmenter_fn(image_bgr)
            if mask_2d.max() > 1.0:
                mask_2d = mask_2d / 255.0

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # To apply Torchvision transforms, we apply it to the RGB image
        pil_img = Image.fromarray(image_rgb)
        
        if self.transform:
            image_tensor = self.transform(pil_img) # Returns 3xHxW
        else:
            # ToTensor roughly
            image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1))).float() / 255.0

        if mask_2d is not None:
            # Resize the mask to match the transformed image size
            _, h, w = image_tensor.shape
            mask_resized = cv2.resize(mask_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float() # 1xHxW
            
            # Concatenate RGB + Mask to get 4 channels
            fused_tensor = torch.cat([image_tensor, mask_tensor], dim=0) # 4xHxW
            
            # Normalize
            fused_tensor = normalize_4_channel(fused_tensor)
            return fused_tensor, label
            
        return image_tensor, label


class DRIVEDataset(Dataset):
    """
    Dataset for DRIVE retinal vessel segmentation dataset.
    Used for evaluating the segmentation pipeline.
    """
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        # Valid files only
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.tif') or f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.images[idx])
        # Mask names usually have _manual1.gif or similar in DRIVE, we'll try to match name
        base_name = self.images[idx].split('_')[0]
        # Common pattern: 01_test.tif -> 01_manual1.gif
        mask_name = os.path.join(self.mask_dir, f"{base_name}_manual1.gif")
        
        if not os.path.exists(mask_name):
            # Fallback for plain same-name png
            mask_name = os.path.join(self.mask_dir, self.images[idx])

        image_bgr = cv2.imread(img_name)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # read mask with PIL because cv2 might not support gif out of the box
        mask_pil = Image.open(mask_name).convert('L')
        mask = np.array(mask_pil) / 255.0

        if self.transform:
             pil_img = Image.fromarray(image_rgb)
             image_tensor = self.transform(pil_img)
        else:
             image_tensor = torch.from_numpy(image_rgb.transpose((2, 0, 1))).float() / 255.0
             
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor, image_bgr
