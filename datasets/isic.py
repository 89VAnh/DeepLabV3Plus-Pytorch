import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class ISICDataset(Dataset):
    """
    ISIC Skin Lesion Images Dataset for segmentation tasks.
    Args:
        images_path (str): Path to the directory containing images.
        masks_path (str): Path to the directory containing masks.
        transform (callable, optional): A function/transform that takes in an image
            and its corresponding mask and returns a transformed version.
        image_set (str): Whether to use 'train', 'val', or 'test' set. Defaults to 'train'.
    """

    def __init__(self, 
                 images_path='/kaggle/input/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input', 
                 masks_path='/kaggle/input/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1_Training_GroundTruth', 
                 transform=None,
                 image_set='train'):
        self.images_path = os.path.expanduser(images_path)
        self.masks_path = os.path.expanduser(masks_path)
        self.transform = transform
        self.image_set = image_set
        self.ids = [image_file[:-4] for image_file in os.listdir(images_path) if image_file.endswith('.jpg')]
        
        # Filter based on image set (you can adjust the filtering logic as per your directory structure)
        if self.image_set == 'train':
            self.ids = self.ids[:int(0.7 * len(self.ids))]  # 70% for training
        elif self.image_set == 'val':
            self.ids = self.ids[int(0.7 * len(self.ids)):int(0.9 * len(self.ids))]  # 20% for validation
        elif self.image_set == 'test':
            self.ids = self.ids[int(0.9 * len(self.ids)):]  # 10% for testing
        else:
            raise ValueError(f"Unknown image_set: {self.image_set}. Use 'train', 'val', or 'test'.")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, mask) where mask is the segmentation mask for the image.
        """
        # Get the image and mask file paths
        image_file = self.ids[index] + '.jpg'
        mask_file = self.ids[index] + '_segmentation.png'
        image_path = os.path.join(self.images_path, image_file)
        mask_path = os.path.join(self.masks_path, mask_file)

        # Load image and mask using PIL
        image = Image.open(image_path).convert('RGB')  # Load image and convert to RGB
        mask = Image.open(mask_path).convert('L')  # Load mask and convert to grayscale

        # Convert mask to binary (foreground is 1, background is 0)
        mask = np.array(mask)
        mask = (mask > 127).astype(np.uint8)

        # Apply transformations if provided
        if self.transform is not None:
            image, mask = self.transform(image, Image.fromarray(mask))
            
        return image, mask

    @staticmethod
    def decode_target(mask):
        """
        Decode the binary mask into an RGB image.
        Args:
            mask (numpy array): The binary mask where 0 is background and 1 is the foreground.
        Returns:
            RGB image: An RGB image representing the decoded mask.
        """
        cmap = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
        return cmap[mask]