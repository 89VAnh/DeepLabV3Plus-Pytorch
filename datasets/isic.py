import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class ISICDataset(Dataset):
    """
    ISIC Skin Lesion Images Dataset for segmentation tasks.
    Args:
        images_path (str): Path to the directory containing images.
        masks_path (str): Path to the directory containing masks.
        transform (callable, optional): A function/transform that takes in an image
            and its corresponding mask and returns a transformed version.
    """

    def __init__(self, 
                 images_path='/kaggle/input/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input', 
                 masks_path='/kaggle/input/isic2018-challenge-task1-data-segmentation/ISIC2018_Task1_Training_GroundTruth', 
                 transform=None):
        self.images_path = os.path.expanduser(images_path)
        self.masks_path = os.path.expanduser(masks_path)
        self.transform = transform
        self.ids = [image_file[:-4] for image_file in os.listdir(images_path) if image_file.endswith('.jpg')]

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

        # Load image and mask using OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Convert image from BGR to RGB
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert mask to binary (foreground is 1, background is 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = (mask > 0).astype(np.uint8)

        # Apply transformations if provided
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Convert numpy arrays to PyTorch tensors
        img = torch.from_numpy(image).permute(2, 0, 1).float() / 255.  # Normalize image to [0, 1]
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # Add channel dimension

        return img, mask

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