import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class COVIDQUExDataset(Dataset):
    """
    COVIDQUExDataset for segmentation tasks.
    Args:
        root_dir (str): Directory with all the images.
        phase (str): 'Train', 'Val', or 'Test'.
        image_type (str): 'COVID-19', 'Non-COVID', or 'all'.
        image_set (str): 'train' or 'test' to specify the subset of the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    
    def __init__(self, root_dir = '/kaggle/input/Infection Segmentation Data', phase='train', image_type='all', transform=None):
        self.root_dir = root_dir
        self.image_type = image_type
        self.phase = phase.capitalize()  # 'train', 'val', or 'test'
        self.transform = transform

        self.data = []
        
        # If 'all' is selected, include both 'COVID-19' and 'Non-COVID' types
        if image_type == 'all':
            image_types = ['COVID-19', 'Non-COVID']
        else:
            image_types = [image_type]
        
        # Collect image and mask paths for the specified types and phase
        for img_type in image_types:
            img_dir = os.path.join(root_dir, 'Infection Segmentation Data', self.phase, img_type, 'images')
            mask_dir = os.path.join(root_dir, 'Infection Segmentation Data', self.phase, img_type, 'infection masks')
            
            for img_name in os.listdir(img_dir):
                if img_name.endswith('.png'):
                    img_path = os.path.join(img_dir, img_name)
                    mask_path = os.path.join(mask_dir, img_name)
                    self.data.append({
                        'image': img_path,
                        'mask': mask_path,
                        'type': img_type
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data sample to fetch.
        Returns:
            tuple: (image, mask) where image is the image data and mask is the corresponding segmentation mask.
        """
        img_dict = self.data[idx]

        # Load image using PIL
        image = Image.open(img_dict['image']).convert('RGB')

        # Load infection mask
        mask = Image.open(img_dict['mask']).convert('L')  # Load in grayscale

        # Apply transforms if any
        if self.transform:
            image, mask = self.transform(image, mask)
        
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
