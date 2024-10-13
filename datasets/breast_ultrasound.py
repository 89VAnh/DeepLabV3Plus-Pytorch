import os
from PIL import Image
import torch.utils.data as data
import numpy as np

class BreastUltrasoundDataset(data.Dataset):
    """
    Breast Ultrasound Images Dataset for segmentation task.
    Args:
        root (string): Root directory of the Breast Ultrasound Dataset.
        image_set (string, optional): Select the image_set to use, ``train`` or ``val``.
        transform (callable, optional): A function/transform that takes in an image
            and its corresponding mask and returns a transformed version.
    """
    
    # Define color map for mask decoding (class 0 -> black, class 1 -> white)
    cmap = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)

    def __init__(self, root="/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT", 
                 image_set="train", transform=None):
        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.transform = transform
        # self.types = ["benign", "malignant", "normal"]  # Classes in the dataset
        self.types = ["benign", "malignant"]  # Classes in the dataset
        self.img_path_lst = []

        # Collect all image paths
        for tumor_type in self.types:
            type_image_path = os.path.join(self.root, tumor_type)
            images = [f for f in os.listdir(type_image_path) if 'mask' not in f and f.endswith('.png')]
            self.img_path_lst += [os.path.join(tumor_type, f) for f in images]

        # Split the dataset into 80% train and 20% validation
        split_index = int(len(self.img_path_lst) * 0.8)
        if image_set == "train":
            self.img_path_lst = self.img_path_lst[:split_index]
        else:
            self.img_path_lst = self.img_path_lst[split_index:]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, mask) where mask is the segmentation mask for the image.
        """
        img_relative_path = self.img_path_lst[index]
        tumor_type = img_relative_path.split("/")[0]  # Extract the type (benign, malignant, normal)
        
        # Load image and corresponding mask using PIL
        image_path = os.path.join(self.root, img_relative_path)
        mask_path = image_path.replace(".png", "_mask.png")
        
        image = Image.open(image_path).convert("RGB")  # Load image and convert to RGB
        mask = Image.open(mask_path).convert("L")  # Load mask as grayscale
        
        # Convert mask to binary (foreground is 1, background is 0)
        mask = Image.fromarray((np.array(mask) > 0).astype(np.uint8))

        # Apply transformations if provided
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        return image, mask

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.img_path_lst)

    @classmethod
    def decode_target(cls, mask):
        """
        Decode the segmentation mask into an RGB image.
        Args:
            mask (numpy array): The binary mask where 0 is background and 1 is the foreground.
        Returns:
            RGB image: An RGB image representing the decoded mask.
        """
        return cls.cmap[mask]