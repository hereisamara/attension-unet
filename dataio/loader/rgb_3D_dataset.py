import torch.utils.data as data
import numpy as np

from os import listdir
from os.path import join
import cv2
import datetime
from PIL import Image

def convert_mask_to_color(mask):
    # Define color map
    color_map = np.array([
        [0, 0, 0],       # Class 0: Black
        [255, 0, 0],     # Class 1: Red
        [0, 255, 0],     # Class 2: Green
        [0, 0, 255]      # Class 3: Blue
    ])

    # Convert the 2D mask to a 3D image (RGB)
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for label in range(4):  # Labels are 0, 1, 2, 3
        color_mask[mask == label] = color_map[label]

    return color_mask

def is_image_file(filename):
    """Check if a file is a valid image file."""
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"])

def load_image_stack(image_path):
    """Loads 3D RGB images (as stacks of 2D images) from a given directory."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return img

def load_mask(mask_path):
    """Loads 2D mask image from a given path."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask

def check_exceptions(input, target):
    """Checks for exceptions or invalid data."""
    assert input is not None and target is not None, "Input or target is None"
    assert input.shape[0] == target.shape[0], "Image and target size mismatch"

class RGB3DDataset(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=False):
        super(RGB3DDataset, self).__init__()
        image_dir = join(root_dir, split, 'images')  # 3D RGB images
        mask_dir = join(root_dir, split, 'masks')  # 2D masks

        self.image_filenames = sorted([join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)])
        self.mask_filenames = sorted([join(mask_dir, x) for x in listdir(mask_dir) if is_image_file(x)])
        assert len(self.image_filenames) == len(self.mask_filenames), "Mismatch in number of images and masks"

        # Report the number of images in the dataset
        print(f"Number of {split} images: {len(self.image_filenames)}")

        self.transform = transform
        self.preload_data = preload_data

        if self.preload_data:
            print(f"Preloading the {split} dataset...")
            self.raw_images = [load_image_stack(ii) for ii in self.image_filenames]
            self.raw_masks = [load_mask(ii) for ii in self.mask_filenames]
            print("Loading is done\n")

    def __getitem__(self, index):
        # Set random seed to avoid workers sampling the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # Load images and masks (either from disk or from preloaded data)
        if not self.preload_data:
            input = load_image_stack(self.image_filenames[index])
            target = load_mask(self.mask_filenames[index])
        else:
            input = np.copy(self.raw_images[index])
            target = np.copy(self.raw_masks[index])

        # Print shapes for debugging
        print(f"Image shape: {input.shape}, Mask shape: {target.shape}")
        target = convert_mask_to_color(target)
        # Handle any exceptions or invalid data
        check_exceptions(input, target)

        # Apply transformations if provided
        if self.transform:
            input, target = self.transform(input, target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
