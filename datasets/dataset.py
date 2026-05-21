
import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
    
class MFFDataset(Dataset):
    def __init__(self, root_dir1, root_dir2, transform=None):
        self.source_1_dir = root_dir1
        self.source_2_dir = root_dir2
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # Assume all directories contain the same file names
        self.image_names = os.listdir(self.source_1_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        source_1_path = os.path.join(self.source_1_dir, img_name)
        source_2_path = os.path.join(self.source_2_dir, img_name)

        source_1_image = Image.open(source_1_path).convert('RGB')
        source_2_image = Image.open(source_2_path).convert('RGB')

        if self.transform:
            source_1_image = self.transform(source_1_image)
            source_2_image = self.transform(source_2_image)

        cls = 0
        save_name = img_name
        return source_1_image, source_2_image, cls, save_name

class CTMRIDataset(Dataset):
    def __init__(self, root_dir1, root_dir2, transform=None):
        self.source_1_dir = root_dir1
        self.source_2_dir = root_dir2

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Assume all directories contain the same file names
        self.image_names = os.listdir(self.source_1_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        source_1_path = os.path.join(self.source_1_dir, img_name)
        source_2_path = os.path.join(self.source_2_dir, img_name)

        source_1_image = Image.open(source_1_path).convert('RGB')
        source_2_image = Image.open(source_2_path).convert('RGB')

        if self.transform:
            source_1_image = self.transform(source_1_image)
            source_2_image = self.transform(source_2_image)

        cls = 0
        save_name = img_name
        return source_1_image, source_2_image, cls, save_name


class PETMRIDataset(Dataset):
    def __init__(self, root_dir1, root_dir2, transform=None):
        self.source_1_dir = root_dir1
        self.source_2_dir = root_dir2
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # Assume all directories contain the same file names
        self.image_names = os.listdir(self.source_1_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        source_1_path = os.path.join(self.source_1_dir, img_name)
        source_2_path = os.path.join(self.source_2_dir, img_name)

        source_1_image = Image.open(source_1_path).convert('RGB')
        source_2_image = Image.open(source_2_path).convert('RGB')

        if self.transform:
            source_1_image = self.transform(source_1_image)
            source_2_image = self.transform(source_2_image)

        cls = 0
        save_name = img_name
        return source_1_image, source_2_image, cls, save_name
    
class VIFtestDataset(Dataset):
    def __init__(self, root_dir1, root_dir2, transform=None):
        self.source_1_dir = root_dir1
        self.source_2_dir = root_dir2
        self.transform = transform

        # Assume all directories contain the same file names
        self.image_names = os.listdir(self.source_1_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        source_1_path = os.path.join(self.source_1_dir, img_name)
        source_2_path = os.path.join(self.source_2_dir, img_name)

        source_1_image = Image.open(source_1_path).convert('RGB')
        source_2_image = Image.open(source_2_path).convert('RGB')

        if self.transform:
            source_1_image = self.transform(source_1_image)
            source_2_image = self.transform(source_2_image)

        cls = 0
        save_name = img_name
        return source_1_image, source_2_image, cls, save_name
    

class TNODataset(Dataset):
    def __init__(self, root_dir1, root_dir2, transform=None):
        self.source_1_dir = root_dir1
        self.source_2_dir = root_dir2
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # Assume all directories contain the same file names
        self.image_names = os.listdir(self.source_1_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        source_1_path = os.path.join(self.source_1_dir, img_name)
        source_2_path = os.path.join(self.source_2_dir, img_name)

        source_1_image = Image.open(source_1_path).convert('RGB')
        source_2_image = Image.open(source_2_path).convert('RGB')

        if self.transform:
            source_1_image = self.transform(source_1_image)
            source_2_image = self.transform(source_2_image)

        cls = 0
        save_name = img_name
        return source_1_image, source_2_image, cls, save_name
    
    
class MEFBDataset(Dataset):
    def __init__(self, root_dir1, root_dir2, transform=None):
        self.source_1_dir = root_dir1
        self.source_2_dir = root_dir2
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # Assume all directories contain the same file names
        self.image_names = os.listdir(self.source_1_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        source_1_path = os.path.join(self.source_1_dir, img_name)
        source_2_path = os.path.join(self.source_2_dir, img_name)

        source_1_image = Image.open(source_1_path).convert('RGB')
        source_2_image = Image.open(source_2_path).convert('RGB')

        if self.transform:
            source_1_image = self.transform(source_1_image)
            source_2_image = self.transform(source_2_image)

        cls = 0
        save_name = img_name
        return source_1_image, source_2_image, cls, save_name
    
    
class RoadDataset(Dataset):
    def __init__(self, root_dir1, root_dir2, transform=None):
        self.source_1_dir = root_dir1
        self.source_2_dir = root_dir2
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # Assume all directories contain the same file names
        self.image_names = os.listdir(self.source_1_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        source_1_path = os.path.join(self.source_1_dir, img_name)
        source_2_path = os.path.join(self.source_2_dir, img_name)

        source_1_image = Image.open(source_1_path).convert('RGB')
        source_2_image = Image.open(source_2_path).convert('RGB')

        if self.transform:
            source_1_image = self.transform(source_1_image)
            source_2_image = self.transform(source_2_image)

        cls = 0
        save_name = img_name
        return source_1_image, source_2_image, cls, save_name