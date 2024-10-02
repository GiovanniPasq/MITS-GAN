import os
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ScanDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".mhd")]
        self.transform = transforms.Compose([
                            transforms.Normalize(mean=[0.5], std=[0.5]), 
                           ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mhd_file = os.path.join(self.data_dir, self.file_list[idx])
        image = self.load_mhd(mhd_file)
        image = torch.tensor((image - image.min()) / (image.max() - image.min()), dtype=torch.double)
        image = self.transform(image)
        return image, mhd_file

    def load_mhd(self, mhd_file):
        image = sitk.ReadImage(mhd_file)
        image_array = sitk.GetArrayFromImage(image)
        return image_array
