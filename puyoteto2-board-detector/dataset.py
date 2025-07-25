import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import random
from torchvision import transforms


class TetrisCellDataset(Dataset):
    def __init__(self, board_info_path, transform=None, is_train=True, max_crop_shift=5):

        with open(board_info_path, 'r') as f:
            self.samples = json.load(f)

        self.is_train = is_train
        self.max_crop_shift = max_crop_shift
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample['image_path']).convert('RGB')
        x1, y1, x2, y2 = sample['cell_pos']
        
        if self.is_train:
            shift_x = random.randint(-self.max_crop_shift, self.max_crop_shift)
            shift_y = random.randint(-self.max_crop_shift, self.max_crop_shift)

            x1 = max(0, x1 + shift_x)
            x2 = max(x1 + 1, x2 + shift_x)

            y1 = max(0, y1 + shift_y)
            y2 = max(y1 + 1, y2 + shift_y)

        cell_img = image.crop((x1, y1, x2, y2))
        label = sample['label']

        if self.transform:
            cell_img = self.transform(cell_img)

        return cell_img, label
