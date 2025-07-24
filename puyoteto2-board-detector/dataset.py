import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import random
from torchvision import transforms


class TetrisCellDataset(Dataset):
    def __init__(self, board_info_path,transform=None, is_train=True, max_crop_shift=5):

        with open(board_info_path, 'r') as f:
            raw_info = json.load(f)

        self.samples = []
        for entry in raw_info:
            image_path = entry["image_path"]
            for board in entry["boards"]:
                pos = board["position"]  # (x1, y1, x2, y2)
                labels_str = board["labels"]
                assert len(labels_str) == 200
                labels = [[int(labels_str[col * 20 + row]) for col in range(10)] for row in range(20)]
                self.samples.append({
                    'image_path': image_path,
                    'position': pos,
                    'labels': labels
                })

        self.is_train = is_train
        self.max_crop_shift = max_crop_shift
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.samples) * 200  # 200 cells per board

    def __getitem__(self, idx):
        board_idx = idx // 200
        cell_idx = idx % 200
        sample = self.samples[board_idx]

        image = Image.open(sample['image_path']).convert('RGB')
        x1, y1, x2, y2 = sample['position']
        board_img = image.crop((x1, y1, x2, y2))

        cell_w = (x2 - x1) / 10
        cell_h = (y2 - y1) / 20

        col = cell_idx // 20  # 左から右（縦20マスある）
        row = cell_idx % 20   # 上から下

        cx1 = int(col * cell_w)
        cy1 = int(row * cell_h)
        cx2 = int((col + 1) * cell_w)
        cy2 = int((row + 1) * cell_h)

        if self.is_train:
            shift_x = random.randint(-self.max_crop_shift, self.max_crop_shift)
            shift_y = random.randint(-self.max_crop_shift, self.max_crop_shift)

            cx1 = max(0, cx1 + shift_x)
            cx2 = max(cx1 + 1, cx2 + shift_x)

            cy1 = max(0, cy1 + shift_y)
            cy2 = max(cy1 + 1, cy2 + shift_y)

        cell_img = board_img.crop((cx1, cy1, cx2, cy2))
        label = sample['labels'][row][col]

        if self.transform:
            cell_img = self.transform(cell_img)

        return cell_img, label
