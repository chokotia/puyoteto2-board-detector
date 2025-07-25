import matplotlib.pyplot as plt
from dataset import TetrisCellDataset
import torch
import torchvision.transforms as transforms
import os

from transform import get_train_transform, get_test_transform


# ラベルIDを文字で確認したい場合
LABEL_NAMES = [
    "E", "I", "O", "T", "L", "J", "S", "Z", "G"
]

dataset = TetrisCellDataset(
    board_info_path='./data/preprocessed_board_info.json',
    transform=get_train_transform(),
)

# 上位200サンプルだけ確認
for i in range(200):
    img, label = dataset[i]
    if isinstance(img, torch.Tensor):
        img = transforms.ToPILImage()(img)

    plt.subplot(8, 25, i + 1)
    plt.imshow(img)
    plt.title(LABEL_NAMES[label] if label < len(LABEL_NAMES) else str(label))
    plt.axis('off')

plt.tight_layout()
plt.show()
