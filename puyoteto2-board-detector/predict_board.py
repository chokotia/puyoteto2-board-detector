import torch
from torch import nn, optim
from torchvision import models, transforms
from PIL import Image
import json
import sys
import os
from transform import get_test_transform

# --- 入力指定 ---
image_idx = 0       # board_info.jsonの何番目の画像か
board_idx = 0       # その画像内の何番目の盤面か
model_path = "models/2025-07-25-1551/epoch_2_acc_99.75.pth"  # 学習済みモデル

# --- デバイス ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデルの構築・読み込み ---
model = models.mobilenet_v2(weights=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 9)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# --- board_info 読み込み ---
with open("data/board_info.json", "r") as f:
    board_info = json.load(f)

entry = board_info[image_idx]
board = entry["boards"][board_idx]
image_path = entry["image_path"]
position = board["position"]  # [x1, y1, x2, y2]

# --- 画像読み込み & 切り出し ---
image = Image.open(image_path).convert("RGB")
x1, y1, x2, y2 = position
board_img = image.crop((x1, y1, x2, y2))

cell_w = (x2 - x1) / 10
cell_h = (y2 - y1) / 20

transform = get_test_transform()
predicted_labels = []

# --- セル単位で切り出して予測 ---
for col in range(10):          # 左から右へ
    for row in range(20):      # 各列の上から下へ
        cx1 = int(col * cell_w)
        cy1 = int(row * cell_h)
        cx2 = int((col + 1) * cell_w)
        cy2 = int((row + 1) * cell_h)

        cell_img = board_img.crop((cx1, cy1, cx2, cy2))
        cell_tensor = transform(cell_img).unsqueeze(0).to(device)  # (1, C, H, W)

        with torch.no_grad():
            output = model(cell_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            predicted_labels.append(str(pred_class))


# --- 出力 ---
label_str = ''.join(predicted_labels)
print(f"[Image {image_idx}, Board {board_idx}]")
print(f"Predicted labels ({len(label_str)} chars):\n{label_str}")

# --- 元ラベルの表示（比較用） ---
true_labels = board["labels"]
print(f"Ground Truth:\n{true_labels}")

# --- 一致率表示（任意） ---
match_count = sum(p == t for p, t in zip(label_str, true_labels))
print(f"Match: {match_count}/200 ({match_count / 2:.1f}%)")
