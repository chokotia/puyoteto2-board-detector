import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from dataset import TetrisCellDataset
from transform import get_train_transform, get_test_transform
import os
from datetime import datetime

# --- 設定 ---
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 保存ディレクトリ作成 ---
timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
save_dir = os.path.join("models", timestamp)
os.makedirs(save_dir, exist_ok=True)

# --- データ読み込み ---
train_dataset = TetrisCellDataset(
    board_info_path='./data/cell_annotations_train.json',
    transform=get_train_transform()
)
test_dataset = TetrisCellDataset(
    board_info_path='./data/cell_annotations_val.json',
    transform=get_test_transform()
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- モデル構築（MobileNetV3 smallベース） ---
from torchvision.models import MobileNet_V3_Small_Weights

model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, 512),
    nn.Hardswish(),
    nn.Dropout(0.2),
    nn.Linear(512, 9)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- 学習ループ ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # --- 評価 ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    test_acc = 100 * correct / total
    print(f"[Epoch {epoch+1}] Test Accuracy: {test_acc:.2f}%")

    # --- モデル保存 ---
    model_path = os.path.join(save_dir, f"epoch_{epoch+1}_acc_{test_acc:.2f}.pth")
    torch.save(model.state_dict(), model_path)

print(f"\n学習完了！モデルは {save_dir} に保存されました。")
