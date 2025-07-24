import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from dataset import TetrisCellDataset
from transform import get_train_transform, get_test_transform
import os
from datetime import datetime

# --- 設定 ---
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 保存ディレクトリ作成 ---
timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')
save_dir = os.path.join("models", timestamp)
os.makedirs(save_dir, exist_ok=True)

# --- データ読み込み & 分割 ---
full_dataset = TetrisCellDataset(
    board_info_path='./data/board_info.json',
    transform=get_train_transform()
)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# テスト用データはtransform差し替え
test_dataset.dataset.transform = get_test_transform()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- モデル構築（ResNet18ベース） ---
model = models.resnet18(weights=True)
model.fc = nn.Linear(model.fc.in_features, 9)
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
    model_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)
