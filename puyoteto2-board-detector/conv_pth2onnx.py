# conv_pth2onnx.py
# カスタムMobileNet v2（テトリス盤面分類）をPyTorchからONNX形式に変換するスクリプト

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- モデル構造の定義（predict_board.pyと同じ） ---
def create_model():
    """カスタムMobileNet v3 smallモデルを作成"""
    model = models.mobilenet_v3_small(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 512),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(512, 9)
    )
    return model

# --- 設定 ---
model_path = "models/2025-07-31-1942/epoch_7_acc_99.88.pth"  # 学習済みモデルパス
output_path = "tetris_mobilenet_v3_small.onnx"  # 出力ONNXファイル名
# Webアプリ用にはCPU固定が推奨（ONNX.jsとの互換性向上）
device = torch.device("cpu")

# --- モデルの構築・読み込み ---
print("モデルを読み込み中...")
model = create_model()
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"モデル読み込み完了: {model_path}")

# --- ONNX変換 ---
print("ONNX形式に変換中...")
with torch.no_grad():
    # ダミー入力（MobileNet v2の標準入力サイズ）
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # ONNX形式でエクスポート（ONNX.js互換性重視）
    torch.onnx.export(
        model,                          # モデル
        dummy_input,                    # ダミー入力
        output_path,                    # 出力ファイル名
        export_params=True,             # パラメータも含める
        opset_version=14,               # v3_smallは9では動作しなかったため14に上げる
        do_constant_folding=True,       # 定数畳み込み最適化
        input_names=['input'],          # 入力名
        output_names=['output'],        # 出力名
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        # ONNX.js互換性向上のための追加オプション
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL
    )

print(f"変換完了: {output_path}")

# --- テスト用の前処理関数（predict_board.pyと同じ） ---
def get_test_transform():
    """テスト用の前処理（元のtransform.pyと同じ）"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

def preprocess_image_for_web(image_path):
    """Web用の画像前処理（正規化なし版）"""
    transform = get_test_transform()
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).numpy()

# --- PyTorchモデルのテスト ---
def test_pytorch_model(image_path):
    """PyTorchモデルでテスト推論"""
    print(f"\nPyTorchモデルのテスト: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"テスト画像が見つかりません: {image_path}")
        return
    
    transform = get_test_transform()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
    print(f"予測クラス: {predicted_class}")
    print(f"信頼度: {confidence:.3f}")
    print(f"全クラス確率: {probabilities[0].tolist()}")

# --- 使用例 ---
if __name__ == "__main__":
    import os
    
    # テスト画像でのテスト（オプション）
    test_image_path = "test_cell.png"  # テスト用のセル画像があれば
    if os.path.exists(test_image_path):
        test_pytorch_model(test_image_path)
    else:
        print(f"\nテスト画像 {test_image_path} が見つかりません")
        print("テスト画像がある場合は、パスを修正して実行してください")
    
    print(f"\n✅ 変換完了！")
    print(f"📁 出力ファイル: {output_path}")
    print(f"🌐 Webアプリで使用可能です")
    
    # クラスラベルの説明
    print(f"\n📊 モデル情報:")
    print(f"   - 入力サイズ: 224x224 RGB")
    print(f"   - 出力クラス数: 9")
    print(f"   - クラス: テトリスのセル状態 (0-8)")
    print(f"   - 前処理: Resize + ToTensor のみ（正規化なし）")
