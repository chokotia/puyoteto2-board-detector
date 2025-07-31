import argparse
import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataset import TetrisCellDataset
from transform import get_visualize_dataset_transform

# ラベルIDを文字で確認したい場合
LABEL_NAMES = [
    "E", "I", "O", "T", "L", "J", "S", "Z", "G"
]

def visualize_dataset(json_path, output_dir, max_images=1000):
    """データセットの画像を可視化して保存"""
    
    # 出力ディレクトリを作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"JSONファイル: {json_path}")
    print(f"出力ディレクトリ: {output_path}")
    print(f"最大保存画像数: {max_images}")
    
    try:
        # データセットとデータセットを作成（可視化専用transform使用）
        visualize_dataset = TetrisCellDataset(
            board_info_path=json_path,
            transform=get_visualize_dataset_transform(),
            # is_train=True, 
            # max_crop_shift=5,
            expansion_factor=2,
            padding_color=(0, 0, 0)
        )                 
                
        dataset_size = len(visualize_dataset)
        print(f"データセットサイズ: {len(visualize_dataset)}")
        
        for idx in range(dataset_size):
            if idx >= max_images:
                break
            
            # 画像とラベルを取得
            pil_image, label_value = visualize_dataset.__getitem__(idx)

            try:
                # ファイル名を作成
                filename = f"img_{idx:05d}_label_{LABEL_NAMES[label_value]}.png"
                filepath = output_path / filename
                
                # 画像を直接保存（変換不要）
                pil_image.save(filepath)
                
                if (idx+1) % 100 == 0:
                    print(f"進捗: {idx+1}/{max_images} 画像を保存済み")
                
            except Exception as e:
                print(f"画像 {idx} の保存でエラー: {e}")
                # デバッグ用に詳細情報を出力
                print(f"  画像タイプ: {type(pil_image)}")
                if hasattr(pil_image, 'size'):
                    print(f"  画像サイズ: {pil_image.size}")
        
        print(f"\n完了! 合計 {idx+1} 枚の画像を保存しました。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='訓練用データセットクラスの画像を可視化')
    parser.add_argument('json_path', help='JSONファイルのパス (cell_annotations.json)')
    parser.add_argument('--output-dir', default='output/train_dataset_images', 
                       help='出力ディレクトリ (デフォルト: output/train_dataset_images)')
    parser.add_argument('--max-images', type=int, default=200,
                       help='保存する最大画像数 (デフォルト: 200)')
    
    args = parser.parse_args()
    
    print("=== 訓練用データセット画像可視化 ===")
    
    # 可視化実行
    success = visualize_dataset(args.json_path, args.output_dir, args.max_images)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()