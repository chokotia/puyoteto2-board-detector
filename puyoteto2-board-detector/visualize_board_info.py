import json
import matplotlib.pyplot as plt
from PIL import Image
import os

# ラベルIDを文字で確認したい場合
LABEL_NAMES = [
    "E", "I", "O", "T", "L", "J", "S", "Z", "G"
]

def load_and_visualize_data(json_path, max_samples=200):
    """
    JSONファイルから画像データを読み込み、ラベルと一緒に表示する
    """
    # JSONファイルを読み込み
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 表示するサンプル数を制限
    samples_to_show = min(len(data), max_samples)
    
    # グリッドサイズを計算（8行×25列 = 200個まで表示可能）
    cols = 25
    rows = (samples_to_show + cols - 1) // cols  # 切り上げ計算
    
    # 図のサイズを設定（画像を小さくするため図のサイズを調整）
    plt.figure(figsize=(15, rows * 1.2))
    
    valid_count = 0
    
    for i in range(samples_to_show):
        item = data[i]
        image_path = item['image_path']
        cell_pos = item['cell_pos']
        label = item['label']
        
        # 画像ファイルが存在するかチェック
        if not os.path.exists(image_path):
            print(f"警告: 画像ファイルが見つかりません: {image_path}")
            continue
        
        try:
            # 画像を読み込み
            img = Image.open(image_path)
            
            # cell_posで指定された領域を切り出し
            # cell_pos = [x1, y1, x2, y2]
            cropped_img = img.crop(cell_pos)
            
            # 画像をリサイズして小さくする
            cropped_img = cropped_img.resize((32, 32), Image.Resampling.LANCZOS)
            
            # サブプロットに表示
            plt.subplot(rows, cols, valid_count + 1)
            plt.imshow(cropped_img)
            
            # タイトルにラベル名のみを表示
            label_text = LABEL_NAMES[label] if label < len(LABEL_NAMES) else "?"
            plt.title(label_text, fontsize=10)
            plt.axis('off')
            
            valid_count += 1
            
        except Exception as e:
            print(f"エラー: {image_path} の処理に失敗しました: {e}")
            continue
    
    plt.tight_layout()
    plt.show()
    
    print(f"表示された画像数: {valid_count}/{samples_to_show}")
    print(f"総データ数: {len(data)}")

def main():
    json_path = 'data/preprocessed_board_info.json'
    
    # ファイルの存在確認
    if not os.path.exists(json_path):
        print(f"エラー: JSONファイルが見つかりません: {json_path}")
        return
    
    # データを読み込んで表示
    load_and_visualize_data(json_path, max_samples=200)

if __name__ == "__main__":
    main()
    