"""
board_annotations.json を前処理して、セル単位のフラットなJSONに変換するスクリプト
"""

import json
import argparse
from pathlib import Path


# ラベルIDを文字で確認したい場合
LABEL_NAMES = [
    "E", "I", "O", "T", "L", "J", "S", "Z", "G"
]


def preprocess_board_annotations(input_path, output_path):
    """
    board_annotations.jsonを前処理してセル単位のフラットなデータに変換
    
    Args:
        input_path: 元のboard_annotations.jsonのパス
        output_path: 出力先のJSONファイルパス
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_info = json.load(f)
    
    cell_data = []
    
    # 各ビデオを処理
    for video_entry in raw_info:
        video_id = video_entry["video_id"]
        
        # 各フレームを処理
        for frame in video_entry["frames"]:
            image_file = frame["image_file"]
            
            # image_pathを構築
            image_path = f"data/img/{video_id}/{image_file}"
            
            # 各盤面を処理
            for board_idx, board in enumerate(frame["boards"]):
                board_pos = board["position"]  # [x1, y1, x2, y2]
                labels_str = board["labels"]
                
                # labelsの文字列が200文字であることを確認
                assert len(labels_str) == 200, f"Labels length should be 200, got {len(labels_str)} for {video_id}/{image_file}/board{board_idx}"
                
                # 盤面の幅と高さを計算
                x1, y1, x2, y2 = board_pos
                board_width = x2 - x1
                board_height = y2 - y1
                cell_width = board_width / 10
                cell_height = board_height / 20
                
                # 各セルの情報を作成
                for idx in range(200):
                    col = idx // 20  # 列（0-9）
                    row = idx % 20   # 行（0-19）
                    
                    # セルの座標を計算（盤面内での相対座標）
                    cell_x1 = int(col * cell_width)
                    cell_y1 = int(row * cell_height)
                    cell_x2 = int((col + 1) * cell_width)
                    cell_y2 = int((row + 1) * cell_height)
                    
                    # セルの絶対座標（画像全体での座標）
                    abs_cell_pos = [
                        x1 + cell_x1,  # 絶対x1
                        y1 + cell_y1,  # 絶対y1
                        x1 + cell_x2,  # 絶対x2
                        y1 + cell_y2   # 絶対y2
                    ]
                    
                    # ラベル（該当する文字）
                    label = int(labels_str[idx])
                    
                    # ラベル名（LABEL_NAMESから対応する文字を取得）
                    label_name = LABEL_NAMES[label] if 0 <= label < len(LABEL_NAMES) else "UNKNOWN"
                    
                    cell_info = {
                        "image_path": image_path,
                        "board_id": board_idx,
                        "board_pos": board_pos,
                        "cell_pos": abs_cell_pos,
                        "cell_index": {"col": col, "row": row},
                        "label": label,
                        "label_name": label_name
                    }
                    
                    cell_data.append(cell_info)
    
    # 結果を保存（コンパクトな形式で）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cell_data, f, indent=2, separators=(',', ': '), ensure_ascii=False)
    
    # さらにコンパクトにするため、配列部分の改行を除去
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 短い配列（座標など）の改行を除去
    import re
    # 4要素以下の数値配列の改行を除去
    content = re.sub(r'\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]', r'[\1, \2, \3, \4]', content)
    # 2要素の数値配列の改行を除去  
    content = re.sub(r'\[\s*(\d+),\s*(\d+)\s*\]', r'[\1, \2]', content)
    # オブジェクト内の改行も調整
    content = re.sub(r'{\s*"col":\s*(\d+),\s*"row":\s*(\d+)\s*}', r'{"col": \1, "row": \2}', content)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"処理完了: {len(cell_data)}個のセルデータを生成")
    print(f"出力ファイル: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="board_annotations.jsonを前処理してセル単位のデータに変換")
    parser.add_argument("input", help="入力のboard_annotations.jsonファイル")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        return
    
    # 出力ファイルは入力ファイルと同じ階層にcell_annotations.jsonとして作成
    output_path = input_path.parent / "cell_annotations.json"
    
    preprocess_board_annotations(input_path, output_path)


if __name__ == "__main__":
    main()
