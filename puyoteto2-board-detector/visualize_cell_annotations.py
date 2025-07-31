import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse

class CellAnnotationValidator:
    def __init__(self, json_path, output_dir="output_cells"):
        """
        Cell Annotation検証クラス
        
        Args:
            json_path (str): JSONファイルのパス
            output_dir (str): 出力ディレクトリ
        """
        self.json_path = json_path
        self.output_dir = Path(output_dir)
        self.data = None
        
        # 出力ディレクトリを作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_json(self):
        """JSONファイルを読み込む"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✓ JSONファイルを正常に読み込みました: {self.json_path}")
            print(f"  総cell数: {len(self.data)}")
            return True
        except FileNotFoundError:
            print(f"✗ JSONファイルが見つかりません: {self.json_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ JSONの形式が正しくありません: {e}")
            return False
        except Exception as e:
            print(f"✗ JSONの読み込み中にエラーが発生しました: {e}")
            return False
    
    def validate_json_structure(self):
        """JSONの構造を検証する"""
        if not isinstance(self.data, list):
            print("✗ JSONのルートは配列である必要があります")
            return False
        
        required_keys = ["image_path", "board_id", "board_pos", "cell_pos", "cell_index", "label", "label_name"]
        
        for i, cell in enumerate(self.data):
            # 必須キーの存在確認
            for key in required_keys:
                if key not in cell:
                    print(f"✗ cell[{i}]に必須キー '{key}' がありません")
                    return False
            
            # データ型の検証
            if not isinstance(cell["board_pos"], list) or len(cell["board_pos"]) != 4:
                print(f"✗ cell[{i}].board_posは4つの数値の配列である必要があります")
                return False
            
            if not isinstance(cell["cell_pos"], list) or len(cell["cell_pos"]) != 4:
                print(f"✗ cell[{i}].cell_posは4つの数値の配列である必要があります")
                return False
            
            if not isinstance(cell["cell_index"], dict):
                print(f"✗ cell[{i}].cell_indexは辞書である必要があります")
                return False
            
            if "col" not in cell["cell_index"] or "row" not in cell["cell_index"]:
                print(f"✗ cell[{i}].cell_indexには'col'と'row'が必要です")
                return False
            
            if not isinstance(cell["board_id"], int):
                print(f"✗ cell[{i}].board_idは整数である必要があります")
                return False
        
        print("✓ JSON構造の検証が完了しました")
        return True
    
    def get_statistics(self):
        """データの統計情報を表示する"""
        if not self.data:
            return
        
        # 画像パス別の集計
        image_counts = {}
        board_counts = {}
        label_counts = {}
        
        for cell in self.data:
            image_path = cell["image_path"]
            board_id = cell["board_id"]
            label_name = cell["label_name"]
            
            image_counts[image_path] = image_counts.get(image_path, 0) + 1
            board_counts[board_id] = board_counts.get(board_id, 0) + 1
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
        print("\n=== データ統計 ===")
        print(f"総cell数: {len(self.data)}")
        print(f"画像数: {len(image_counts)}")
        print(f"board_id数: {len(board_counts)}")
        print(f"ラベル種類数: {len(label_counts)}")
        
        print(f"\nboard_id別分布: {dict(sorted(board_counts.items()))}")
        print(f"ラベル分布: {dict(sorted(label_counts.items()))}")
    
    def check_image_files(self):
        """画像ファイルの存在を確認する"""
        unique_images = set(cell["image_path"] for cell in self.data)
        existing_images = []
        missing_images = []
        
        for image_path in unique_images:
            if os.path.exists(image_path):
                existing_images.append(image_path)
            else:
                missing_images.append(image_path)
        
        if missing_images:
            print(f"\n✗ {len(missing_images)} 個の画像ファイルが見つかりません:")
            for img in missing_images[:5]:  # 最初の5個だけ表示
                print(f"   - {img}")
            if len(missing_images) > 5:
                print(f"   ... 他 {len(missing_images) - 5} 個")
        
        print(f"✓ {len(existing_images)} 個の画像ファイルが存在します")
        return existing_images, missing_images
    
    def extract_video_id_and_image_name(self, image_path):
        """画像パスからvideo_idと画像名を抽出する"""
        # data/img/hSMUJcMYNfg/1.png -> video_id: hSMUJcMYNfg, image_name: 1
        path_parts = Path(image_path).parts
        if len(path_parts) >= 3:
            video_id = path_parts[-2]  # ディレクトリ名
            image_name = Path(path_parts[-1]).stem  # 拡張子を除いたファイル名
            return video_id, image_name
        return "unknown", "unknown"
    
    def process_cell(self, cell_data):
        """単一のcellを処理する"""
        try:
            image_path = cell_data["image_path"]
            board_id = cell_data["board_id"]
            cell_pos = cell_data["cell_pos"]
            cell_index = cell_data["cell_index"]
            label_name = cell_data["label_name"]
            
            # 画像パスから情報を抽出
            video_id, image_name = self.extract_video_id_and_image_name(image_path)
            
            # 出力ファイル名を生成
            row = cell_index["row"]
            col = cell_index["col"]
            output_filename = f"cell_{image_name}_b{board_id}_r{row:02d}_c{col:02d}_{label_name}.png"
            
            # video_id別のディレクトリを作成
            video_output_dir = self.output_dir / video_id
            video_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = video_output_dir / output_filename
            
            # 画像を開いてcell部分を切り出し
            with Image.open(image_path) as img:
                x1, y1, x2, y2 = cell_pos
                
                # cell部分を切り出してそのまま保存
                cell_img = img.crop((x1, y1, x2, y2))
                cell_img.save(output_path, 'PNG')
                
                return output_path, True
                
        except Exception as e:
            print(f"✗ cell処理エラー: {e}")
            return None, False
    
    def process_all_cells(self):
        """すべてのcellを処理する"""
        if not self.data:
            print("✗ JSONデータが読み込まれていません")
            return False
        
        existing_images, missing_images = self.check_image_files()
        
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        
        print(f"\n=== Cell画像処理開始 ===")
        total_cells = len(self.data)
        
        for i, cell in enumerate(self.data):
            image_path = cell["image_path"]
            
            # 進捗表示
            if (i + 1) % 100 == 0 or i == 0:
                print(f"処理中: {i + 1}/{total_cells} ({(i + 1) / total_cells * 100:.1f}%)")
            
            # 画像ファイルが存在しない場合はスキップ
            if not os.path.exists(image_path):
                skipped_count += 1
                continue
            
            # cell処理
            output_path, success = self.process_cell(cell)
            
            if success:
                processed_count += 1
                if i < 5:  # 最初の5個だけパスを表示
                    print(f"  ✓ {output_path}")
            else:
                failed_count += 1
        
        print(f"\n=== 処理結果 ===")
        print(f"✓ 成功: {processed_count} 個")
        print(f"✗ 失敗: {failed_count} 個")
        print(f"⚠ スキップ: {skipped_count} 個 (画像ファイル不存在)")
        print(f"出力ディレクトリ: {self.output_dir}")
        
        return processed_count > 0
    
    def run_validation(self):
        """完全な検証プロセスを実行する"""
        print("=== Cell Annotation検証スクリプト ===\n")
        
        # 1. JSONファイルの読み込み
        if not self.load_json():
            return False
        
        # 2. JSON構造の検証
        if not self.validate_json_structure():
            return False
        
        # 3. 統計情報の表示
        self.get_statistics()
        
        # 4. cell画像の処理
        if not self.process_all_cells():
            return False
        
        print("\n✓ すべての検証と処理が完了しました！")
        return True

def main():
    parser = argparse.ArgumentParser(description='Cell Annotation検証スクリプト')
    parser.add_argument('json_path', help='JSONファイルのパス')
    parser.add_argument('--output-dir', default='output/cell_annotations', 
                       help='出力ディレクトリ (デフォルト: output/cell_annotations)')
    
    args = parser.parse_args()
    
    # 検証実行
    validator = CellAnnotationValidator(args.json_path, args.output_dir)
    success = validator.run_validation()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
