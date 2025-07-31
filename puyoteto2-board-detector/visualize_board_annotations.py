import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse

class JSONImageValidator:
    def __init__(self, json_path, output_dir="output", base_image_dir="data/img"):
        """
        JSON画像検証クラス
        
        Args:
            json_path (str): JSONファイルのパス
            output_dir (str): 出力ディレクトリ
            base_image_dir (str): 画像ベースディレクトリ
        """
        self.json_path = json_path
        self.output_dir = Path(output_dir)
        self.base_image_dir = Path(base_image_dir)
        self.data = None
        
        # 出力ディレクトリを作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_json(self):
        """JSONファイルを読み込む"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✓ JSONファイルを正常に読み込みました: {self.json_path}")
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
        
        required_video_keys = ["video_id", "video_url", "video_title", "frames"]
        required_frame_keys = ["image_file", "timestamp", "boards"]
        required_board_keys = ["position", "labels"]
        
        for i, video in enumerate(self.data):
            # videoレベルの検証
            for key in required_video_keys:
                if key not in video:
                    print(f"✗ video[{i}]に必須キー '{key}' がありません")
                    return False
            
            # framesの検証
            if not isinstance(video["frames"], list):
                print(f"✗ video[{i}].framesは配列である必要があります")
                return False
            
            for j, frame in enumerate(video["frames"]):
                # frameレベルの検証
                for key in required_frame_keys:
                    if key not in frame:
                        print(f"✗ video[{i}].frames[{j}]に必須キー '{key}' がありません")
                        return False
                
                # boardsの検証
                if not isinstance(frame["boards"], list):
                    print(f"✗ video[{i}].frames[{j}].boardsは配列である必要があります")
                    return False
                
                for k, board in enumerate(frame["boards"]):
                    for key in required_board_keys:
                        if key not in board:
                            print(f"✗ video[{i}].frames[{j}].boards[{k}]に必須キー '{key}' がありません")
                            return False
                    
                    # positionは4つの数値の配列である必要がある
                    if not isinstance(board["position"], list) or len(board["position"]) != 4:
                        print(f"✗ video[{i}].frames[{j}].boards[{k}].positionは4つの数値の配列である必要があります")
                        return False
        
        print("✓ JSON構造の検証が完了しました")
        return True
    
    def check_image_files(self):
        """画像ファイルの存在を確認する"""
        missing_images = []
        existing_images = []
        
        for video in self.data:
            video_id = video["video_id"]
            for frame in video["frames"]:
                image_file = frame["image_file"]
                image_path = self.base_image_dir / video_id / image_file
                
                if image_path.exists():
                    existing_images.append(str(image_path))
                else:
                    missing_images.append(str(image_path))
        
        if missing_images:
            print(f"✗ {len(missing_images)} 個の画像ファイルが見つかりません:")
            for img in missing_images[:5]:  # 最初の5個だけ表示
                print(f"   - {img}")
            if len(missing_images) > 5:
                print(f"   ... 他 {len(missing_images) - 5} 個")
        
        print(f"✓ {len(existing_images)} 個の画像ファイルが存在します")
        return existing_images, missing_images
    
    def get_font(self, size=24):
        """フォントを取得する（日本語対応）"""
        # 日本語フォントを試す
        font_paths = [
            "/System/Library/Fonts/Arial Unicode MS.ttf",  # macOS
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",  # macOS
            "C:\\Windows\\Fonts\\msgothic.ttc",  # Windows
            "C:\\Windows\\Fonts\\meiryo.ttc",  # Windows
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except:
                    continue
        
        # デフォルトフォントを使用
        try:
            return ImageFont.load_default()
        except:
            return None
    
    def process_image(self, video_data, frame_data, image_path, output_path):
        """画像に情報を付与して処理する"""
        try:
            # 画像を開く
            with Image.open(image_path) as img:
                # RGBAモードに変換（透明度サポート）
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # 描画オブジェクトを作成
                draw = ImageDraw.Draw(img)
                
                # フォントを取得
                font_large = self.get_font(28)
                font_small = self.get_font(20)
                
                # メタデータを左上に描画
                video_id = video_data["video_id"]
                video_title = video_data["video_title"]
                video_url = video_data["video_url"]
                timestamp = frame_data["timestamp"]
                
                # 背景用の半透明矩形を描画
                text_lines = [
                    f"Video ID: {video_id}",
                    f"Title: {video_title[:50]}{'...' if len(video_title) > 50 else ''}",
                    f"URL: {video_url}",
                    f"Timestamp: {timestamp}"
                ]
                
                # テキストの背景矩形を計算
                max_width = 0
                total_height = 0
                line_heights = []
                
                for line in text_lines:
                    if font_large:
                        bbox = draw.textbbox((0, 0), line, font=font_large)
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                    else:
                        width = len(line) * 10
                        height = 20
                    
                    max_width = max(max_width, width)
                    line_heights.append(height)
                    total_height += height + 5
                
                # 背景矩形を描画
                padding = 10
                bg_rect = [5, 5, max_width + padding * 2, total_height + padding]
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
                img = Image.alpha_composite(img, overlay)
                draw = ImageDraw.Draw(img)
                
                # テキストを描画
                y_offset = 15
                for i, line in enumerate(text_lines):
                    if font_large:
                        draw.text((15, y_offset), line, fill=(255, 255, 255, 255), font=font_large)
                        y_offset += line_heights[i] + 5
                    else:
                        draw.text((15, y_offset), line, fill=(255, 255, 255, 255))
                        y_offset += 25
                
                # boardsの位置を枠で囲む
                colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), 
                         (255, 255, 0, 255), (255, 0, 255, 255), (0, 255, 255, 255)]
                
                for i, board in enumerate(frame_data["boards"]):
                    position = board["position"]
                    x1, y1, x2, y2 = position
                    color = colors[i % len(colors)]
                    
                    # 太い枠線を描画
                    line_width = 4
                    for offset in range(line_width):
                        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], 
                                     outline=color, width=1)
                    
                    # ボード番号を描画
                    label_text = f"Board {i + 1}"
                    if font_small:
                        label_bbox = draw.textbbox((x1, y1 - 30), label_text, font=font_small)
                        label_bg = [label_bbox[0] - 5, label_bbox[1] - 5, 
                                   label_bbox[2] + 5, label_bbox[3] + 5]
                        draw.rectangle(label_bg, fill=color)
                        draw.text((x1, y1 - 30), label_text, fill=(0, 0, 0, 255), font=font_small)
                    else:
                        draw.rectangle([x1, y1 - 25, x1 + 80, y1], fill=color)
                        draw.text((x1 + 5, y1 - 20), label_text, fill=(0, 0, 0, 255))
                
                # 画像を保存
                # RGBモードに変換してJPEGで保存
                if img.mode == 'RGBA':
                    # 白背景に合成
                    white_bg = Image.new('RGB', img.size, (255, 255, 255))
                    white_bg.paste(img, mask=img.split()[-1])  # アルファチャンネルをマスクとして使用
                    img = white_bg
                
                img.save(output_path, 'JPEG', quality=95)
                return True
                
        except Exception as e:
            print(f"✗ 画像処理エラー ({image_path}): {e}")
            return False
    
    def process_all_images(self):
        """すべての画像を処理する"""
        if not self.data:
            print("✗ JSONデータが読み込まれていません")
            return False
        
        existing_images, missing_images = self.check_image_files()
        
        processed_count = 0
        failed_count = 0
        
        for video in self.data:
            video_id = video["video_id"]
            
            # 出力用のディレクトリを作成
            video_output_dir = self.output_dir / video_id
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            for frame in video["frames"]:
                image_file = frame["image_file"]
                image_path = self.base_image_dir / video_id / image_file
                
                if not image_path.exists():
                    print(f"⚠ スキップ: {image_path} (ファイルが見つかりません)")
                    failed_count += 1
                    continue
                
                # 出力パス
                output_filename = f"processed_{image_file}"
                output_path = video_output_dir / output_filename
                
                # 画像を処理
                if self.process_image(video, frame, image_path, output_path):
                    print(f"✓ 処理完了: {output_path}")
                    processed_count += 1
                else:
                    failed_count += 1
        
        print(f"\n=== 処理結果 ===")
        print(f"✓ 成功: {processed_count} 個")
        print(f"✗ 失敗: {failed_count} 個")
        print(f"出力ディレクトリ: {self.output_dir}")
        
        return processed_count > 0
    
    def run_validation(self):
        """完全な検証プロセスを実行する"""
        print("=== JSON画像検証スクリプト ===\n")
        
        # 1. JSONファイルの読み込み
        if not self.load_json():
            return False
        
        # 2. JSON構造の検証
        if not self.validate_json_structure():
            return False
        
        # 3. 画像ファイルの確認と処理
        if not self.process_all_images():
            return False
        
        print("\n✓ すべての検証と処理が完了しました！")
        return True

def main():
    parser = argparse.ArgumentParser(description='JSON画像検証スクリプト')
    parser.add_argument('json_path', help='JSONファイルのパス')
    parser.add_argument('--output-dir', default='output/board_annotations', help='出力ディレクトリ (デフォルト: output)')
    parser.add_argument('--image-dir', default='data/img', help='画像ベースディレクトリ (デフォルト: data/img)')
    
    args = parser.parse_args()
    
    # 検証実行
    validator = JSONImageValidator(args.json_path, args.output_dir, args.image_dir)
    success = validator.run_validation()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
