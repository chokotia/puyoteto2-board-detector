
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TetrisCellDataset(Dataset):
    def __init__(self, board_info_path, transform=None, is_train=True, 
                 max_crop_shift=5, expansion_factor=2, padding_color=(0, 0, 0)):
        """
        Args:
            board_info_path: アノテーションファイルのパス
            transform: 画像変換
            is_train: 訓練モードかどうか
            max_crop_shift: クロップ位置のランダムシフト最大値
            expansion_factor: セル拡張倍率（3なら3x3セル）
            padding_color: はみ出し部分の色 (R, G, B)
        """
        with open(board_info_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        self.is_train = is_train
        self.max_crop_shift = max_crop_shift
        self.expansion_factor = expansion_factor
        self.padding_color = padding_color
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def _expand_cell_region(self, cell_pos, board_pos):
        """セル領域を指定倍率で拡張"""
        x1, y1, x2, y2 = cell_pos
        board_x1, board_y1, board_x2, board_y2 = board_pos
        
        # セルの幅と高さ
        cell_width = x2 - x1
        cell_height = y2 - y1
        
        # 拡張量計算（中心から各方向への拡張）
        expand_width = cell_width * (self.expansion_factor - 1) // 2
        expand_height = cell_height * (self.expansion_factor - 1) // 2
        
        # 拡張後の座標
        expanded_x1 = x1 - expand_width
        expanded_y1 = y1 - expand_height
        expanded_x2 = x2 + expand_width
        expanded_y2 = y2 + expand_height
        
        return expanded_x1, expanded_y1, expanded_x2, expanded_y2

    def _create_padded_crop(self, image, crop_region, board_pos):
        """
        指定領域をクロップし、ボード外の部分をパディング色で埋める
        """
        img_width, img_height = image.size
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
        board_x1, board_y1, board_x2, board_y2 = board_pos
        
        # クロップ領域のサイズ
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1
        
        # パディング画像を作成
        padded_img = Image.new('RGB', (crop_width, crop_height), self.padding_color)
        
        # 元画像からコピーする領域を計算
        # 画像境界内に収める
        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(img_width, crop_x2)
        src_y2 = min(img_height, crop_y2)
        
        # ボード境界外は除外
        src_x1 = max(src_x1, board_x1)
        src_y1 = max(src_y1, board_y1)
        src_x2 = min(src_x2, board_x2)
        src_y2 = min(src_y2, board_y2)
        
        # コピー先の座標
        dst_x1 = src_x1 - crop_x1
        dst_y1 = src_y1 - crop_y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        # 有効な領域がある場合のみコピー
        if src_x2 > src_x1 and src_y2 > src_y1:
            src_crop = image.crop((src_x1, src_y1, src_x2, src_y2))
            padded_img.paste(src_crop, (dst_x1, dst_y1))
        
        return padded_img

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample['image_path']).convert('RGB')
        cell_pos = sample['cell_pos']
        board_pos = sample['board_pos']
        
        # セル領域を拡張
        expanded_region = self._expand_cell_region(cell_pos, board_pos)
        
        # 訓練時のランダムシフト
        if self.is_train:
            shift_x = random.randint(-self.max_crop_shift, self.max_crop_shift)
            shift_y = random.randint(-self.max_crop_shift, self.max_crop_shift)
            
            expanded_region = (
                expanded_region[0] + shift_x,
                expanded_region[1] + shift_y,
                expanded_region[2] + shift_x,
                expanded_region[3] + shift_y
            )
        
        # パディング付きクロップ
        cell_img = self._create_padded_crop(image, expanded_region, board_pos)
        label = sample['label']

        if self.transform:
            cell_img = self.transform(cell_img)

        return cell_img, label
