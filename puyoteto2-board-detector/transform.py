from torchvision import transforms
from PIL import ImageDraw
import random

# 白線ノイズ（アニメーションの白い線を再現）
class RandomWhiteLineNoise:
    def __init__(self, p=0.5, max_lines=4, max_line_width=10):
        self.p = p
        self.max_lines = max_lines
        self.max_line_width = max_line_width  # 最大線幅

    def __call__(self, img):
        if random.random() > self.p:
            return img
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for _ in range(random.randint(1, self.max_lines)):
            line_width = random.randint(1, self.max_line_width)  # ランダムに線幅決定
            if random.random() < 0.5:
                y = random.randint(0, h - 1)
                draw.line([(0, y), (w, y)], fill=(255, 255, 255), width=line_width)
            else:
                x = random.randint(0, w - 1)
                draw.line([(x, 0), (x, h)], fill=(255, 255, 255), width=line_width)
        return img


# 白飛びノイズ（アニメーションの一部白化を再現）
class RandomWhiteSpotNoise:
    def __init__(self, p=0.5, max_spots=7, max_radius=10):
        self.p = p
        self.max_spots = max_spots
        self.max_radius = max_radius  # 最大半径

    def __call__(self, img):
        if random.random() > self.p:
            return img
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for _ in range(random.randint(1, self.max_spots)):
            r = random.randint(1, self.max_radius)  # ランダムに半径決定
            x = random.randint(0, w - r - 1)
            y = random.randint(0, h - r - 1)
            draw.ellipse((x, y, x + r, y + r), fill=(255, 255, 255))
        return img


def get_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),                # 左右反転（上下は禁止）
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),            # 明るさ/コントラスト/彩度/色相
        RandomWhiteLineNoise(p=0.5),
        RandomWhiteSpotNoise(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

def get_visualize_dataset_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),                # 左右反転（上下は禁止）
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),            # 明るさ/コントラスト/彩度/色相
        RandomWhiteLineNoise(p=0.5),
        RandomWhiteSpotNoise(p=0.5)
    ])