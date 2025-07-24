# board_info.jsonについて

## boards.labelsについて
`board_info.json` の `boards.labels` フィールドは、以下のサイトの MapCode を利用しています：

https://blox.askplays.com/map-maker

---

## JSON構造の例

```json
[
  {
    "image_path": "hSMUJcMYNfg/1.png",
    "boards": [
      {
        "position": [305, 156, 667, 879],
        "labels": "<MapCode>"
      },
      {
        "position": [1249, 156, 1611, 879],
        "labels": "<MapCode>"
      }
    ]
  }
]
```

### 各フィールドの説明

- `image_path`: フレーム画像のパスを指定します。形式は `<YouTube動画ID>/<連番>.png` です。  
  例：`hSMUJcMYNfg/1.png` の場合、`hSMUJcMYNfg` がYouTubeの動画IDで、`1.png` は切り出した1枚目のフレーム画像を表します。

- `position`: テトリス盤面が画像内のどこにあるかを示す座標です。  
  配列 `[左上X, 左上Y, 右下X, 右下Y]` の形式で指定します。

- `labels`: 各盤面の状態を、[blox.askplays.com/map-maker](https://blox.askplays.com/map-maker) で生成したMapCodeで表します。
