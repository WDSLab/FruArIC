import argparse
import json
import math
import os
import shutil
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def points_to_aabb(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    xmin = math.floor(min(xs))
    ymin = math.floor(min(ys))
    xmax = math.ceil(max(xs))
    ymax = math.ceil(max(ys))
    return int(xmin), int(ymin), int(xmax), int(ymax)


def build_yolo_txt_lines(sample: dict, W: int, H: int):
    lines = []
    annot = sample.get("annotations", [])
    for item in annot:
        bbox = (item or {}).get("bbox", {})
        xmin, ymin, xmax, ymax = points_to_aabb(bbox["points"])
        w = xmax - xmin
        h = ymax - ymin
        xc = (xmin + w / 2) / W
        yc = (ymin + h / 2) / H
        ww = w / W
        hh = h / H
        lines.append(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
    return lines


def j2t(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_path = json_path.parent.parent / "images" / f"{json_path.stem}.jpg"
    with Image.open(img_path) as img:
        W, H = img.size

    lines = build_yolo_txt_lines(data, W, H)
    out_path = json_path.with_suffix(".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    json_path.unlink()
    return out_path


def build_data(src_img, src_json, dst_data):
    src_img = Path(src_img)
    src_json = Path(src_json)
    dst_data = Path(dst_data)

    img_out = dst_data / "images"
    label_out = dst_data / "labels"

    os.mkdir(dst_data)
    os.mkdir(img_out)
    os.mkdir(label_out)

    json_files = sorted(src_json.glob("*.json"))

    for jp in tqdm(json_files, desc="copy images and convert labels"):
        img_path = src_img / f"{jp.stem}.jpg"
        json_path = label_out / jp.name

        shutil.copy2(img_path, img_out / img_path.name)
        shutil.copy2(jp, json_path)

        j2t(json_path)

    print(len(os.listdir(img_out)))
    print(len(os.listdir(label_out)))
 
 
def split_data(dir_data, seed):
    dir_data = Path(dir_data)
    img_dir = dir_data / "images"
    label_dir = dir_data / "labels"

    img_lst = sorted([str(p) for p in img_dir.glob("*.jpg")])
    lab_lst = sorted([str(p) for p in label_dir.glob("*.txt")])
    print(len(img_lst))
    print(len(lab_lst))

    train_lst, test_lst = train_test_split(img_lst, test_size=0.2, random_state=seed)
    test_lst, val_lst = train_test_split(test_lst, test_size=0.5, random_state=seed)
    print(len(train_lst))
    print(len(test_lst))
    print(len(val_lst))

    split_dict = {"train": train_lst,
                  "test": test_lst,
                  "val": val_lst}

    for split in split_dict.keys():
        os.mkdir(dir_data / split)
        os.mkdir(dir_data / split / "images")
        os.mkdir(dir_data / split / "labels")

    for split, split_lst in split_dict.items():
        dst_img = dir_data / split / "images"
        dst_lab = dir_data / split / "labels"

        for img_path in split_lst:
            img_path = Path(img_path)
            label_name = img_path.stem + ".txt"

            shutil.move(str(img_path), str(dst_img / img_path.name))
            shutil.move(str(label_dir / label_name), str(dst_lab / label_name))

    print(len(os.listdir(dir_data / "train" / "images")),
          len(os.listdir(dir_data / "test" / "images")),
          len(os.listdir(dir_data / "val" / "images")))

    print(len(os.listdir(dir_data / "train" / "labels")),
          len(os.listdir(dir_data / "test" / "labels")),
          len(os.listdir(dir_data / "val" / "labels")))
    
    
def make_yaml(dir_data):
    dir_data = Path(dir_data)

    data_yaml = f"""train: {dir_data / "train" / "images"}
val: {dir_data / "val" / "images"}
test: {dir_data / "test" / "images"}

nc: 1
names: ['BIC']
"""

    with open(dir_data / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_img", required=True, help="path to preprocessed source image directory")
    parser.add_argument("--src_json", required=True, help="path to preprocessed source json directory")
    parser.add_argument("--dir_data", required=True, help="path to output dataset directory")
    parser.add_argument("--seed", type=int, default=42)

    opt = parser.parse_args()
    build_data(opt.src_img, opt.src_json, opt.dir_data)
    split_data(opt.dir_data, opt.seed)
    make_yaml(opt.dir_data)
        