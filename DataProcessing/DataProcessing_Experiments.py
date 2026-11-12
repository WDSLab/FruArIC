import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __package__:
    from .DataProcessing_Src import clean_data
else:
    from DataProcessing_Src import clean_data


def select_data(src_img, src_json, dst_img, dst_json, n_select, mode, seed=None):
    src_img = Path(src_img)
    src_json = Path(src_json)
    dst_img = Path(dst_img)
    dst_json = Path(dst_json)

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_json.mkdir(parents=True, exist_ok=True)

    img_lst = sorted([p.name for p in src_img.iterdir() if p.suffix.lower() == ".jpg"])
    if not 1 <= n_select <= len(img_lst):
        raise ValueError("n_select must be between 1 and the number of source images")
    if mode == "uniform":
        if seed is not None:
            raise ValueError("seed is only supported in random mode")
        indices = np.linspace(0, len(img_lst) - 1, n_select, dtype=int)
    elif mode == "random":
        if seed is None or seed < 0:
            raise ValueError("random mode requires a non-negative seed")
        indices = np.random.default_rng(seed).choice(len(img_lst), size=n_select, replace=False)
    else:
        raise ValueError("mode must be 'uniform' or 'random'")
    selected_lst = [img_lst[i] for i in indices]

    for img_name in selected_lst:
        img_src = src_img / img_name
        lab_src = src_json / Path(img_name).with_suffix(".json").name

        img_dst = dst_img / img_name
        lab_dst = dst_json / Path(img_name).with_suffix(".json").name

        shutil.move(str(img_src), str(img_dst))
        shutil.move(str(lab_src), str(lab_dst))

    print(f"Source total: {len(img_lst)}")
    print("Remaining source image:", len(os.listdir(src_img)))
    print("Remaining source label:", len(os.listdir(src_json)))

    print(f"Selected total: {len(selected_lst)}")
    print("Selected image:", len(os.listdir(dst_img)))
    print("Selected label:", len(os.listdir(dst_json)))


def build_data(exp_img, exp_json, processed_dir):
    exp_img = Path(exp_img)
    exp_json = Path(exp_json)
    processed_dir = Path(processed_dir)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for jp in tqdm(sorted(exp_json.glob("*.json")), desc="making processed experiment images"):
        data = json.loads(jp.read_text(encoding="utf-8"))
        ann = data["annotations"][0]
        bbox = ann.get("bbox", {})
        text_val = str(bbox.get("text", "")).strip()
        shutil.copy2(exp_img / f"{jp.stem}.jpg", processed_dir / f"{text_val}.jpg")

    print("Processed image:", len(os.listdir(processed_dir)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_img", required=True, help="path to source experiment image directory")
    parser.add_argument("--src_json", required=True, help="path to source experiment json directory")
    parser.add_argument("--exp_img", required=True, help="path to selected experiment image directory")
    parser.add_argument("--exp_json", required=True, help="path to selected experiment json directory")
    parser.add_argument("--processed_dir", required=True, help="path to processed experiment image directory")
    parser.add_argument("--n_select", type=int, default=1000)
    parser.add_argument("--mode", choices=["uniform", "random"], required=True)
    parser.add_argument("--seed", type=int, help="non-negative seed; required for random mode")

    opt = parser.parse_args()
    if opt.mode == "random" and opt.seed is None:
        parser.error("--mode random requires --seed")
    if opt.mode == "uniform" and opt.seed is not None:
        parser.error("--seed is only supported with --mode random")
    if opt.seed is not None and opt.seed < 0:
        parser.error("--seed must be non-negative")
    clean_data(opt.src_img, opt.src_json)
    select_data(opt.src_img, opt.src_json, opt.exp_img, opt.exp_json,
                opt.n_select, opt.mode, opt.seed)
    build_data(opt.exp_img, opt.exp_json, opt.processed_dir)
