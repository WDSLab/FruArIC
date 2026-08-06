import argparse
import os
import cv2
import json
import shutil
import unicodedata
import numpy as np

from pathlib import Path
from tqdm import tqdm
from natsort import natsorted
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance
import lmdb

def make_dirs(dir_dataset):
    os.mkdir(dir_dataset)
    os.mkdir(dir_dataset / "srcs")
    os.mkdir(dir_dataset / "train")
    os.mkdir(dir_dataset / "test")
    os.mkdir(dir_dataset / "validation")
    os.mkdir(dir_dataset / "gt")


def crop_annotation_based(src_img, src_json, output_dir):
    for jp in tqdm(sorted(src_json.glob("*.json")), desc="annotation-based crop"):
        data = json.loads(jp.read_text(encoding="utf-8"))

        annotations = []
        for ann in data.get("annotations", []):
            bbox = ann.get("bbox", {})
            text_val = str(bbox.get("text", "")).strip()
            points = bbox.get("points", [])

            if text_val and points:
                annotations.append(ann)

        img = Image.open(f"{src_img / jp.stem}.jpg").convert("RGB")
        #enhancer = ImageEnhance.Sharpness(img)
        #enhanced_img = enhancer.enhance(1.05)
        w_img, h_img = img.width, img.height


        for idx, ann in enumerate(annotations):
            bbox = ann.get("bbox", {})
            text_val = str(bbox.get("text", "")).strip()
            points = bbox.get("points", [])

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            xmin = int(min(xs))
            ymin = int(min(ys))
            xmax = int(max(xs))
            ymax = int(max(ys))

            # xmin = max(0, xmin - 5)
            # ymin = max(0, ymin - 5)
            # xmax = min(w_img, xmax + 5)
            # ymax = min(h_img, ymax + 5)

            cropped_img = img.crop((xmin, ymin, xmax, ymax))
            cropped_img.save(f"{output_dir / text_val}.jpg", quality=95, subsampling=0)


def normalize_filenames(src_cropped):
    target_dir = str(src_cropped)
    for f in os.listdir(target_dir):
        os.rename(os.path.join(target_dir, f),
                  os.path.join(target_dir, unicodedata.normalize("NFC", f).replace(" ", "")))  


def split_data(src_cropped, dir_tr, dir_te, dir_vl, seed):
    img_lst = os.listdir(src_cropped)

    train_lst, validation_lst = train_test_split(img_lst, test_size=0.2, random_state=seed)
    validation_lst, test_lst = train_test_split(validation_lst, test_size=0.5, random_state=seed)

    print(f"train\n{len(train_lst)}\n\nvalidation\n{len(validation_lst)}\n\ntest\n{len(test_lst)}")

    src_img = str(src_cropped)
    dst_tr = str(dir_tr)
    dst_te = str(dir_te)
    dst_vl = str(dir_vl)

    for fname in train_lst:
        fname = fname.split("/")[-1]
        shutil.move(src_img + "/" + fname, dst_tr + "/" + fname)

    for fname in test_lst:
        fname = fname.split("/")[-1]
        shutil.move(src_img + "/" + fname, dst_te + "/" + fname)

    for fname in validation_lst:
        fname = fname.split("/")[-1]
        shutil.move(src_img + "/" + fname, dst_vl + "/" + fname)

    print("train_img:", len(os.listdir(dir_tr)))
    print("test_img:", len(os.listdir(dir_te)))
    print("valid_img:", len(os.listdir(dir_vl)))

    return train_lst, test_lst, validation_lst                

        
def make_gt(train_lst, test_lst, validation_lst, dir_gt):
    gt_path = str(dir_gt) + "/"

    split_dict = {"train": train_lst,
                  "test": test_lst,
                  "validation": validation_lst}

    for split_name, img_lst in split_dict.items():
        gt_file = open(gt_path + f"gt_{split_name}.txt", "w", encoding="utf-8")

        for img in natsorted(img_lst):
            text = img.strip().split(".")[0]
            gt_file.write(f"{split_name}/{img}\t{text}\n")

        print(f"gt_{split_name} is made")
        gt_file.close()
        
        
def check_gt(dir_dataset, dir_gt):
    targets = ["train", "test", "validation"]
    for target in targets:
        gt_file = os.path.join(str(dir_gt), f"gt_{target}.txt")
        img_dir = os.path.join(str(dir_dataset), target)

        with open(gt_file, "r", encoding="utf-8") as file:
            lines = file.readlines()

        gt_img_lst = []
        for line in lines:
            img = line.strip().split("\t")[0].split("/")[1]
            gt_img_lst.append(img)

        folder_img_lst = os.listdir(img_dir)

        # 1. GT에는 있는데 실제 폴더에는 없는 이미지
        missing_img_lst = []
        for img in gt_img_lst:
            if img not in folder_img_lst:
                print(f"except: {img}")
                missing_img_lst.append(img)

        # 2. 실제 폴더에는 있는데 GT에는 없는 이미지
        unlabeled_img_lst = []
        for img in folder_img_lst:
            if img not in gt_img_lst:
                unlabeled_img_lst.append(img)
        if unlabeled_img_lst:
            for img in unlabeled_img_lst:
                os.remove(os.path.join(img_dir, img))

        print(target)
        print("len(gt):", len(lines))
        print("missing_img_lst:", len(missing_img_lst))
        print("unlabeled_img_lst:", len(unlabeled_img_lst), "\n")

    print("train_img:", len(os.listdir(os.path.join(str(dir_dataset), "train"))))
    print("test_img:", len(os.listdir(os.path.join(str(dir_dataset), "test"))))
    print("valid_img:", len(os.listdir(os.path.join(str(dir_dataset), "validation"))))
    

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

    
def createdataset(inputPath, outputPath, gtPath, enc, checkValid=True):
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=10737418240)
    cache = {}
    cnt = 1

    with open(gtPath, "r", encoding=enc) as data:
        datalist = data.readlines()

    nSamples = len(datalist)

    for i in range(nSamples):
        imagePath, label = datalist[i].strip("\n").split("\t")
        imagePath = os.path.join(inputPath, imagePath)

        if not os.path.exists(imagePath):
            print("%s does not exist" % imagePath)
            continue

        with open(imagePath, "rb") as f:
            imageBin = f.read()

        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print("%s is not a valid image" % imagePath)
                    continue
            except:
                print("error occured", i)
                with open(outputPath + "/error_image_log.txt", "a") as log:
                    log.write("%s-th image data occured error\n" % str(i))
                continue

        imageKey = "image-%09d".encode() % cnt
        labelKey = "label-%09d".encode() % cnt

        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print("Written %d / %d" % (cnt, nSamples))

        cnt += 1

    nSamples = cnt - 1
    cache["num-samples".encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print("Created dataset with %d samples" % nSamples)    
    
    
def make_lmdb(dir_dataset, dir_lmdb, dir_gt):
    inputPath = str(dir_dataset)
    enc = "utf-8"

    outputPath = str(dir_lmdb / "train")
    gtPath = str(dir_gt / "gt_train.txt")
    createdataset(inputPath, outputPath, gtPath, enc)

    outputPath = str(dir_lmdb / "test")
    gtPath = str(dir_gt / "gt_test.txt")
    createdataset(inputPath, outputPath, gtPath, enc)

    outputPath = str(dir_lmdb / "validation")
    gtPath = str(dir_gt / "gt_validation.txt")
    createdataset(inputPath, outputPath, gtPath, enc)    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_img", required=True)
    parser.add_argument("--src_json", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--seed", type=int, default=1111)

    opt = parser.parse_args()
    
    src_img = Path(opt.src_img)
    src_json = Path(opt.src_json)
    dir_dataset = Path(opt.work_dir)
    src_cropped = dir_dataset / "srcs"
    dir_tr = dir_dataset / "train"
    dir_te = dir_dataset / "test"
    dir_vl = dir_dataset / "validation"
    dir_gt = dir_dataset / "gt"
    dir_lmdb = dir_dataset / "lmdb"
    make_dirs(dir_dataset)

    crop_annotation_based(src_img, src_json, src_cropped)
    normalize_filenames(src_cropped)

    train_lst, test_lst, validation_lst = split_data(src_cropped, dir_tr, dir_te, dir_vl, opt.seed)
    make_gt(train_lst, test_lst, validation_lst, dir_gt)
    check_gt(dir_dataset, dir_gt)

    make_lmdb(dir_dataset, dir_lmdb, dir_gt)

