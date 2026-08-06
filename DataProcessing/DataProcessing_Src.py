import argparse
import json
import os
from pathlib import Path


def clean_data(src_img, src_json):
    src_img = Path(src_img)
    src_json = Path(src_json)

    jsons = sorted(src_json.glob("*.json"))
    duplicated = set()

    for j in jsons:
        with open(j, "r", encoding="utf-8") as f:
            data = json.load(f)

        annot = data.get("annotations", [])
        bic_annot = [a for a in annot if a.get("bbox", {}).get("classid") == "BIC"]
        if len(bic_annot) != 1:
            os.remove(j)
            continue

        bbox = bic_annot[0].get("bbox", {})
        text_val = str(bbox.get("text", "")).strip()
        if text_val == "":
            os.remove(j)
            continue

        if text_val in duplicated:
            os.remove(j)
            continue
        duplicated.add(text_val)

        data["annotations"] = bic_annot
        with open(j, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    stem_json = {p.stem for p in src_json.glob("*.json")}
    stem_img = {p.stem for p in src_img.glob("*.jpg")}
    
    for img_path in src_img.glob("*.jpg"):
        if img_path.stem not in stem_json:
            os.remove(img_path)
                 
    for json_path in src_json.glob("*.json"):
        if json_path.stem not in stem_img:
            json_path.unlink()
        
    print("Source preprocessing is done.")
    print("images:", len(list(src_img.glob("*.jpg"))))
    print("jsons :", len(list(src_json.glob("*.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_img", required=True, help="path to source image directory")
    parser.add_argument("--src_json", required=True, help="path to source json directory")
    
    opt = parser.parse_args()
    clean_data(opt.src_img, opt.src_json)
