import argparse
import re
import os
import time
import numpy as np

import torch
from PIL import Image
from ultralytics import YOLO

from CharRec.dataset import ResizeNormalize
from CharRec.utils import CTCLabelConverter
from CharRec.model import Model

device = torch.device('cpu')

    
def load_det(opt):
    detector = YOLO(opt.detector_path)
    detector.to(device)
    return detector


def load_rec(opt):
    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    recognizer = torch.nn.DataParallel(Model(opt)).to(device)
    recognizer.load_state_dict(torch.load(opt.recognizer_path, map_location=device))
    return recognizer.module, converter
 

class Measurer:
    def __init__(self, opt, det, rec, converter):
        self.device = torch.device('cpu')
        self.opt = opt
        self.det = det
        self.rec = rec
        self.converter = converter
        self.transform = ResizeNormalize((150, 45))
        

    def detect(self, img):
        results = self.det.predict(img, verbose=False)
        if results[0].boxes.xyxy.shape[0] == 0:
            return None

        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax().item())
        xmin, ymin, xmax, ymax = boxes.xyxy[best_idx].tolist()
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

        return xmin, ymin, xmax, ymax


    def recognize(self, cropped_img):
        with torch.no_grad():
            preds = self.rec(self.transform(cropped_img.convert('L')).unsqueeze(0).to(self.device))
        preds_size = torch.IntTensor([preds.size(1)])
        _, preds_index = preds.max(2)
        pred_str = self.converter.decode(preds_index, preds_size)[0]

        return pred_str
    
    
    def time_execution(self, model, img):
        model.eval()
        execution_times = []
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(img)
                
            for _ in range(100):
                t0 = time.perf_counter()
                _ = model(img)
                t1 = time.perf_counter()
                execution_times.append((t1 - t0) * 1000.0)
                
        latency = float(np.median(execution_times))
        #latency = float(np.mean(execution_times))
        return latency


    def measure(self, target_path):
        self.det.eval()
        self.rec.eval()

        results_lst = []
        det_fail_lst = []
        rec_fail_lst = []

        img_lst = [f for f in os.listdir(target_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in img_lst:
            s_name = re.sub(r'-\d+', '', os.path.splitext(f)[0])
            with Image.open(os.path.join(target_path, f)) as img:
                img = img.convert('RGB')
                img = img.resize((640, 640), Image.BICUBIC)

            bbox = self.detect(img)
            if bbox is None:
                results_lst.append(f'{s_name} -> DET_FAIL')
                det_fail_lst.append(s_name)
                continue
            xmin, ymin, xmax, ymax = bbox
            cropped_img = img.crop((xmin-5, ymin-5, xmax+5, ymax+5))

            preds_str = self.recognize(cropped_img)
            results_lst.append(f'{s_name} -> {preds_str}')
            if s_name.lower() != preds_str.lower():
                rec_fail_lst.append(f'{s_name} -> {preds_str}')

        total_count = len(img_lst)
        correct_count = total_count - (len(det_fail_lst) + len(rec_fail_lst))
        pipeline_accuracy = round((correct_count / total_count) * 100, 3) if total_count > 0 else 0.0
       
        print(f"Pipeline accuracy: {pipeline_accuracy}%")
        # print(f"Detection failures: {len(det_fail_lst)}")
        # print(f"Recognition failures: {len(rec_fail_lst)}")
        
        det_img = torch.rand((1, 3, 640, 640)).to(self.device)
        rec_img = torch.rand((1, 1, 45, 150)).to(self.device)

        det_latency = self.time_execution(self.det.model, det_img)
        rec_latency = self.time_execution(self.rec, rec_img)
        total_latency = det_latency + rec_latency

        print(f"Pipeline latency: {total_latency:.3f} ms")
        # print(f"Detector latency: {det_latency:.3f} ms")
        # print(f"Recognizer latency: {rec_latency:.3f} ms")
        

def main():
    parser = argparse.ArgumentParser()
    
    """ Pipeline inputs """
    parser.add_argument("--image_folder", required=True, help="path to test images")
    parser.add_argument("--detector_path", required=True, help="path to detector checkpoint")
    parser.add_argument("--recognizer_path", required=True, help="path to recognizer checkpoint")

    """ Recognizer data processing """
    parser.add_argument("--character", type=str, default="0123456789abcdefghijklmnopqrstuvwxyz")

    """ Recognizer model architecture """
    parser.add_argument("--FeatureExtraction", type=str, default='CNN_s')
    parser.add_argument("--input_channel", type=int, default=1)
    parser.add_argument("--output_channel", type=int, default=512)
    opt = parser.parse_args()
    
    detector = load_det(opt)
    recognizer, converter = load_rec(opt)

    measurer = Measurer(opt=opt, det=detector, rec=recognizer, converter=converter)
    measurer.measure(opt.image_folder)

if __name__ == "__main__":
    main()
