# This file is adapted from CLOVA AI's deep-text-recognition-benchmark:
# https://github.com/clovaai/deep-text-recognition-benchmark
#
# Original copyright:
# Copyright (c) 2019-present NAVER Corp.
#
# Licensed under the Apache License, Version 2.0.
# Modifications were made for the proposed container code recognition framework.

import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance

from CharRec.utils import CTCLabelConverter, Averager
from CharRec.dataset import hierarchical_dataset, AlignCollate
from CharRec.model import Model
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)
        
        preds = model(image)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)
        valid_loss_avg.add(cost)
        
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index.data, preds_size.data)


        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if pred == gt:
                n_correct += 1

            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1] # multiply of pred_max_prob
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels


def test(opt):
    """ model config """
    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    print('model input parameters', opt.imgH, opt.imgW, opt.input_channel, opt.output_channel, opt.num_class, opt.batch_max_length, opt.FeatureExtraction)

    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    model = model.module.to(device) #for cpu env

    """ setup loss """
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)

    """ evaluation """
    model.eval()
    with torch.no_grad():
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
        eval_data = hierarchical_dataset(root=opt.eval_data, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(eval_data, batch_size=opt.batch_size, shuffle=False, num_workers=int(0), collate_fn=AlignCollate_evaluation, pin_memory=True)
        _, accuracy_by_best_model, _, _, _, _ = validation(model, criterion, evaluation_loader, converter, opt)
        print(f'{accuracy_by_best_model:0.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=45, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=150, help='the width of the input image')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    
    """ Model Architecture """
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. DeformableGRCNN_l|DeformableCNN_l|DeformableCNN_s|CNN_s')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    opt = parser.parse_args()

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)
