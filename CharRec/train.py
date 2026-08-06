# This file is adapted from CLOVA AI's deep-text-recognition-benchmark:
# https://github.com/clovaai/deep-text-recognition-benchmark
#
# Original copyright:
# Copyright (c) 2019-present NAVER Corp.
#
# Licensed under the Apache License, Version 2.0.
# Modifications were made for the proposed container code recognition framework.

import os
import sys
import time
import copy
import random
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np

from CharRec.utils import CTCLabelConverter, Averager
from CharRec.dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from CharRec.model import Model
from CharRec.test import validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_teacher(opt):
    teacher_opt = copy.deepcopy(opt)
    teacher_opt.FeatureExtraction = opt.teacher_backbone
    teacher_opt.output_channel = opt.teacher_output
    teacher = Model(teacher_opt)
    teacher = torch.nn.DataParallel(teacher).to(device)
    teacher.load_state_dict(torch.load(opt.teacher_path, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def train(opt):
    """ dataset preparation """
    print('Filtering the images containing characters which are not in opt.character')
    print('Filtering the images whose label is longer than opt.batch_max_length')

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
    valid_dataset = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=AlignCollate_valid, pin_memory=True)
    print('-' * 80)
    
    """ model configuration """
    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    if opt.distillation:
        teacher = build_teacher(opt)
        print(f"Teacher:{opt.teacher_backbone} [output channel: {opt.teacher_output}, params: {sum(p.numel() for p in teacher.module.parameters()):,}]")
        
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    print('model input parameters', opt.imgH, opt.imgW, opt.input_channel, opt.output_channel, opt.num_class, opt.batch_max_length, opt.FeatureExtraction)
    print(f"params: {sum(p.numel() for p in model.module.parameters()):,}")
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        model.load_state_dict(torch.load(opt.saved_model))
    if opt.distillation and opt.copy_weights:
        teacher_sd = teacher.state_dict()
        student_sd = model.state_dict()
        for k in student_sd.keys():
            if k in teacher_sd and student_sd[k].shape == teacher_sd[k].shape:
                student_sd[k] = teacher_sd[k]
        model.load_state_dict(student_sd)    

    """ setup loss """
    criterion_ctc = torch.nn.CTCLoss(zero_infinity=True)
    if opt.distillation:
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
    loss_avg = Averager()

    m = model.module
    for p in m.parameters():
        p.requires_grad = True
    optimizer = optim.Adadelta((p for p in m.parameters() if p.requires_grad), lr=opt.lr, rho=opt.rho, eps=opt.eps)

    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    model.train()
    
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter
    
    early_stop_patience = 1000      
    early_stop_threshold = 0.001  
    best_valid_loss = float('inf')
    last_improved_iter = 0

    while(True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)
        
        if opt.distillation:
            with torch.no_grad():
                preds_t = teacher(image)
            preds_s = model(image)
            preds_size_s = torch.IntTensor([preds_s.size(1)] * batch_size)
            cost_task = criterion_ctc(preds_s.log_softmax(2).permute(1, 0, 2), text, preds_size_s, length)
            cost_kd = criterion_kl(F.log_softmax(preds_s.mean(dim=1)/opt.temperature, dim=-1), F.softmax(preds_t.mean(dim=1)/opt.temperature, dim=-1))
            cost = opt.alpha*(opt.temperature*opt.temperature)*cost_kd + (1-opt.alpha)*cost_task
        else:
            preds = model(image)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.log_softmax(2).permute(1, 0, 2)
            cost = criterion_ctc(preds, text, preds_size, length)

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()
        loss_avg.add(cost)

        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            elapsed_time = time.time() - start_time
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels = validation(model, criterion_ctc, valid_loader, converter, opt)
                model.train()
                
                #  early stopping check 
                if best_valid_loss - valid_loss > early_stop_threshold:
                    best_valid_loss = valid_loss
                    last_improved_iter = iteration + 1

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy < 100:
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED < 1.0:
                    if current_norm_ED > best_norm_ED:
                        best_norm_ED = current_norm_ED
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
                
                # early stopping stop condition
                if (iteration + 1) - last_improved_iter >= early_stop_patience:
                    early_stop_msg = (f'Early stopping at iteration {iteration+1}: '
                                      f'valid loss did not improve by more than {early_stop_threshold} '
                                      f'for {early_stop_patience} iterations.')
                    print(early_stop_msg)
                    log.write(early_stop_msg + '\n')
                    sys.exit()

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=24000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=10, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')

    """ Data processing """
    parser.add_argument('--select_data', type=str, default='/', help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='1', help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0', help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=45, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=150, help='the width of the input image')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    
    """ Model Architecture """
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. DeformableGRCNN_l|DeformableCNN_l|DeformableCNN_s|CNN_s')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    
    """ Distillation """
    parser.add_argument('--distillation', action='store_true', help='use teacher-student logit distillation')
    parser.add_argument('--teacher_path', default='', help='path to pretrained teacher model. required when --distillation is set')
    parser.add_argument('--teacher_backbone', type=str, default='', help='teacher FeatureExtraction stage. required when --distillation is set')
    parser.add_argument('--teacher_output', type=int, default=1280, help='teacher output channel. required when --distillation is set')
    parser.add_argument('--copy_weights', action='store_true', help='copy teacher weights with identical names and shapes to the student before training')
    parser.add_argument('--alpha', type=float, default=0.16)
    parser.add_argument('--temperature', type=float, default=10.0)

    opt = parser.parse_args()
    opt.manualSeed = 1111
    
    if not opt.exp_name:
        opt.exp_name = f'{opt.FeatureExtraction}'
    if opt.distillation:
        opt.exp_name = f'{opt.exp_name}_from{opt.teacher_backbone}'
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    """ Seed and GPU setting """
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    
    opt.num_gpu = torch.cuda.device_count()
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        opt.batch_size = opt.batch_size * opt.num_gpu

    train(opt)
    