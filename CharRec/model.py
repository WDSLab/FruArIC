# This file is adapted from CLOVA AI's deep-text-recognition-benchmark:
# https://github.com/clovaai/deep-text-recognition-benchmark
#
# Original copyright:
# Copyright (c) 2019-present NAVER Corp.
#
# Licensed under the Apache License, Version 2.0.
# Modifications were made for the proposed container code recognition framework.

import torch.nn as nn
if __package__:
    from .modules.feature_extraction import (DeformableGRCNN_l_FeatureExtractor,
                                            DeformableCNN_l_FeatureExtractor, DeformableCNN_s_FeatureExtractor,
                                            CNN_s_FeatureExtractor)
else:
    from modules.feature_extraction import (DeformableGRCNN_l_FeatureExtractor,
                                            DeformableCNN_l_FeatureExtractor, DeformableCNN_s_FeatureExtractor,
                                            CNN_s_FeatureExtractor)

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt

        """ FeatureExtraction """       
        if opt.FeatureExtraction == 'DeformableGRCNN_l':
            self.FeatureExtraction = DeformableGRCNN_l_FeatureExtractor()      

        elif opt.FeatureExtraction == 'DeformableCNN_l':
            self.FeatureExtraction = DeformableCNN_l_FeatureExtractor()      
                 
        elif opt.FeatureExtraction == 'DeformableCNN_s':
            self.FeatureExtraction = DeformableCNN_s_FeatureExtractor()    
                    
        elif opt.FeatureExtraction == 'CNN_s':
            self.FeatureExtraction = CNN_s_FeatureExtractor()           
                        
        else:
            raise Exception('No FeatureExtraction module specified')

        self.FeatureExtraction_output = opt.output_channel 
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1)) 

        """ Prediction """
        self.Prediction = nn.Linear(self.FeatureExtraction_output, opt.num_class)


    def forward(self, input):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Prediction stage """
        prediction = self.Prediction(visual_feature.contiguous())

        return prediction