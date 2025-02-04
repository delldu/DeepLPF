# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

'''
import matplotlib
matplotlib.use('agg')
import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from util import ImageProcessing
from skimage.metrics import structural_similarity as ssim
import logging
import pdb

np.set_printoptions(threshold=sys.maxsize)


class Evaluator():

    def __init__(self, criterion, data_loader, split_name, log_dirpath):
        """Initialisation function for the data loader

        :param criterion: loss function
        :param data_loader: an instance of the DataLoader class for the dataset of interest
        :param split_name: name of the split e.g. "test", "validation"
        :param log_dirpath: logging directory
        :returns: N/A
        :rtype: N/A

        """
        super().__init__()
        self.criterion = criterion
        self.data_loader = data_loader
        self.split_name = split_name
        self.log_dirpath = log_dirpath

    def evaluate(self, net, epoch=0):
        """Evaluates a network on a specified split of a dataset e.g. test, validation

        :param net: PyTorch neural network data structure
        :param epoch: current epoch
        :returns: average loss, average PSNR, average SSIM
        :rtype: float, float, float

        """
        
        psnr_avg = 0.0
        ssim_avg = 0.0
        examples = 0
        running_loss = 0
        num_batches = 0
        batch_size = 1

        out_dirpath = self.log_dirpath + "/" + self.split_name.lower()
        if not os.path.isdir(out_dirpath):
            os.mkdir(out_dirpath)

        # switch model to evaluation mode
        net.eval()
        net.cuda()

        progress_count = 0
        with torch.no_grad():
            # for batch_num, data in enumerate(self.data_loader, 0):
            for input_tensor, gt_tensor, name in self.data_loader:
                print("progress: ", progress_count, " ...")
                progress_count += 1

                batch_num = input_tensor.size(0)

                # input_img_batch, output_img_batch, name = Variable(data['input_img'], requires_grad=False).cuda(), Variable(data['output_img'],
                #   requires_grad=False).cuda(), data['name']
                # input_tensor.size() -- [1, 3, 341, 512], 
                # input_tensor.min(), input_tensor.max() -- 0.0275, 0.882
                                                                  
                input_img_batch = input_tensor.cuda()
                output_img_batch = gt_tensor.cuda()


                input_img_batch = input_img_batch.unsqueeze(0)

                print(name)


                for i in range(0, input_img_batch.shape[0]):

                    img = input_img_batch[i, :, :, :]
                    img = torch.clamp(img, 0, 1)

                    net_output_img_example = net(img)

                    if net_output_img_example.shape[2]!=output_img_batch.shape[2]:
                        net_output_img_example=net_output_img_example.transpose(2,3)

                    loss = self.criterion(net_output_img_example[:, 0:3, :, :],
                                          output_img_batch[:, 0:3, :, :])

                    input_img_example = (input_img_batch.cpu(
                    ).data[0, 0:3, :, :].numpy() * 255).astype('uint8')

                    output_img_batch_numpy = output_img_batch.squeeze(
                        0).data.cpu().numpy()
                    output_img_batch_numpy = ImageProcessing.swapimdims_3HW_HW3(
                        output_img_batch_numpy)
                    output_img_batch_rgb = output_img_batch_numpy
                    output_img_batch_rgb = ImageProcessing.swapimdims_HW3_3HW(
                        output_img_batch_rgb)
                    output_img_batch_rgb = np.expand_dims(
                        output_img_batch_rgb, axis=0)

                    net_output_img_example_numpy = net_output_img_example.squeeze(
                        0).data.cpu().numpy()
                    net_output_img_example_numpy = ImageProcessing.swapimdims_3HW_HW3(
                        net_output_img_example_numpy)
                    net_output_img_example_rgb = net_output_img_example_numpy
                    net_output_img_example_rgb = ImageProcessing.swapimdims_HW3_3HW(
                        net_output_img_example_rgb)
                    net_output_img_example_rgb = np.expand_dims(
                        net_output_img_example_rgb, axis=0)
                    net_output_img_example_rgb = np.clip(
                        net_output_img_example_rgb, 0, 1)

                    running_loss += loss.data[0]
                    examples += batch_size
                    num_batches += 1

                    psnr_example = ImageProcessing.compute_psnr(output_img_batch_rgb.astype(np.float32),
                                                                net_output_img_example_rgb.astype(np.float32), 1.0)
                    ssim_example = ImageProcessing.compute_ssim(output_img_batch_rgb.astype(np.float32),
                                                                net_output_img_example_rgb.astype(np.float32))

                    psnr_avg += psnr_example
                    ssim_avg += ssim_example
                    
                    if batch_num > 30:
                        '''
                        We save only the first 30 images down for time saving
                        purposes
                        '''
                        continue
                    else:

                        output_img_example = (
                            output_img_batch_rgb[0, 0:3, :, :] * 255).astype('uint8')
                        net_output_img_example = (
                            net_output_img_example_rgb[0, 0:3, :, :] * 255).astype('uint8')

                        plt.imsave(out_dirpath + "/" + name[0].split(".")[0] + "_" + self.split_name.upper() + "_" + str(epoch + 1) + "_" + str(
                            examples) + "_PSNR_" + str("{0:.3f}".format(psnr_example)) + "_SSIM_" + str(
                            "{0:.3f}".format(ssim_example)) + ".jpg",
                            ImageProcessing.swapimdims_3HW_HW3(net_output_img_example))

                    del net_output_img_example_numpy
                    del net_output_img_example_rgb
                    del output_img_batch_rgb
                    del output_img_batch_numpy
                    del input_img_example
                    del output_img_batch

        psnr_avg = psnr_avg / num_batches
        ssim_avg = ssim_avg / num_batches

        logging.info('loss_%s: %.5f psnr_%s: %.3f ssim_%s: %.3f' % (
            self.split_name, (running_loss / examples), self.split_name, psnr_avg, self.split_name, ssim_avg))

        loss = (running_loss / examples)

        return loss, psnr_avg, ssim_avg
