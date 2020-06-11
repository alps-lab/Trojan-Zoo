# -*- coding: utf-8 -*-


import sys
sys.path.append(r'/home/panrusheng/Trojan-Zoo')
print(sys.path)
from trojanzoo.utils.model import AverageMeter
from trojanzoo.attack.backdoor.badnet import BadNet
from trojanzoo.utils.output import prints
from trojanzoo.utils import to_tensor
import random
from typing import Union, List

import os
import torch
import torch.nn as nn


class HiddenBackdoor(BadNet):

    name = 'hiddenbackdoor'

    def __init__(self, poisoned_image_num: int = 100, poison_generation_iteration: int = 5000, poison_lr: float = 0.01,preprocess_layer: str = 'features', epsilon: int = 16, decay: bool = False, decay_iteration: int = 2000, decay_ratio: float = 0.95, **kwargs):
        """
        HiddenBackdoor attack is different with trojan nn(References: https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech),the mark and mask is designated and stable, we continue these used in paper(References: https://arxiv.org/abs/1910.00033).

        :param poisoned_image_num: the number of poisoned images, defaults to 100
        :type poisoned_image_num: int, optional
        :param poison_generation_iteration: the iteration times used to generate one poison image, defaults to 5000
        :type poison_generation_iteration: int, optional
        :param poison_lr: the learning rate used for generating poisoned images, defaults to 0.01
        :type poison_lr: float, optional
        :param preprocess_layer: the chosen specific layer that on which the feature space of source images patched by trigger is close to poisoned images, defaults to 'feature'
        :type preprocess_layer: str, optional
        :param epsilon: the threshold in pixel space to ensure the poisoned image is not visually distinguishable from the target image, defaults to 16
        :type epsilon: int, optional
        :param decay: specify whether the learning rate decays with iteraion times, defaults to False
        :type decay: bool, optional
        :param decay_iteration: specify the number of iteration time interval, the learning rate will decays once, defaults to 2000
        :type decay_iteration: int, optional
        :param decay_ratio: specify the learning rate decay proportion, defaults to 0.95
        :type decay_ratio: float, optional
        """
        super().__init__(**kwargs)
        self.poisoned_image_num = poisoned_image_num
        self.poison_generation_iteration = poison_generation_iteration
        self.poison_lr = poison_lr
        self.preprocess_layer = preprocess_layer
        self.epsilon = epsilon
        self.decay = decay
        self.decay_iteration = decay_iteration
        self.decay_ratio = decay_ratio

        self.percent = float(self.poisoned_image_num /self.dataset.get_full_dataset('train').__len__())  # update self.percent according to self.poisoned_image_num
        prints("The percent of poisoned image:{}".format(self.percent), indent=self.indent)


    def adjust_lr(self, iteration, decay: bool = False, decay_ratio: float = None, decay_iteration: int = None) -> (float):
        """
        In the process of generating poisoned inputs, the learning rate will change with the iteration times.
        :param iteration: the number of iteration in the process of generating poisoned image
        :type iteration: int, optional
        :param decay: specify whether the learning rate decays with iteraion times, defaults to False
        :type decay: bool, optional
        :param decay_ratio: specify the learning rate decay proportion, defaults to 0.95
        :type decay_ratio: float, optional
        :param decay_iteration: specify the number of iteration time interval, the learning rate will decays once, defaults to 2000
        :type decay_iteration: int, optional
        :return: lr or self.poison_lr: the computed learning rate
        :rtype: float
        """
        if decay is None:
            decay = self.decay
        if decay_ratio is None:
            decay_ratio = self.decay_ratio
        if decay_iteration is None:
            decay_iteration = self.decay_iteration

        if decay:
            lr = self.poison_lr
            lr = lr * (decay_ratio**(iteration // decay_iteration))
            return lr
        else:
            return self.poison_lr

    def generate_poisoned_image(self, source_image: torch.utils.data.Subset, target_image: torch.utils.data.Subset,preprocess_layer: str = None, poison_generation_iteration: int = None, epsilon: int = None,decay: bool = None,decay_ratio: float = None, decay_iteration: int = None, **kwargs) -> (torch.Tensor):
        """
        According to the sampled target images and the sampled source images patched by the trigger ,modify the target inputs to generate poison inputs ,that is close to inputs of target category in pixel space and also close to source inputs patched by the trigger in feature space.
        :param source_image: self.poisoned_image_num source images, other than target category, sampled from train dataset
        :type source_image: torch.Tensor, torch.LongTensor, optional
        :param target_image: self.poisoned_image_num target images sampled from the images of target category in train dataset
        :type target_image: torch.Tensor, torch.LongTensor, optional
        :param preprocess_layer: the chosen specific layer that on which the feature space of source images patched by trigger is close to poisoned images
        :type preprocess_layer: str, optional
        :param epsilon: the threshold in pixel space to ensure the poisoned image is not visually distinguishable from the target image
        :type epsilon: int, optional
        :param decay: specify whether the learning rate decays with iteraion times
        :type decay: bool, optional
        :param decay_ratio: specify the learning rate decay proportion
        :type decay_ratio: float, optional
        :param decay_iteration: specify the number of iteration time interval, the learning rate will decays once
        :type decay_iteration: int, optional
        :return: generated_poisoned_input: the generated poisoned inputs
        :rtype: torch.Tensor
        """

        if preprocess_layer is None:
            preprocess_layer = self.preprocess_layer
        if poison_generation_iteration is None:
            poison_generation_iteration = self.poison_generation_iteration
        if epsilon is None:
            epsilon = self.epsilon
        if decay is None:
            decay = self.decay
        if decay_ratio is None:
            decay_ratio = self.decay_ratio
        if decay_iteration is None:
            decay_iteration = self.decay_iteration
        losses = AverageMeter(name = 'poison_losses')
        source_input = []
        source_label = []
        target_input = []
        target_label = []
        for i in range(self.poisoned_image_num):
            source_input.append(list(source_image[i])[0])
            source_label.append(list(source_image[i])[1])
            target_input.append(list(target_image[i])[0])
            target_label.append(list(target_image[i])[1])
        source_input = to_tensor(source_input)
        source_label = to_tensor(source_label)
        target_input = to_tensor(target_input)
        target_label = to_tensor(target_label)
        source_input.requires_grad = True
        target_input.requires_grad = True
        generated_poisoned_input = torch.zeros_like(source_input).to(source_input.device)

        pert = torch.zeros_like(target_input,requires_grad=True).to(source_input.device)
        source_input = self.add_mark(source_input)
        # source_input.requires_grad = True
        # target_input.requires_grad = True
        feat1 = to_tensor(
            self.model.get_layer(
                source_input, layer_output=preprocess_layer)).detach().clone()
        for j in range(poison_generation_iteration):
            feat2 = to_tensor(
                self.model.get_layer(
                    target_input + pert,
                    layer_output=preprocess_layer)).detach().clone()
            # feat2.requires_grad = True
            # feat1.requires_grad = True
            
            feat11 = feat1.clone()
            feat11.requires_grad = True
            dist = torch.cdist(feat1, feat2)
            for _ in range(feat2.size(0)):
                dist_min_index = (dist == torch.min(dist)).nonzero().squeeze()
                feat1[dist_min_index[1]] = feat11[dist_min_index[0]]
                dist[dist_min_index[0], dist_min_index[1]] = 1e5

            loss1 = ((feat1 - feat2)**2).sum(dim=1) #  Decrease the distance between sourced images patched by trigger and target images
            # loss1.requires_grad = True
            loss = loss1.sum()
            
            losses.update(loss.item(), source_input.size(0))
            # loss.requires_grad = True
            loss.backward()
           
            

            
            
            lr = self.adjust_lr(iteration=j,
                                decay=decay,
                                decay_ratio=decay_ratio,
                                decay_iteration=decay_iteration)
            
            pert = pert - lr * pert.grad  # pert.grad or 1
            
            pert = torch.clamp(pert, - epsilon / 255.0, epsilon / 255.0).detach_()
            pert = pert + target_input
            pert = pert.clamp(0, 1)  # restrict the pixel value range resonable
            if j % 100 == 0:
                print(" i: {} | iter: {:5d} | LR: {:2.4f} | Loss Val: {:5.3f} | Loss Avg: {:5.3f}".format(i, j, lr, losses.val, losses.avg))
            if loss1.max().item() < 10 or j == (poison_generation_iteration -1):
                for k in range(target_input.size(0)):
                    input2_pert = (pert[k].clone())
                    generated_poisoned_input[k] = input2_pert
                break
            pert = pert - target_input
            pert.requires_grad = True
        return generated_poisoned_input

    def get_data(self, data: (torch.Tensor, torch.LongTensor), keep_org: bool = True, **kwargs) -> (torch.Tensor, torch.LongTensor):
        """
        When keep_org= True, get the normal inputs and labels.
        When keep_org= False, get the normal inputs and labels and poisoned inputs and their labels.
        :param data: the original input and label
        :type data: torch.Tensor, torch.LongTensor
        :param keep_orig: specify whether to insert poisoned inputs and labels, defaults to True
        :type keep_orig: bool, optional
        :return: _input, _label
        :rtype: torch.Tensor, torch.LongTensor
        """
        

        _input, _label = self.model.get_data(data)
        if not keep_org or random.uniform(0, 1) < self.percent:
            org_input, org_label = _input, _label
            source_image =  self.dataset.get_class_set(self.dataset.get_full_dataset('train'), self.target_class)
            target_image =  self.dataset.get_non_class_set(self.dataset.get_full_dataset('train'), self.target_class)
            _input = self.generate_poisoned_image(source_image, target_image)
            
            _label = self.target_class * torch.ones_like(org_label)

            if keep_org:
                _input = torch.cat((_input, org_input))
                _label = torch.cat((_label, org_label))
        return _input, _label
    

    




        
        
    