# -*- coding: UTF-8 -*-
"""
@Function:from two-stage to one-stage
@File: DG_one_model.py
@Date: 2021/9/14 20:45 
@Author: Hever
"""
import torch
from .base_model import BaseModel
from . import baseline_methods
from util import metrics
import os
from .guided_filter_pytorch.HFC_filter import HFCFilter


def hfc_mul_mask(hfc_filter, image, mask):
    hfc = hfc_filter(image, mask)
    # return hfc
    return (hfc + 1) * mask - 1
    # return image


class BASELINEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--segmentation_loss', type=str, default='BCELoss')
            parser.add_argument('--smooth_factor', type=float, default=0.1,
                                help='the rolling average factor for visualization smoothing.')
        else:
            parser.add_argument('--metrics', type=str, default='f1,acc', )
            parser.add_argument('--confusion_threshold', type=float, default=0.5)

        parser.add_argument('--baseline_name', type=str, required=True)

        parser.add_argument('--filter_width', type=int, default=27, help='weight for G loss')
        parser.add_argument('--nsig', type=int, default=9, help='weight for G loss')

        parser.add_argument('--do_hfc', action='store_true', help='do hfc before inputting into network')
        parser.add_argument('--no_fact', action='store_true')

        parser.add_argument('--deep_super', action='store_true')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_smooth']

        self.no_fact = opt.no_fact

        self.do_hfc = opt.do_hfc
        if self.do_hfc:
            self.hfc_filter = HFCFilter(opt.filter_width, opt.nsig, sub_low_ratio=1, sub_mask=True, is_clamp=True).to(
                self.device)
            self.visual_names_train = ['image_original', 'label', 'out_seg', 'mask']
            if not self.no_fact:
                self.visual_names_train += ['image_fact', 'high_fact']
            self.visual_names_test = ['image_original', 'high_original', 'label', 'out_seg', 'out_seg_binary']
        else:
            self.visual_names_train = ['image_original', 'label', 'out_seg', 'mask']
            if not self.no_fact:
                self.visual_names_train += ['image_fact']
            self.visual_names_test = ['image_original', 'label', 'out_seg', 'out_seg_binary']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
            self.visual_names = self.visual_names_train
            self.smooth_factor = opt.smooth_factor
        else:
            self.model_names = ['G']
            self.visual_names = self.visual_names_test
            self.confusion_matrix = metrics.Metric(opt.output_nc, threshold=opt.confusion_threshold)

        # define networks (both generator and discriminator)
        self.netG = baseline_methods.find_model_using_name(opt.baseline_name)(3, 1).to(device=self.device)

        self.deep_super = opt.deep_super

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            # define loss functions
            self.criterion_segmentation = getattr(torch.nn, opt.segmentation_loss)()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.loss_G_smooth = None

    def set_input(self, input, isTrain=None):
        """
        set the input
        """
        self.image_paths = input['source_path']

        if not self.isTrain or isTrain is not None:
            self.image_original = input['image_original'].to(self.device)
            self.mask = input['mask'].to(self.device)
            self.label = input['label'].to(self.device)
            if self.do_hfc:
                self.high_original = hfc_mul_mask(self.hfc_filter, self.image_original, self.mask)
        else:
            self.image_original = input['image_original'].to(self.device)
            if not self.no_fact:
                self.image_fact = input['image_fact'].to(self.device)
                if self.do_hfc:
                    self.high_fact = hfc_mul_mask(self.hfc_filter,
                                                  self.image_original if self.no_fact else self.image_fact, self.mask)
            elif self.do_hfc:
                self.high_original = hfc_mul_mask(self.hfc_filter, self.image_original, self.mask)
            self.label = input['label'].to(self.device)
            self.mask = input['mask'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.do_hfc:
            self.out_seg = self.netG(self.high_original if self.no_fact else self.high_fact)
        else:
            self.out_seg = self.netG(self.image_original if self.no_fact else self.image_fact)

        if self.deep_super:
            self.out_seg = [torch.sigmoid(x) * self.mask for x in self.out_seg]
        else:
            self.out_seg = torch.sigmoid(self.out_seg) * self.mask

    def compute_visuals(self):
        self.label = self.label * 2 - 1
        if self.deep_super:
            self.out_seg = self.out_seg[0] * 2 - 1
        else:
            self.out_seg = self.out_seg * 2 - 1
        self.mask = self.mask * 2 - 1
        if not self.isTrain:
            if self.deep_super:
                self.out_seg_binary = self.out_seg_binary[0] * 2 - 1
            else:
                self.out_seg_binary = self.out_seg_binary * 2 - 1

    def test(self):
        self.visual_names = self.visual_names_test
        with torch.no_grad():
            # For visualisation
            if self.do_hfc:
                self.out_seg = self.netG(self.high_original)
            else:
                self.out_seg = self.netG(self.image_original)

            if self.deep_super:
                self.out_seg = [torch.sigmoid(x) * self.mask for x in self.out_seg]
                self.confusion_matrix.update(self.out_seg[0], self.label)
            else:
                self.out_seg = torch.sigmoid(self.out_seg) * self.mask
                self.confusion_matrix.update(self.out_seg, self.label)

            if self.deep_super:
                self.out_seg_binary = [x > 0.5 for x in self.out_seg]
            else:
                self.out_seg_binary = self.out_seg > 0.5

            self.compute_visuals()

    def train(self):
        """Make models eval mode during test time"""
        self.visual_names = self.visual_names_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def backward_G(self):
        # print(self.out_seg.shape, self.label.shape)

        # LR
        if self.deep_super:
            self.loss_G = sum([self.criterion_segmentation(x, self.label) for x in self.out_seg])
        else:
            self.loss_G = self.criterion_segmentation(self.out_seg, self.label)
        self.loss_G.backward()

        if self.loss_G_smooth is None:
            self.loss_G_smooth = self.loss_G.clone().detach()
        else:
            self.loss_G_smooth = self.loss_G_smooth * (
                    1 - self.smooth_factor) + self.loss_G.clone().detach() * self.smooth_factor

    def optimize_parameters(self):
        # self.set_requires_grad([self.netG], True)  # D requires no gradients when optimizing G
        self.forward()  # compute fake images: G(A)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def get_metric_results(self):
        results = self.confusion_matrix.evalutate()
        metrics_list = self.opt.metrics.split(',')
        # return {name: results[name].item() for name in metrics_list}
        return {name: results[name][1].item() for name in metrics_list}

    def save_networks(self, prefix):
        save_filename = '%s_net_%s.pth' % (prefix, self.opt.name)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.netG.cpu().state_dict(), save_path)
        self.netG.to(device=self.device)

    def load_networks(self, prefix, do_print=True):
        load_filename = '%s_net_%s.pth' % (prefix, self.opt.name)
        load_path = os.path.join(self.save_dir, load_filename)
        if do_print:
            print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        self.netG.load_state_dict(state_dict)
