import torch
from .base_model import BaseModel
from . import networks
from models.guided_filter_pytorch.HFC_filter import HFCFilter
from util import metrics


def hfc_mul_mask(hfc_filter, image, mask):
    hfc = hfc_filter(image, mask)
    return (hfc + 1) * mask - 1
    # return image


class HGCNETModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='unet_cascade', no_dropout=True, lr=0.001, repeat_size=10)
        if is_train:
            parser.add_argument('--lambda_seg', type=float, default=1.0)
            parser.add_argument('--lambda_high', type=float, default=1.0)
            parser.add_argument('--segmentation_loss', type=str, default='BCELoss')
            parser.add_argument('--high_loss', type=str, default='L1Loss')
            parser.add_argument('--no_fda', action='store_true')
        else:
            parser.add_argument('--metrics', type=str, default='f1,acc', )
            parser.add_argument('--confusion_threshold', type=float, default=0.5)
        parser.add_argument('--filter_width', type=int, default=27, help='weight for G loss')
        parser.add_argument('--nsig', type=int, default=9, help='weight for G loss')

        parser.add_argument('--original_input', action='store_true', help='do not do hfc before inputting into network')

        parser.add_argument('--original_dense', action='store_true')
        parser.add_argument('--no_high_loss', action='store_true')
        parser.add_argument('--no_fact', action='store_true', help='useful only if no high loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.no_high_loss = opt.no_high_loss
        self.no_fact = opt.no_fact
        self.original_input = opt.original_input

        if self.no_high_loss:
            self.loss_names = ['G', 'G_seg']
        else:
            self.loss_names = ['G', 'G_seg', 'G_high']

        self.visual_names_train = ['image_original', 'image_fact', 'target', 'label', 'out_seg', 'mask']
        self.visual_names_test = ['image_original', 'label', 'out_seg', 'out_seg_binary']
        if not self.original_input:
            self.visual_names_train += ['high_fact']
            self.visual_names_test += ['high_original']
        if not self.no_high_loss:
            self.visual_names_train += ['high_original', 'out_high']
            self.visual_names_test += ['out_high']

        if self.isTrain:
            self.model_names = ['G']
            self.visual_names = self.visual_names_train
            self.no_fda = opt.no_fda
        else:
            self.model_names = ['G']
            self.visual_names = self.visual_names_test
            self.confusion_matrix = metrics.Metric(opt.output_nc, threshold=opt.confusion_threshold)

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      last_layer='Sigmoid', verbose=opt.verbose, original_dense=opt.original_dense)

        self.hfc_filter = HFCFilter(opt.filter_width, opt.nsig, sub_low_ratio=1, sub_mask=True, is_clamp=True).to(
            self.device)

        if self.isTrain:
            # define loss functions
            self.criterion_segmentation = getattr(torch.nn, opt.segmentation_loss)()
            self.criterion_high = getattr(torch.nn, opt.high_loss)()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input, isTrain=None):
        """
        set the input
        """
        self.image_paths = input['source_path']

        if not self.isTrain or isTrain is not None:
            self.image_original = input['image_original'].to(self.device)
            self.mask = input['mask'].to(self.device)
            self.label = input['label'].to(self.device)
            if not self.original_input:
                self.high_original = hfc_mul_mask(self.hfc_filter, self.image_original, self.mask)
        else:
            self.image_original = input['image_original'].to(self.device)
            self.image_fact = input['image_fact'].to(self.device)
            self.label = input['label'].to(self.device)
            self.mask = input['mask'].to(self.device)
            self.target = input['target'].to(self.device)
            self.high_original = hfc_mul_mask(self.hfc_filter, self.image_original, self.mask)
            if not self.original_input:
                self.high_fact = hfc_mul_mask(self.hfc_filter, self.image_fact, self.mask)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.original_input:
            self.out_high, self.out_seg = self.netG(self.image_fact if not self.no_fact else self.image_original)
        else:
            self.out_high, self.out_seg = self.netG(self.high_fact if not self.no_fact else self.high_original)
        self.out_high = (self.out_high + 1) * self.mask - 1
        self.out_seg = self.out_seg * self.mask

    def compute_visuals(self):
        self.label = self.label * 2 - 1
        self.out_seg = self.out_seg * 2 - 1
        self.mask = self.mask * 2 - 1
        if self.isTrain:
            self.target = self.target * 2 - 1
        if not self.isTrain:
            self.out_seg_binary = self.out_seg_binary * 2 - 1

    def test(self):
        self.visual_names = self.visual_names_test
        with torch.no_grad():
            # For visualisation
            if self.original_input:
                self.out_high, self.out_seg = self.netG(self.image_original)
            else:
                self.out_high, self.out_seg = self.netG(self.high_original)

            self.out_high = (self.out_high + 1) * self.mask - 1
            self.out_seg = self.out_seg * self.mask

            self.confusion_matrix.update(self.out_seg, self.label)

            self.out_seg_binary = self.out_seg > 0.5

            self.compute_visuals()

    def train(self):
        self.visual_names = self.visual_names_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def backward_G(self):
        self.loss_G_seg = self.criterion_segmentation(self.out_seg, self.label) * self.opt.lambda_seg

        if not self.no_high_loss:
            self.loss_G_high = self.criterion_high(self.out_high, self.high_original) * self.opt.lambda_high

            self.loss_G = self.loss_G_seg + self.loss_G_high
        else:
            self.loss_G = self.loss_G_seg
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # generate prediction mask and intermediate high frequency image
        self.optimizer_G.zero_grad() # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights

    def get_metric_results(self):
        results = self.confusion_matrix.evalutate()
        metrics_list = self.opt.metrics.split(',')
        # return {name: results[name].item() for name in metrics_list}
        return {name: results[name][1].item() for name in metrics_list}
