import os.path
import random

from data.base_dataset import BaseDataset, get_params, get_transform_six_channel
from PIL import Image
from util import fda
from torchvision import transforms


class ONLINEFACTDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--target_root', type=str, help='the directory of the target dataset used for FACT')
        parser.add_argument('--fact_l_lower', type=float, default=0.4)
        parser.add_argument('--fact_l_upper', type=float, default=0.8)
        parser.add_argument('--fact_mode', type=str, default='am')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.data_root = opt.dataroot
        self.target_root = opt.target_root

        assert (os.path.exists(self.data_root))
        assert (os.path.exists(self.target_root))

        self.len = len(os.listdir(self.data_root))

        self.target_list = os.listdir(self.target_root)

        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image

        self.input_nc = 3
        self.output_nc = 1
        self.isTrain = opt.isTrain

        # if 'crop' not in opt.preprocess:
        #     opt.preprocess += ',crop'

        self.fda_module = fda.FDAModule(opt.fact_mode)
        self.l_lower_bound = opt.fact_l_lower
        self.l_upper_bound = opt.fact_l_upper

        assert(0 <= self.l_lower_bound <= self.l_upper_bound <= 1)

    def __getitem__(self, index):

        image_path = os.path.join(self.data_root, str(index), 'image.png')
        label_path = os.path.join(self.data_root, str(index), 'label.png')
        mask_path = os.path.join(self.data_root, str(index), 'mask.png')
        target_path = os.path.join(self.target_root, random.choice(self.target_list))

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        target = Image.open(target_path).convert('RGB')

        transform_params = get_params(self.opt, image.size)
        raw_transform, label_transform = get_transform_six_channel(self.opt, transform_params, grayscale=False, do_norm=False)

        image = raw_transform(image)
        mask = label_transform(mask)
        label = label_transform(label)
        target = transforms.ToTensor()(target)

        random_l = random.random() * (self.l_upper_bound - self.l_lower_bound) + self.l_lower_bound
        fact = self.fda_module(image, target, random_l)

        norm_func = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        image = norm_func(image)
        fact = norm_func(fact) * mask + mask - 1

        return {'image_original': image, 'image_fact': fact, 'mask': mask,
                'source_path': image_path, 'label': label, 'target': target}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.len
