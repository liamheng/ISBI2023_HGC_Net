import os.path
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel, get_transform_four
from PIL import Image
from torchvision import transforms


class NOFDADataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.data_root = opt.dataroot

        assert (os.path.exists(self.data_root))

        self.len = len(os.listdir(os.path.join(self.data_root, 'original')))

        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image

        self.input_nc = 3
        self.output_nc = 1
        self.isTrain = opt.isTrain

        # if 'crop' not in opt.preprocess:
        #     opt.preprocess += ',crop'

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """

        image_path = os.path.join(self.data_root, 'original', str(index), 'image.png')
        label_path = os.path.join(self.data_root, 'original', str(index), 'label.png')
        mask_path = os.path.join(self.data_root, 'original', str(index), 'mask.png')

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        transform_params = get_params(self.opt, image.size)
        # original_transform, fact_transform, mask_transform, label_transform = \
        #         get_transform_four(self.opt, transform_params, grayscale=(self.input_nc == 1))
        raw_transform, label_transform = get_transform_six_channel(self.opt, transform_params, grayscale=False, do_norm=False)

        image = raw_transform(image)
        mask = label_transform(mask)
        label = label_transform(label)

        norm_func = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        image = norm_func(image)

        return {'image_original': image, 'mask': mask, 'source_path': image_path, 'label': label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.len
