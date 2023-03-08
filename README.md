# Self-supervision Boosted Retinal Vessel Segmentation for Cross-domain Data (ISBI 2023)

This repository contains the code for the paper Self-supervision Boosted Retinal Vessel Segmentation for Cross-domain Data (ISBI 2022).

![architecture.png](images%2Farchitecture.png)

## Datasets

The training set DRIVE is available at [DRIVE](https://drive.grand-challenge.org/).
The first test set CHASEDB1 is available at [CHASEDB1](https://blogs.kingston.ac.uk/retinal/chasedb1/).
The second test set AVRDB is available at [AVRDB](http://biomisa.org/index.php/dataset-for-hypertensive-retinopathy/).
All of the above datasets should be organized in the following structure:

```
<root_name>
- 0
    - image.png
    - label.png
    - mask.png
- 1
- 2
...
```

where the `image.png` is the original fundus color image, `label.png` is the ground truth of vessel segmentation, and `mask.png` is the FoV mask, usually a solid circle.

The FACT augmentation target dataset EYEPACS is available at [EYEPACS](https://www.kaggle.com/c/diabetic-retinopathy-detection). All fundus images in the target set must be in the same directory, the file name does not need to be specified.

## Dependencies

* torch>=0.4.1
* torchvision>=0.2.1
* dominate>=2.3.1
* visdom>=0.1.8.3

## Training

Before running the training code, you need to start the visdom server by running the following command:

```
python -m visdom.server -p <some free port> --host 0.0.0.0
```

The training code is in `train.py`. The training command is as follows:

```
python ISBI_HGC_Net\procedure_main_method\train.py --model hgcnet --input_nc 3 --output_nc 1 --original_dense --dataset_mode online_fact --dataroot <your training set directory> --target_root <your augmentation target set directory> --preprocess rotate_crop --load_size 565 --crop_size 512 --lr 0.001 --n_epochs 80 --n_epochs_decay 120 --display_port <visdom port> --print_freq 10 --display_freq 40 --save_epoch_freq 30 --repeat_size 10 --name hgcnet --time_suffix --batch_size 2 --gpu_ids <gpu ids> 
```

## Acknowledgement

This work was supported in part by Basic and Applied Fundamental Research Foundation of Guangdong Province (2020A1515110286), The National Natural Science Foundation of China (8210072776), Guangdong Provincial Department of Education (2020ZDZX3043), Guangdong Provincial Key Laboratory (2020B121201001), and Shenzhen Natural Science Fund (JCYJ20200109140820699, 20200925174052004).

## Citation

If you find this work useful for your research, please cite our paper:

```
@inproceedings{li2023self,
  title={Self-supervision Boosted Retinal Vessel Segmentation for Cross-domain Data},
  author={Li, Haojin and Li, Heng and Shu, Hai and Chen, Jianyu and Hu, Yan and Jiang, Liu},
  booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
