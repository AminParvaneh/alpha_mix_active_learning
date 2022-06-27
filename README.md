# Active Learning by Feature Mixing (ALFA-Mix)
PyTorch implementation of ALFA-Mix. For details, read the paper [Active Learning by Feature Mixing](https://arxiv.org/abs/2203.07034), which is accepted in CVPR 2022.

<div align="center">
  <img width="45%" alt="ALFA-Mix" src="Img1.png">
</div>

The code includes the implementations of all the baselines presented in the paper. Parts of the code are borrowed from https://github.com/JordanAsh/badge.

## Setup
The dependencies are in [`requirements.txt`](requirements.txt). Python=3.8.3 is recommended for the installation of the environment.


## Datasets
The code supports torchvision built-in implementations of MNIST, EMNIST, SVHN, CIFAR10 and CIFAR100.
Additionally, it supports [https://www.kaggle.com/datasets/whitemoon/miniimagenet?select=mini-imagenet-cache-val.pkl](MiniImageNet), [http://ai.bu.edu/M3SDA/](DomainNet-Real) (and two subsets of that) and [https://www.openml.org/](openml) datasets.

## Training
For running an AL strategy in a single setting, use the following script that by default uses 5 different initial random seeds for the specified setting. 
```python
python main.py --data_name MNIST --data_dir your_data_directory --n_init_lb 100 --n_query 100 --n_round 10 --learning_rate 0.001 --n_epoch 1000 --model mlp \
               --strategy AlphaMixSampling --alpha_opt --alpha_closed_form_approx --alpha_cap 0.2
```

## Evaluation
To evaluate over all the experiments and get the final comparison matrix, put all the results in a folder with this structure: Overall/Dataset/Setting (e.g. Overall/MNIST/MLP_small_budget).
The script below explores all the settings in each dataset and acculumulate resulst to generate the overall comparison matrix:
```python
python agg_results.py --directory Path/Overall --dir_type general 
```
By setting 'dir_type' to 'dataset' or 'setting' you can evaluate the results at dataset or setting level respectively.

## Citing
```
@inproceedings{parvaneh2022active,
  title={Active Learning by Feature Mixing},
  author={Parvaneh, Amin and Abbasnejad, Ehsan and Teney, Damien and Haffari, Gholamreza Reza and van den Hengel, Anton and Shi, Javen Qinfeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12237--12246},
  year={2022}
}
```