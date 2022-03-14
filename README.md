# Active Learning by Feature Mixing
PyTorch implementation of ALFA-Mix. For details, read the paper Active Learning by Feature Mixing, which is accepted in CVPR 2022.

<div align="center">
  <img width="45%" alt="ALFA-Mix" src="Img1.png">
</div>

The code includes the implementations of all the baselines presented in the paper. Parts of the code are borrowed from https://github.com/JordanAsh/badge.

## Setup
The dependencies are in [`requirements.txt`](requirements.txt). Python=3.8.3 is recommended for the installation of the environment.


## Datasets
The code supports torchvision built-in implementations of MNIST, EMNIST, SVHN, CIFAR10 and CIFAR100, in addition to MiniImageNet, DomainNet-Real (and two subsets of that) and openml datasets.

## Training
For running an AL strategy in a single setting, use the following script that by default uses 5 different initial random seeds for the specified setting. 
```python
python main.py --data_name MNIST --data_dir your_data_directory --n_init_lb 100 --n_query 100 --n_round 15 --learning_rate 0.001 --n_epoch 1000 --model mlp --strategy AlphaMixSampling --alpha_opt --alpha_closed_form_approx --alpha_cap 0.2
```

To aggregate all the results accross various settings and create the comparison matrix, use the following script:
```python
python agg_results.py --dir_type general 
```