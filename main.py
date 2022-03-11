import argparse
import json
import os
import csv
import time
import math

import numpy as np
from dataset import get_dataset, get_handler, is_openml
from models.model import get_net, MLPNet
from sklearn.manifold import TSNE
from torchvision import transforms
import torch
from torch.utils.tensorboard import SummaryWriter
from query_strategies import *
from models.lenet import LeNet5
from models.resnet import ResNetClassifier
from models.vgg import VGGClassifier
from models.densenet import DenseNetClassifier
from models.vision_transformer import VisionTransformerClassifier
import matplotlib.pyplot as plt
import sys
from models.training import Training
from models.cdal_model import CDALModel
import matplotlib
import matplotlib.colors as mcolors


font_size = 25
font = {'family' : 'serif',
        'size'   : font_size}

matplotlib.rc('font', **font)


ALL_STRATEGIES = [
    'RandomSampling',
    'EntropySampling',
    'BALDDropout',
    'CoreSet',
    'AdversarialDeepFool',
    'BadgeSampling',
    'CDALSampling',
    'GCNSampling'
]


def supervised_learning(args):
    # supervised training args
    train_parser = argparse.ArgumentParser(description="Training parser for training hyper-parrameters at ech checkpoint.")

    train_parser.add_argument('--data_augmentation', action='store_const', default=False, const=True)
    train_parser.add_argument('--n_epoch', type=int, default=200)
    train_parser.add_argument('--n_early_stopping', type=int, default=200)
    train_parser.add_argument('--optimizer', type=str, default='Adam')
    train_parser.add_argument('--batch_size', type=int, default=64)
    train_parser.add_argument('--learning_rate', type=float, default=0.01)
    train_parser.add_argument('--momentum', type=float, default=0.9)
    train_parser.add_argument('--weight_decay', type=float, default=0.0005)
    train_parser.add_argument('--lr_warmup', type=int, default=0)
    train_parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=None)
    train_parser.add_argument('--lr_schedule', action="store_const", default=False, const=True)
    train_parser.add_argument('--lr_T_0', type=int, default=200)
    train_parser.add_argument('--lr_T_mult', type=int, default=1)
    train_parser.add_argument('--train_to_end', action="store_const", default=False, const=True)
    train_parser.add_argument('--n_validation_set', type=int, default=0)
    train_parser.add_argument('--choose_best_val_model', action="store_const", default=False, const=True)

    train_parser.add_argument('--model', type=str, default='resnet50')
    train_parser.add_argument('--emb_size', type=int, default=256)
    train_parser.add_argument('--dropout', type=float, default=0)
    train_parser.add_argument('--fine_tune_layers', type=int, default=1)
    train_parser.add_argument('--pretrained_model', action='store_const', default=False, const=True)
    train_parser.add_argument('--continue_training', action='store_const', default=False, const=True)

    train_parser.add_argument('--vit_patch_size', type=int, default=16)
    train_parser.add_argument('--vit_n_last_blocks', type=int, default=4)
    train_parser.add_argument('--vit_avgpool_patchtokens', action='store_const', default=False, const=True)
    train_parser.add_argument('--vit_pretrained_weights', type=str, default='./pretrained_models/dino_vitbase16_pretrain.pth')

    train_args, _ = train_parser.parse_known_args()

    train_params_pool = {
        'MNIST':
            {'n_epoch': train_args.n_epoch,
             'n_training_set': 60000,
             'n_label': 10,
             'image_size': 28,
             'in_channels': 1,
             'transform': transforms.Compose(
                 [
                     #transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))
                 ]),
             'test_transform': transforms.Compose(
                 [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/MNIST',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training
             },
        'EMNIST':
            {'n_epoch': train_args.n_epoch,
             'n_training_set': 130000,
             'n_label': 26,
             'image_size': 28,
             'in_channels': 1,
             'transform': transforms.Compose(
                 [
                     #transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
             'test_transform': transforms.Compose(
                 [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/EMNIST',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training
             },
        'SVHN':
            {'n_epoch': train_args.n_epoch,
             'n_label': 10,
             'n_training_set': 60000,
             'image_size': 32,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     #transforms.RandomCrop(32, padding=4),
                     #transforms.RandomHorizontalFlip(),
                     #transforms.RandomRotation(60),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                 ]
             ),
             'test_transform': transforms.Compose(
                 [
                     transforms.ToTensor(),
                     transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                 ]
             ),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/SVHN',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'CIFAR10':
            {'n_epoch': train_args.n_epoch,
             'n_label': 10,
             'n_training_set': 60000,
             'image_size': 32,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 ] if train_args.data_augmentation else
                 [
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 ]
             ),
             'test_transform': transforms.Compose(
                 [
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 ]
             ),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             # 'optimizer_args': {'lr': 0.05, 'momentum': 0.9},
             'optimizer_args': {'lr': train_args.learning_rate},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/CIFAR10',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'CIFAR100':
            {'n_epoch': train_args.n_epoch,
             'n_label': 100,
             'n_training_set': 50000,
             'image_size': 32,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 ] if train_args.data_augmentation else
                 [
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 ]),
             'test_transform': transforms.Compose(
                 [
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                 ]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/CIFAR100',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'MiniImageNet':
            {'n_epoch': train_args.n_epoch,
             'n_label': 100,
             'n_training_set': 50000,
             'image_size': 84,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     #transforms.Resize((224, 224)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]
                 if train_args.data_augmentation else
                 [
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'test_transform': transforms.Compose(
                 [
                     #transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},    #, 'weight_decay': 5e-4},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/MiniImageNet',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'domain_net-real':
            {'n_epoch': train_args.n_epoch,
             'n_label': 345,
             'n_training_set': 130000,
             'image_size': 224,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     transforms.Resize((224, 224)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]
                 if train_args.data_augmentation else
                 [
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'test_transform': transforms.Compose(
                 [
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},    #, 'weight_decay': 5e-4},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/domain_net-real',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'mini_domain_net-real':
            {'n_epoch': train_args.n_epoch,
             'n_label': 20,
             'n_training_set': 8286,
             'image_size': 224,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     transforms.Resize((224, 224)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]
                 if train_args.data_augmentation else
                 [
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'test_transform': transforms.Compose(
                 [
                     #transforms.Resize((84, 84)),
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},    #, 'weight_decay': 5e-4},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/mini_domain_net-real',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'tiny_domain_net-real':
            {'n_epoch': train_args.n_epoch,
             'n_label': 10,
             'n_training_set': 5000,
             'image_size': 224,
             'in_channels': 3,
             'transform': transforms.Compose(
                 [
                     transforms.Resize((224, 224)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]
                 if train_args.data_augmentation else
                 [
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'test_transform': transforms.Compose(
                 [
                     #transforms.Resize((84, 84)),
                     transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 ]),
             'loader_tr_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'loader_te_args': {'batch_size': train_args.batch_size, 'num_workers': 10},
             'optimizer_args': {'lr': train_args.learning_rate},    #, 'weight_decay': 5e-4},
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'log_dir': './logs/mini_domain_net-real',
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training},
        'openml':
            {'n_epoch': train_args.n_epoch,
             'n_training_set': 50000,
             'in_channels': 1,
             'loader_tr_args': {'batch_size': 128, 'num_workers': 1},
             'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
             'optimizer_args': {'lr': train_args.learning_rate},
             'transform': transforms.Compose([transforms.ToTensor()]),
             'test_transform': transforms.Compose([transforms.ToTensor()]),
             'lr_decay_epochs': train_args.lr_decay_epochs,
             'train_to_end': train_args.train_to_end,
             'n_early_stopping': train_args.n_early_stopping,
             'continue_training': train_args.continue_training}
    }

    if is_openml(args.data_name):
        train_params = train_params_pool['openml']
    else:
        train_params = train_params_pool[args.data_name]

    train_params['optimizer'] = train_args.optimizer
    if train_args.optimizer == 'SGD':
        train_params['optimizer_args']['momentum'] = train_args.momentum
    train_params['lr_warmup'] = train_args.lr_warmup
    train_params['lr_schedule'] = train_args.lr_schedule
    train_params['lr_T_0'] = train_args.lr_T_0
    train_params['lr_T_mult'] = train_args.lr_T_mult
    train_params['choose_best_val_model'] = train_args.choose_best_val_model

    if args.strategy == 'All':
        for strategy in ALL_STRATEGIES:
            al_train(args, train_args, train_params, strategy)
    else:
        al_train(args, train_args, train_params, args.strategy)


def al_train(args, train_args, train_params, strategy_name):
    main_path = os.path.join(args.log_dir, args.data_name)
    if not os.path.exists(main_path):
        os.makedirs(main_path)

    general_path = os.path.join(main_path,
                                'init' + str(args.n_init_lb) + '_query' + str(args.n_query) + '_' + str(args.query_growth_ratio) +
                                '_rounds' + str(args.n_round) + '_' + train_args.model + '_emb' + str(train_args.emb_size) +
                                '_bs' + str(train_args.batch_size) + ('_augmentation' if train_args.data_augmentation else '') +
                                '_epochs' + str(train_args.n_epoch) + ('_full' if train_args.train_to_end else '') +
                                ('_continue' if train_args.continue_training else '') +
                                ('' if train_args.model[:3] != 'vit' else ('_patch' + str(train_args.vit_patch_size) + '_nblock' + str(train_args.vit_n_last_blocks) + ('_avgpool' if train_args.vit_avgpool_patchtokens else ''))) +
                                (('_pretrained' + str(train_args.fine_tune_layers)) if train_args.pretrained_model else '') +
                                ('_lr_schedule' if train_args.lr_schedule else '') +
                                (('_lr_warmup' + str(train_args.lr_warmup)) if train_args.lr_warmup > 0 else '') +
                                '_dropout' + str(train_args.dropout) +
                                '_lr' + str(train_args.learning_rate) +
                                ('' if train_args.lr_decay_epochs is None else ('_lrd' + '-'.join([str(e) for e in train_args.lr_decay_epochs]))) +
                                '_valid' + str(train_args.n_validation_set) +
                                '_es' + str(train_args.n_early_stopping))

    if not os.path.exists(general_path):
        os.makedirs(general_path)

    for seed in args.seeds:
        al_train_sub_experiment(args, train_args, train_params, strategy_name, general_path, seed)


def set_seeds(seed):
    # set seed
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False


def save_args(args, path, name):
    config = vars(args)
    with open(os.path.join(path, name + '.json'), 'w') as f:
        hps = {key: val for key, val in config.items() if not isinstance(val, type)}
        json.dump(hps, f, indent=2)


def al_train_sub_experiment(args, train_args, train_params, strategy_name, general_path, seed):
    exp_name = strategy_name + '_seed' + str(seed)

    sub_path = os.path.join(general_path, exp_name)
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    save_args(args, sub_path, 'args')
    save_args(train_args, sub_path, 'train_args')

    if args.print_to_file:
        orig_stdout = sys.stdout
        log_file = open(os.path.join(sub_path, 'logs.txt'), 'w')
        sys.stdout = log_file

    writer = SummaryWriter(log_dir=sub_path)

    result_file = open(sub_path + '.csv', 'w')
    result_writer = csv.writer(result_file, quoting=csv.QUOTE_ALL)

    # set seed
    set_seeds(seed)

    # load dataset
    X_tr, Y_tr, X_te, Y_te = get_dataset(args.data_name, args.data_dir)

    if is_openml(args.data_name):
        train_params['n_label'] = int(max(Y_tr) + 1)
        train_params['emb_size'] = train_args.emb_size  #1024
        train_params['dim'] = X_tr.shape[1]
    else:
        train_params['emb_size'] = train_args.emb_size  #256
        train_params['dim'] = np.shape(X_tr)[1:]

    args.n_label = train_params['n_label']

    # Generate the validation set
    if train_args.n_validation_set > 0:
        # shuffle the training data
        idxs_tmp = np.arange(len(X_tr))
        np.random.shuffle(idxs_tmp)
        X_tr = X_tr[idxs_tmp]
        Y_tr = Y_tr[idxs_tmp]

        X_val = X_tr[-train_args.n_validation_set:]
        Y_val = Y_tr[-train_args.n_validation_set:]
    else:
        X_val = []
        Y_val = []

    X_tr = X_tr[:min(train_params['n_training_set'], len(X_tr) - len(X_val))]
    Y_tr = Y_tr[:min(train_params['n_training_set'], len(Y_tr) - len(Y_val))]

    # start experiment
    n_pool = len(Y_tr)
    n_test = len(Y_te)

    # generate initial labeled pool
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_tmp = np.arange(n_pool)
    np.random.shuffle(idxs_tmp)

    if args.init_lb_method == 'general_random':
        idxs_lb[idxs_tmp[:args.n_init_lb]] = True
    else:
        for i in range(Y_tr.max().item() + 1):
            idx = (Y_tr == i).nonzero().squeeze()
            idxs_lb[idx[:args.n_init_lb]] = True

    print('number of labeled pool: {}'.format(idxs_lb.sum()))
    print('number of unlabeled pool: {}'.format(n_pool - idxs_lb.sum()))
    print('number of validation pool: {}'.format(len(Y_val)))
    print('number of testing pool: {}'.format(n_test))

    np.save(open(os.path.join(sub_path, 'query_0.np'), 'wb'), idxs_tmp[idxs_lb])

    # load network
    if train_args.model == 'baseline_cnn':
        net = get_net(args.data_name, is_openml(args.data_name))
        net_args = {'n_label': train_params['n_label']}
    elif train_args.model == 'lenet':
        net = LeNet5
        net_args = {'n_label': train_params['n_label']}
    elif train_args.model == 'mlp':
        net = MLPNet
        net_args = {'dim': train_params['dim'],
                    'emb_size': train_params['emb_size'],
                    'n_label': train_params['n_label']}
    elif train_args.model[:3] == 'vgg':
        net = VGGClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'pretrained': train_args.pretrained_model,
                    'fine_tune_layers': train_args.fine_tune_layers,
                    'emb_size': train_params['emb_size'],
                    'in_channels': train_params['in_channels']}
    elif len(train_args.model) >= 8 and train_args.model[:8] == 'densenet':
        net = DenseNetClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'pretrained': train_args.pretrained_model,
                    'fine_tune_layers': train_args.fine_tune_layers,
                    'emb_size': train_params['emb_size'],
                    'in_channels': train_params['in_channels']}
    elif len(train_args.model) >= 8 and train_args.model[:6] == 'resnet':
        net = ResNetClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'pretrained': train_args.pretrained_model,
                    'fine_tune_layers': train_args.fine_tune_layers,
                    'emb_size': train_params['emb_size'],
                    'in_channels': train_params['in_channels'],
                    'dropout': train_args.dropout}
    else:
        net = VisionTransformerClassifier
        net_args = {'arch_name': train_args.model, 'n_label': train_params['n_label'],
                    'pretrained': train_args.pretrained_model,
                    'fine_tune_layers': train_args.fine_tune_layers,
                    'emb_size': train_params['emb_size'],
                    'dropout': train_args.dropout,
                    'patch_size': train_args.vit_patch_size,
                    'n_last_blocks': train_args.vit_n_last_blocks,
                    'avgpool_patchtokens': train_args.vit_avgpool_patchtokens,
                    'pretrained_weights': train_args.vit_pretrained_weights}
    handler = get_handler(args.data_name)

    use_cuda = torch.cuda.is_available()
    print('Using %s device.' % ("cuda" if use_cuda else "cpu"))
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.strategy == 'CDALSampling':
        model = CDALModel(net, net_args, handler, train_params, writer, device, init_model=True)
    else:
        model = Training(net, net_args, handler, train_params, writer, device, init_model=True)

    cls = globals()[strategy_name]
    strategy = cls(X_tr, Y_tr, idxs_lb, X_val, Y_val, model, args, device, writer)

    # print info
    print(args.data_name)
    print('SEED {}'.format(seed))
    print(type(strategy).__name__)

    # round 0 accuracy
    strategy.train(name='0')
    P = strategy.predict(X_te, Y_te)
    acc = np.zeros(args.n_round + 1)
    acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    print('Round 0\ntesting accuracy {}'.format(acc[0]))
    writer.add_scalar('test_accuracy', acc[0], 0)

    result_writer.writerow([acc[0], 0.])
    result_file.flush()

    for rd in range(1, args.n_round + 1):
        print('Round {}'.format(rd))

        start_time = time.time()

        budget = args.n_query * int(math.pow(args.query_growth_ratio, rd - 1))
        print('query budget: %d' % budget)
        q_idxs, embeddings, preds, probs, u_idxs, candidate_idxs = strategy.query(budget)

        duration = time.time() - start_time

        query_result = torch.zeros(Y_tr.size(), dtype=torch.bool)
        query_result[q_idxs] = True

        if preds is not None:
            s_gt_y = strategy.Y[q_idxs]
            s_y = preds[u_idxs]
            s_embddings = embeddings[u_idxs]
            s_probs = probs[u_idxs]

            get_query_diversity_uncertainty(s_embddings, s_gt_y, s_y, s_probs, writer, rd)

        if args.save_images:
            all_embeds = np.zeros((Y_tr.size()[0], embeddings.size(1)), dtype=float)
            all_embeds[~idxs_lb] = embeddings

            visualise_results(all_embeds, q_idxs if candidate_idxs is None else candidate_idxs, Y_tr, q_idxs,
                              os.path.join(sub_path, 'embedding_round_%d.ckp' % (rd)))

        # update
        idxs_lb[q_idxs] = True
        strategy.update(idxs_lb)

        print('training with %d labeled samples.' % idxs_lb.sum())
        strategy.train(str(rd))

        # round accuracy
        P = strategy.predict(X_te, Y_te)
        acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)

        if args.save_checkpoints:
            torch.save(strategy.model.clf.state_dict(), os.path.join(sub_path, 'model_round_%d.pt' % (rd)))
        np.save(open(os.path.join(sub_path, 'query_' + str(rd) + '.np'), 'wb'), q_idxs)

        print('testing accuracy {}'.format(acc[rd]))
        writer.add_scalar('test_accuracy', acc[rd], rd)
        result_writer.writerow([acc[rd], duration])
        result_file.flush()

    # print results
    print('SEED {}'.format(seed))
    print(type(strategy).__name__)
    print(acc)

    writer.close()
    result_file.close()

    if args.print_to_file:
        sys.stdout = orig_stdout
        log_file.close()


def get_query_diversity_uncertainty(embeddings, gt_y, p_y, probs, writer, rd):
    print('number of samples that are misclassified and selected: %d (%0.2f%%)' % (
    (p_y != gt_y).sum().item(), (p_y != gt_y).sum().item() / float(p_y.size(0)) * 100))

    writer.add_scalar('selection_statistics/pred_error_count', (p_y != gt_y).sum().item(), rd)
    writer.add_scalar('selection_statistics/pred_error_precentage', (p_y != gt_y).sum().item() / float(p_y.size(0)) * 100, rd)

    a = np.matmul(embeddings, embeddings.transpose(1, 0))
    sign, logdet = np.linalg.slogdet(a)
    print('Log Determinant of the Gram Matrix: %f' % logdet.item())
    writer.add_scalar('selection_statistics/log_det_gram', logdet.item(), rd)

    print('Signed Log Determinant of the Gram Matrix: %f' % (sign.item() * logdet.item()))
    writer.add_scalar('selection_statistics/singned_log_det_gram', (sign.item() * logdet.item()), rd)

    conf = probs.max(1)[0].mean().item()
    print('Confidence: %f' % conf)
    writer.add_scalar('selection_statistics/confidence', conf, rd)

    probs_sorted, idxs = probs.sort(descending=True)
    margin = (probs_sorted[:, 0] - probs_sorted[:, 1]).mean().item()
    print('Margin: %f' % margin)
    writer.add_scalar('selection_statistics/margin', margin, rd)

    from scipy.stats import entropy
    p = np.zeros(probs.size(1), dtype=np.float)
    for i in range(probs.size(1)):
        p[i] = (p_y == i).sum().item() / p_y.size(0)
    ent = entropy(p)
    print('Predicted Entropy: %f' % ent)
    writer.add_scalar('selection_statistics/predicted_entropy', ent, rd)

    p = np.zeros(probs.size(1), dtype=np.float)
    for i in range(probs.size(1)):
        p[i] = (gt_y == i).sum().item() / p_y.size(0)
    ent = entropy(p)
    print('GT Entropy: %f' % ent)
    writer.add_scalar('selection_statistics/gt_entropy', ent, rd)

    c = idxs[:, 0:2].min(dim=1)[0] * 1000 + idxs[:, 0:2].max(dim=1)[0]
    n = int(probs.size(1) * (probs.size(1) - 1) / 2)
    p = np.zeros(n, dtype=np.float)
    idx = 0
    for i in range(probs.size(1) - 1):
        for j in range(i + 1, probs.size(1)):
            p[idx] = (c == (i * 1000) + j).sum().item() / p_y.size(0)
            idx += 1
    ent = entropy(p)
    print('Border Entropy: %f' % ent)
    writer.add_scalar('selection_statistics/border_entropy', ent, rd)


def visualise_results(all_embeddings, can_idxs, Y_tr, q_idxs, path):
    #embeddings = torch.cat([all_embeddings[idxs_lb], all_embeddings[q_idxs]], dim=1)
    if all_embeddings.shape[-1] == 2:
        tsne_results = all_embeddings
    else:
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(all_embeddings)

    lbls = Y_tr * 3
    lbls[can_idxs] += 1
    lbls[q_idxs] += 1
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.020, right=.995, top=.995, bottom=0.025, wspace=0.2, hspace=0.2)
    plt.axis('off')
    for label in range(Y_tr.max() + 1):
        # c = np.random.rand(3)
        c = list(mcolors.TABLEAU_COLORS.values())[label]
        ax.scatter(tsne_results[lbls == label * 3][:, 0],
                   tsne_results[lbls == label * 3][:, 1],
                   alpha=0.15, edgecolors='none', c=c, marker='o', s=30)
        ax.scatter(tsne_results[can_idxs][lbls[can_idxs] == label * 3 + 1][:, 0],
                   tsne_results[can_idxs][lbls[can_idxs] == label * 3 + 1][:, 1],
                   alpha=1., edgecolors='none', c=c, marker='*', s=180)
        ax.scatter(tsne_results[q_idxs][lbls[q_idxs] == label * 3 + 2][:, 0],
                   tsne_results[q_idxs][lbls[q_idxs] == label * 3 + 2][:, 1],
                   label=str(label), alpha=1., edgecolors='none', c=c, marker='o', s=120)
    ax.grid(False)
    plt.savefig(path, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General active learning hyper-parameters")

    parser.add_argument('--data_name', type=str, choices=['MNIST', 'EMNIST',
                                                          'SVHN', 'CIFAR10',
                                                          'CIFAR100', 'MiniImageNet',
                                                          'domain_net-real', 'mini_domain_net-real', 'tiny_domain_net-real',
                                                          'openml_6', 'openml_155'])
    parser.add_argument('--n_label', type=int, default=10, help='The number of distinct classes in the dataset.')

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_checkpoints', action='store_const', default=False, const=True)

    parser.add_argument('--save_images', action="store_const", default=False, const=True)
    parser.add_argument('--print_to_file', action="store_const", default=False, const=True)

    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 10, 100, 1000, 10000])
    parser.add_argument('--n_init_lb', type=int, default=100)
    parser.add_argument('--init_lb_method', type=str, default='general_random',
                        choices=['general_random', 'per_class_random'])
    parser.add_argument('--n_query', type=int, default=100)
    parser.add_argument('--query_growth_ratio', type=int, default=1)
    parser.add_argument('--n_round', type=int, default=15)

    parser.add_argument('--strategy', type=str,
                        choices=['RandomSampling', 'EntropySampling',
                                 'BALDDropout', 'CoreSet',
                                 'AdversarialDeepFool',
                                 'BadgeSampling', 'CDALSampling', 'GCNSampling', 'AlphaMixSampling',
                                 'All'])

    parser.add_argument('--n_drop', type=int, default=5)
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--max_iter', type=int, default=50)

    # AlphaMix hyper-parameters
    parser.add_argument('--alpha_cap', type=float, default=0.03125)
    parser.add_argument('--alpha_opt', action="store_const", default=False, const=True)
    parser.add_argument('--alpha_closed_form_approx', action="store_const", default=False, const=True)

    # Gradient descent Alpha optimisation
    parser.add_argument('--alpha_learning_rate', type=float, default=0.1,
                        help='The learning rate of finding the optimised alpha')
    parser.add_argument('--alpha_clf_coef', type=float, default=1.0)
    parser.add_argument('--alpha_l2_coef', type=float, default=0.01)
    parser.add_argument('--alpha_learning_iters', type=int, default=5,
                        help='The number of iterations for learning alpha')
    parser.add_argument('--alpha_learn_batch_size', type=int, default=1000000)

    args, _ = parser.parse_known_args()

    supervised_learning(args)
