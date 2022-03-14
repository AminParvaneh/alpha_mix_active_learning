import os
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import openml
from sklearn.preprocessing import LabelEncoder
import pickle


mini_domain_net_class_ids = {
    #'airplane': 2,
    'bat': 21,
    'bathtub': 22,
    'bear': 24,
    'bed': 26,
    'bench': 29,
    'bicycle': 30,
    'bird': 32,
    'bus': 48,
    'butterfly': 50,
    'car': 62,
    'carrot': 63,
    'cat': 65,
    'chair': 69,
    'couch': 81,
    'cruise_ship': 87,
    'dog': 92,
    'pizza': 226,
    'strawberry': 291,
    'table': 302,
    'zebra': 345,
}


tiny_domain_net_class_ids = {
    'bat': 21,
    #'bathtub': 22,
    'bear': 24,
    #'bed': 26,
    #'bench': 29,
    'bicycle': 30,
    'bird': 32,
    'bus': 48,
    'butterfly': 50,
    'car': 62,
    #'carrot': 63,
    'cat': 65,
    #'chair': 69,
    #'couch': 81,
    #'cruise_ship': 87,
    'dog': 92,
    #'pizza': 226,
    #'strawberry': 291,
    #'table': 302,
    'zebra': 345,
}


def is_openml(name):
    return len(name) > 6 and name[:6] == 'openml'


def get_openml_id(name):
    return int(name[7:])


def get_dataset(name, data_dir):
    if name == 'MNIST':
        return get_MNIST(data_dir)
    if name == 'EMNIST':
        return get_EMNIST(data_dir)
    elif name == 'SVHN':
        return get_SVHN(data_dir)
    elif name == 'CIFAR10':
        return get_CIFAR10(data_dir)
    elif name == 'CIFAR100':
        return get_CIFAR100(data_dir)
    elif name == 'MiniImageNet':
        return get_MiniImageNet(data_dir)
    elif name == 'domain_net-real':
        return get_DomainNet_Real(data_dir)
    elif name == 'mini_domain_net-real':
        return get_Mini_DomainNet_Real(data_dir)
    elif name == 'tiny_domain_net-real':
        return get_Tiny_DomainNet_Real(data_dir)
    elif is_openml(name):
        return get_openml(data_dir, get_openml_id(name))


def get_MNIST(data_dir):
    raw_tr = datasets.MNIST(os.path.join(data_dir, 'MNIST'), train=True, download=True)
    raw_te = datasets.MNIST(os.path.join(data_dir, 'MNIST'), train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_te, Y_te


def get_EMNIST(data_dir):
    raw_tr = datasets.EMNIST(os.path.join(data_dir, 'EMNIST'), train=True, download=True, split='letters')
    raw_te = datasets.EMNIST(os.path.join(data_dir, 'EMNIST'), train=False, download=True, split='letters')
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets - 1
    X_te = raw_te.data
    Y_te = raw_te.targets - 1
    return X_tr, Y_tr, X_te, Y_te


def get_SVHN(data_dir):
    data_tr = datasets.SVHN(os.path.join(data_dir, 'SVHN'), split='train', download=True)
    data_te = datasets.SVHN(os.path.join(data_dir, 'SVHN'), split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te


def get_CIFAR10(data_dir):
    data_tr = datasets.CIFAR10(os.path.join(data_dir, 'CIFAR10'), train=True, download=True)
    data_te = datasets.CIFAR10(os.path.join(data_dir, 'CIFAR10'), train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te


def get_CIFAR100(data_dir):
    data_tr = datasets.CIFAR100(os.path.join(data_dir, 'CIFAR100'), train=True, download=True)
    data_te = datasets.CIFAR100(os.path.join(data_dir, 'CIFAR100'), train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te


def get_MiniImageNet(data_dir):
    f = open(os.path.join(data_dir, 'MiniImageNet', 'mini-imagenet-cache-train.pkl'), 'rb')
    train_data = pickle.load(f)
    f = open(os.path.join(data_dir, 'MiniImageNet', 'mini-imagenet-cache-val.pkl'), 'rb')
    val_data = pickle.load(f)
    f = open(os.path.join(data_dir, 'MiniImageNet', 'mini-imagenet-cache-test.pkl'), 'rb')
    test_data = pickle.load(f)

    labels = list(train_data['class_dict'].keys()) + list(val_data['class_dict'].keys()) + list(test_data['class_dict'].keys())
    image_count = len(train_data['class_dict'][labels[0]])
    test_proportion = int(image_count * 0.2)
    train_proportion = image_count - test_proportion

    image_dim = train_data['image_data'].shape[1]
    X_tr = np.zeros((len(labels) * train_proportion, image_dim, image_dim, 3), dtype=np.uint8)
    Y_tr = torch.ones((len(labels) * train_proportion), dtype=torch.long)
    X_te = np.zeros((len(labels) * test_proportion, image_dim, image_dim, 3), dtype=np.uint8)
    Y_te = torch.ones((len(labels) * test_proportion), dtype=torch.long)

    idx = 0
    for label in train_data['class_dict']:
        X_te[idx * test_proportion:(idx + 1) * test_proportion] = train_data['image_data'][train_data['class_dict'][label][:test_proportion]]
        Y_te[idx * test_proportion:(idx + 1) * test_proportion] *= labels.index(label)

        X_tr[idx * train_proportion:(idx + 1) * train_proportion] = train_data['image_data'][train_data['class_dict'][label][test_proportion:]]
        Y_tr[idx * train_proportion:(idx + 1) * train_proportion] *= labels.index(label)

        idx += 1

    for label in val_data['class_dict']:
        X_te[idx * test_proportion:(idx + 1) * test_proportion] = val_data['image_data'][val_data['class_dict'][label][:test_proportion]]
        Y_te[idx * test_proportion:(idx + 1) * test_proportion] *= labels.index(label)

        X_tr[idx * train_proportion:(idx + 1) * train_proportion] = val_data['image_data'][val_data['class_dict'][label][test_proportion:]]
        Y_tr[idx * train_proportion:(idx + 1) * train_proportion] *= labels.index(label)

        idx += 1

    for label in test_data['class_dict']:
        X_te[idx * test_proportion:(idx + 1) * test_proportion] = test_data['image_data'][test_data['class_dict'][label][:test_proportion]]
        Y_te[idx * test_proportion:(idx + 1) * test_proportion] *= labels.index(label)

        X_tr[idx * train_proportion:(idx + 1) * train_proportion] = test_data['image_data'][test_data['class_dict'][label][test_proportion:]]
        Y_tr[idx * train_proportion:(idx + 1) * train_proportion] *= labels.index(label)

        idx += 1

    return X_tr, Y_tr, X_te, Y_te


def get_DomainNet_Real(data_dir, type='full'):
    data_dir = os.path.join(data_dir, 'domain_net-real')
    import pandas as pd
    df = pd.read_feather(os.path.join(data_dir, 'labels_%s' % type, 'labels_train.feather'))

    X_tr, Y_tr, X_te, Y_te = [], [], [], []
    label_map = {}

    for idx, row in df.iterrows():
        X_tr.append(os.path.join(data_dir, 'domain_net-real_%s' % type, 'train', row[0]))
        if row[1] not in label_map:
            label_map[row[1]] = len(label_map)
        Y_tr.append(label_map[row[1]])

    df = pd.read_feather(os.path.join(data_dir, 'labels_%s' % type, 'labels_test.feather'))
    for idx, row in df.iterrows():
        X_te.append(os.path.join(data_dir, 'domain_net-real_%s' % type, 'test', row[0]))
        if row[1] not in label_map:
            label_map[row[1]] = len(label_map) + 1
        Y_te.append(label_map[row[1]])

    return np.array(X_tr), torch.from_numpy(np.array(Y_tr)), np.array(X_te), torch.from_numpy(np.array(Y_te))


def get_Mini_DomainNet_Real(data_dir, type='full'):
    data_dir = os.path.join(data_dir, 'domain_net-real')
    import pandas as pd
    df = pd.read_feather(os.path.join(data_dir, 'labels_%s' % type, 'labels_train.feather'))

    X_tr, Y_tr, X_te, Y_te = [], [], [], []
    label_map = {}

    for idx, row in df.iterrows():
        if row[1] in mini_domain_net_class_ids:
            X_tr.append(os.path.join(data_dir, 'domain_net-real_%s' % type, 'train', row[0]))
            if row[1] not in label_map:
                label_map[row[1]] = len(label_map)
            Y_tr.append(label_map[row[1]])

    df = pd.read_feather(os.path.join(data_dir, 'labels_%s' % type, 'labels_test.feather'))
    for idx, row in df.iterrows():
        #print(row[1])
        if row[1] in mini_domain_net_class_ids:
            X_te.append(os.path.join(data_dir, 'domain_net-real_%s' % type, 'test', row[0]))
            if row[1] not in label_map:
                label_map[row[1]] = len(label_map) + 1
            Y_te.append(label_map[row[1]])

    return np.array(X_tr), torch.from_numpy(np.array(Y_tr)), np.array(X_te), torch.from_numpy(np.array(Y_te))


def get_Tiny_DomainNet_Real(data_dir, type='full'):
    data_dir = os.path.join(data_dir, 'domain_net-real')
    import pandas as pd
    df = pd.read_feather(os.path.join(data_dir, 'labels_%s' % type, 'labels_train.feather'))

    X_tr, Y_tr, X_te, Y_te = [], [], [], []
    label_map = {}

    for idx, row in df.iterrows():
        if row[1] in tiny_domain_net_class_ids:
            X_tr.append(os.path.join(data_dir, 'domain_net-real_%s' % type, 'train', row[0]))
            if row[1] not in label_map:
                label_map[row[1]] = len(label_map)
            Y_tr.append(label_map[row[1]])

    df = pd.read_feather(os.path.join(data_dir, 'labels_%s' % type, 'labels_test.feather'))
    for idx, row in df.iterrows():
        #print(row[1])
        if row[1] in tiny_domain_net_class_ids:
            X_te.append(os.path.join(data_dir, 'domain_net-real_%s' % type, 'test', row[0]))
            if row[1] not in label_map:
                label_map[row[1]] = len(label_map) + 1
            Y_te.append(label_map[row[1]])

    return np.array(X_tr), torch.from_numpy(np.array(Y_tr)), np.array(X_te), torch.from_numpy(np.array(Y_te))


def get_openml(data_dir, dataset_id):
    openml.config.apikey = '3411e20aff621cc890bf403f104ac4bc'
    openml.config.set_cache_directory(data_dir)
    ds = openml.datasets.get_dataset(dataset_id)
    data = ds.get_data(target=ds.default_target_attribute)
    X = np.asarray(data[0], dtype=np.int64)
    y = np.asarray(data[1])
    y = LabelEncoder().fit(y).transform(y)

    nClasses = int(max(y) + 1)
    nSamps, dim = np.shape(X)
    testSplit = .1
    inds = np.random.permutation(nSamps)
    X = X[inds]
    y = y[inds]

    split = int((1. - testSplit) * nSamps)
    while True:
        inds = np.random.permutation(split)
        if len(inds) > 50000: inds = inds[:50000]
        X_tr = X[:split]
        X_tr = X_tr[inds]
        X_tr = torch.Tensor(X_tr)

        y_tr = y[:split]
        y_tr = y_tr[inds]
        Y_tr = torch.Tensor(y_tr).long()

        X_te = torch.Tensor(X[split:])
        Y_te = torch.Tensor(y[split:]).long()

        if len(np.unique(Y_tr)) == nClasses: break

    return X_tr, Y_tr, X_te, Y_te


def get_handler(name):
    if name == 'MNIST':
        return DataHandler1
    if name == 'Imbalanced_MNIST':
        return DataHandler1
    if name == 'MiniMNIST':
        return DataHandler1
    if name == 'SmallMNIST':
        return DataHandler1
    if name == 'EMNIST':
        return DataHandler1
    if name == 'Balanced_EMNIST':
        return DataHandler1
    if name == 'MNIST_3_32_32':
        return DataHandler2
    elif name == 'FashionMNIST':
        return DataHandler1
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'MiniSVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3
    elif name == 'CIFAR100':
        return DataHandler3
    elif name == 'ImageNet':
        return DataHandler3
    elif name == 'MiniImageNet':
        return DataHandler3
    elif name == 'domain_net-real':
        return DataHandler4
    elif name == 'mini_domain_net-real':
        return DataHandler4
    elif name == 'tiny_domain_net-real':
        return DataHandler4
    elif is_openml(name):
        return DataHandler5


class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, idx):
        x = self.transform(Image.open(self.X[idx]))
        class_id = self.Y[idx]
        y = class_id.clone().detach()
        return x, y, idx

    def __len__(self):
        return len(self.X)


class DataHandler5(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)
