import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_net(name, is_openml=False):
    if name == 'MNIST':
        return Net1
    elif name == 'EMNIST':
        return Net1
    elif name == 'SVHN':
        return Net2
    elif name == 'CIFAR10':
        return Net3
    elif name == 'CIFAR100':
        return Net4
    elif is_openml:
        return Net1


class Net1(nn.Module):
    def __init__(self, n_label=10):
        super(Net1, self).__init__()
        self.n_label = n_label

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, self.n_label)
        #self.fc1_1 = nn.Linear(50, 2)
        #self.fc2 = nn.Linear(2, self.n_label)

    def forward(self, x, embedding=False):
        if embedding:
            e1 = x
        else:
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            #x = self.fc1_1(x)
            e1 = F.relu(self.fc1(x))

        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

    def get_classifier(self):
        return self.fc2


class Net2(nn.Module):
    def __init__(self, n_label=10):
        super(Net2, self).__init__()
        self.n_label = n_label

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x, embedding=False):
        if embedding:
            e1 = x
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
            x = x.view(-1, 1152)
            x = F.relu(self.fc1(x))
            e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

    def get_classifier(self):
        return self.fc3


class Net3(nn.Module):
    def __init__(self, n_label=10):
        self.n_label = n_label

        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, embedding=False):
        if embedding:
            e1 = x
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.relu(F.max_pool2d(self.conv3(x), 2))
            x = x.view(-1, 1024)
            e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

    def get_classifier(self):
        return self.fc2


class Net4(nn.Module):
    def __init__(self, n_label=100):
        self.n_label = n_label

        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 100)

    def forward(self, x, embedding=False):
        if embedding:
            e1 = x
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.relu(F.max_pool2d(self.conv3(x), 2))
            x = x.view(-1, 1024)
            e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 500

    def get_classifier(self):
        return self.fc2


class MLPNet(nn.Module):
    def __init__(self, dim, emb_size=256, n_label=10, dropout=0., add_hidden_layer=False):
        super(MLPNet, self).__init__()
        self.emb_size = emb_size
        self.n_label = n_label
        self.dim = int(np.prod(dim))
        self.do = None if dropout <= 0 else nn.Dropout(dropout)

        self.add_hidden_layer = add_hidden_layer
        if self.add_hidden_layer:
            self.lm1 = nn.Linear(self.dim, 256)
            self.lmh = nn.Linear(256, emb_size)
        else:
            self.lm1 = nn.Linear(self.dim, emb_size)

        self.lm2 = nn.Linear(emb_size, n_label)
        self.return_embeddings = True

    def forward(self, x, embedding=False):
        #embedding = False
        if embedding:
            emb = x
        else:
            x = x.view(-1, self.dim)
            emb = F.relu(self.lm1(x))
            if self.do:
                emb = self.do(emb)

            if self.add_hidden_layer:
                emb = self.lmh(emb)

        out = self.lm2(emb)
        return (out, emb) if self.return_embeddings else out

    def get_embedding_dim(self):
        return self.emb_size

    def get_classifier(self):
        return self.lm2
