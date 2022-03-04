import time

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Bernoulli
import numpy as np
from .training import Training


class CDALModel(Training):
    def __init__(self, net, net_args, handler, args, writer, device, init_model=True):
        super(CDALModel, self).__init__(net, net_args, handler, args, writer, device, init_model)

        self.hidden_dim = 1024
        self.max_epoch = 60
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.num_episode = 5
        self.classes = self.clf.n_label
        self.hidden_dim = self.clf.get_embedding_dim()
        self.beta = 0.01

        # self.dsn = DSN(in_dim=self.classes, hid_dim=self.hidden_dim)

    def train_cdal(self, name, features, number_of_picks):
        model = DSN(in_dim=self.classes, hid_dim=self.hidden_dim).to(self.device)
        #model = nn.DataParallel(model, device_ids=[0, 1])
        #model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        baseline = 0.
        best_reward = 0.0
        best_pi = []

        for epoch in range(self.max_epoch):
            seq = features
            seq = torch.from_numpy(seq).unsqueeze(0)  # input shape (1, seq_len, dim)
            seq = seq.to(self.device)
            probs = model(seq)  # output shape (1, seq_len, 1)
            cost = self.beta * (probs.mean() - 0.5) ** 2
            m = Bernoulli(probs)
            epis_rewards = []
            for _ in range(self.num_episode):
                actions = m.sample()
                log_probs = m.log_prob(actions)
                reward, pick_idxs = compute_reward(seq, actions, probs, nc=self.classes, picks=number_of_picks)
                if (reward > best_reward):
                    best_reward = reward
                    best_pi = pick_idxs
                expected_reward = log_probs.mean() * (reward - baseline)
                cost -= expected_reward  # minimize negative expected reward
                epis_rewards.append(reward.item())

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            baseline = 0.9 * baseline + 0.1 * np.mean(epis_rewards)  # update baseline reward via moving average
            print("epoch {}/{}\t reward {}\t".format(epoch + 1, self.max_epoch, np.mean(epis_rewards)))
            self.writer.add_scalar('cdal_reward/%s' % name, np.mean(epis_rewards), epoch)

        return best_pi

    def select_coreset(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        print('selecting coreset...')
        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs


class DSN(nn.Module):
    """Deep Summarization Network"""

    def __init__(self, in_dim=19, hid_dim=256, num_layers=1):
        super(DSN, self).__init__()
        # in_dim = in_dim*in_dim # for semantic segementation
        in_dim = in_dim
        self.lstm = nn.LSTM(in_dim, hid_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc2 = nn.Linear(hid_dim * 2, hid_dim * 2)
        self.fc1 = nn.Linear(hid_dim * 2, 1)

    def forward(self, x, in_dim=19):
        # x=torch.reshape(x,(x.shape[0],x.shape[1],in_dim*in_dim)) # for semantic segmentation
        x = x.float()
        x, _ = self.lstm(x)
        x = self.fc2(x)
        x = self.fc1(x)
        p = F.sigmoid(x)
        return p


def KL_classification(a, b):
    a = F.softmax(a)
    b = F.softmax(b)
    kl1 = a * torch.log(a / b)
    kl2 = b * torch.log(b / a)
    kl = -0.5 * (torch.sum(kl1)) - 0.5 * (torch.sum(kl2))
    return abs(kl)


def KL_object(a, b):
    kl1 = a * torch.log(a / b)
    kl2 = b * torch.log(b / a)
    kl = -0.5 * (torch.sum(kl1)) - 0.5 * (torch.sum(kl2))
    return abs(kl)


def KL_symmetric(a, b):
    kl1 = a * torch.log(a / b)
    kl2 = b * torch.log(b / a)
    kl = 0.5 * (torch.sum(kl1)) + 0.5 * (torch.sum(kl2))
    return kl


def KL_segment(ac, bc, nc):  ## nc = number of classes
    kl_classes = []
    reward_kl = 0.0
    for i in range(nc):
        a = ac[i, :]
        b = bc[i, :]
        kl1 = a * torch.log(a / b)
        kl2 = b * torch.log(b / a)
        kl = -0.5 * (torch.sum(kl1)) - 0.5 * (torch.sum(kl2))
        if (kl == kl and not torch.isinf(kl)):
            kl_classes.append(abs(kl))
    if (len(kl_classes) != 0):
        reward_kl = sum(kl_classes)
    if reward_kl == 0.0:
        return torch.tensor(0.0)
    reward_kl = reward_kl.float()
    return reward_kl


def pairwise_distances_old(a, b):
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    start_time = time.time()
    dist = np.zeros((a.size(0), b.size(0)), dtype=np.float)
    for i in range(a.size(0)):
        for j in range(b.size(0)):
            # dist[i][j] = KL_object(torch.from_numpy(a[i] + 1e-8), torch.from_numpy(b[j] + 1e-8))
            dist[i][j] = KL_symmetric(a[i], b[j])

    duration = time.time() - start_time
    print('duration of kl calcculation: %d' % duration)
    return dist


def pairwise_distances(a, b):
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)

    dist = np.zeros((a.size(0), b.size(0)), dtype=np.float)
    for i in range(b.size(0)):
        b_i = b[i]
        kl1 = a * torch.log(a / b_i)
        kl2 = b_i * torch.log(b_i / a)
        dist[:, i] = 0.5 * (torch.sum(kl1, dim=1)) + 0.5 * (torch.sum(kl2, dim=1))
    return dist


def CD(seq, pick_idxs, nc):
    reward_kl = []
    for k in pick_idxs:
        for l in pick_idxs:
            reward_kl.append(KL_object(seq[k, :], seq[l, :]))  # change function according to the task.
    reward_kl = torch.stack(reward_kl)
    reward_kl = torch.mean(reward_kl)
    return reward_kl


def V_rep(_seq, pick_idxs):
    n = _seq.shape[0]
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:, pick_idxs.copy()]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    # reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0]
    reward_rep = torch.exp(-dist_mat.mean() / 50)
    return reward_rep


def compute_reward(seq, actions, probs, nc, picks, use_gpu=False):
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    probs = probs.detach().cpu().numpy().squeeze()
    top_pick_idxs = probs.argsort()[-1 * picks:][::-1]
    _seq = _seq.squeeze()
    n = _seq.size(0)

    reward_kl = CD(_seq, top_pick_idxs.squeeze(), nc)
    rep_reward = V_rep(_seq, top_pick_idxs.squeeze())
    reward = rep_reward * 0.5 + reward_kl * 1.5
    return reward, top_pick_idxs
