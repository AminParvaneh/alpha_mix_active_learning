import numpy as np
from .strategy import Strategy
from datetime import datetime
from sklearn.metrics import pairwise_distances


class CoreSet(Strategy):
    def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer):
        super(CoreSet, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer)
        self.tor = 1e-4

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def query(self, n):
        t_start = datetime.now()
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        lb_flag = self.idxs_lb.copy()
        embed = self.get_embedding(self.X, self.Y)
        embedding = embed.numpy()

        chosen = self.furthest_first(embedding[idxs_unlabeled, :], embedding[lb_flag, :], n)

        return idxs_unlabeled[chosen], embed[idxs_unlabeled], None, None, None, None
