from .strategy import Strategy
import pdb
from scipy import stats
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from sklearn.cluster import KMeans


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    #print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        #print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll


class BadgeSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer):
        super(BadgeSampling, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs, embeddings = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        _, idxs = probs.sort(descending=True)

        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y[idxs_unlabeled]).numpy()
        chosen = init_centers(gradEmbedding, n)

        return idxs_unlabeled[chosen], embeddings, probs.max(1)[1], probs, chosen, None
