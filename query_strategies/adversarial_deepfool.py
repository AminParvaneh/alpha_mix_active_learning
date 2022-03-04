import numpy as np
import torch
from tqdm import tqdm

from .strategy import Strategy


class AdversarialDeepFool(Strategy):
    def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer):
        super(AdversarialDeepFool, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer)

    def cal_dis(self, x):
        nx = torch.unsqueeze(x, 0)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)

        out, e1 = self.model.clf(nx+eta)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.args.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())

                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi

            eta += ri.clone()
            nx.grad.data.zero_()
            out, e1 = self.model.clf(nx+eta)
            py = out.max(1)[1].item()
            i_iter += 1

        return (eta*eta).sum()

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        self.model.clf.cpu()
        self.model.clf.eval()
        dis = np.zeros(idxs_unlabeled.shape)

        data_pool = self.model.handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], transform=self.model.args['transform'])
        for i in tqdm(range(len(idxs_unlabeled))):
            #if i % 100 == 0:
            #    print('adv {}/{}'.format(i, len(idxs_unlabeled)))
            x, y, idx = data_pool[i]
            dis[i] = self.cal_dis(x)

        self.model.clf.cuda()

        probs, embeddings = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

        selected = dis.argsort()[:n]
        return idxs_unlabeled[selected], embeddings, probs.max(1)[1], probs, selected, None


