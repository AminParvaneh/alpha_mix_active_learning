import numpy as np
from .strategy import Strategy
import torch.nn.functional as F


class CDALSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer):
		super(CDALSampling, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer)

	def query(self, n):
		self.query_count += 1
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

		# CDAL_RL
		#probs, embeddings = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		#chosen = self.model.train_cdal(str(self.query_count), F.softmax(probs, dim=1).numpy(), n)

		#return idxs_unlabeled[chosen], embeddings, None, None, None, None

		# CDAL_CS
		probs, embeddings = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		probs_l, _ = self.predict_prob_embed(self.X[self.idxs_lb], self.Y[self.idxs_lb])

		chosen = self.model.select_coreset(F.softmax(probs, dim=1).numpy(), F.softmax(probs_l, dim=1).numpy(), n)

		return idxs_unlabeled[chosen], embeddings, None, None, None, None
