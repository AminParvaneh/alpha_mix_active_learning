import copy
import math

import numpy as np
from torch.autograd import Variable

from .strategy import Strategy
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F


class AlphaMixSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, X_val, Y_val, model, args, device, writer):
		super(AlphaMixSampling, self).__init__(X, Y, idxs_lb, X_val, Y_val, model, args, device, writer)

	def query(self, n, idxs_prohibited=None):
		self.query_count += 1

		idxs = self.idxs_lb if idxs_prohibited is None else (self.idxs_lb + idxs_prohibited)
		idxs_unlabeled = np.arange(self.n_pool)[~idxs]

		ulb_probs, org_ulb_embedding = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

		probs_sorted, probs_sort_idxs = ulb_probs.sort(descending=True)
		pred_1 = probs_sort_idxs[:, 0]

		lb_probs, org_lb_embedding = self.predict_prob_embed(self.X[self.idxs_lb], self.Y[self.idxs_lb])

		ulb_embedding = org_ulb_embedding
		lb_embedding = org_lb_embedding

		unlabeled_size = ulb_embedding.size(0)
		embedding_size = ulb_embedding.size(1)

		min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
		candidate = torch.zeros(unlabeled_size, dtype=torch.bool)

		if self.args.alpha_closed_form_approx:
			var_emb = Variable(ulb_embedding, requires_grad=True).to(self.device)
			out, _ = self.model.clf(var_emb, embedding=True)
			loss = F.cross_entropy(out, pred_1.to(self.device))
			grads = torch.autograd.grad(loss, var_emb)[0].data.cpu()
			del loss, var_emb, out
		else:
			grads = None

		alpha_cap = 0.
		while alpha_cap < 1.0:
			alpha_cap += self.args.alpha_cap

			tmp_pred_change, tmp_min_alphas = \
				self.find_candidate_set(
					lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap=alpha_cap,
					Y=self.Y[self.idxs_lb],
					grads=grads)

			is_changed = min_alphas.norm(dim=1) >= tmp_min_alphas.norm(dim=1)

			min_alphas[is_changed] = tmp_min_alphas[is_changed]
			candidate += tmp_pred_change

			print('With alpha_cap set to %f, number of inconsistencies: %d' % (alpha_cap, int(tmp_pred_change.sum().item())))

			if candidate.sum() > n:
				break

		if candidate.sum() > 0:
			print('Number of inconsistencies: %d' % (int(candidate.sum().item())))

			print('alpha_mean_mean: %f' % min_alphas[candidate].mean(dim=1).mean().item())
			print('alpha_std_mean: %f' % min_alphas[candidate].mean(dim=1).std().item())
			print('alpha_mean_std %f' % min_alphas[candidate].std(dim=1).mean().item())

			self.writer.add_scalar('stats/candidate_set_size', candidate.sum().item(), self.query_count)
			self.writer.add_scalar('stats/alpha_mean_mean', min_alphas[candidate].mean(dim=1).mean().item(), self.query_count)
			self.writer.add_scalar('stats/alpha_std_mean', min_alphas[candidate].mean(dim=1).std().item(), self.query_count)
			self.writer.add_scalar('stats/alpha_mean_std', min_alphas[candidate].std(dim=1).mean().item(), self.query_count)

			c_alpha = F.normalize(org_ulb_embedding[candidate].view(candidate.sum(), -1), p=2, dim=1).detach()

			selected_idxs = self.sample(min(n, candidate.sum().item()), feats=c_alpha)
			u_selected_idxs = candidate.nonzero(as_tuple=True)[0][selected_idxs]
			selected_idxs = idxs_unlabeled[candidate][selected_idxs]
		else:
			selected_idxs = np.array([], dtype=np.int)

		if len(selected_idxs) < n:
			remained = n - len(selected_idxs)
			idx_lb = copy.deepcopy(self.idxs_lb)
			idx_lb[selected_idxs] = True
			selected_idxs = np.concatenate([selected_idxs, np.random.choice(np.where(idx_lb == 0)[0], remained)])
			print('picked %d samples from RandomSampling.' % (remained))

		return np.array(selected_idxs), ulb_embedding, pred_1, ulb_probs, u_selected_idxs, idxs_unlabeled[candidate]

	def find_alpha(self):
		assert self.args.alpha_num_mix <= self.args.n_label - (
			0 if self.args.alpha_use_highest_class_mix else 1), 'c_num_mix should not be greater than number of classes'

		self.query_count += 1

		idxs_unlabeled = np.arange(self.n_pool)[self.idxs_lb]

		ulb_probs, ulb_embedding = self.predict_prob_embed(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

		probs_sorted, probs_sort_idxs = ulb_probs.sort(descending=True)
		pred_1 = probs_sort_idxs[:, 0]
		gt_lables = self.Y[idxs_unlabeled]
		preds = pred_1 == gt_lables

		ulb_embedding = ulb_embedding[preds]
		ulb_probs = ulb_probs[preds]
		pred_1 = pred_1[preds]

		lb_embedding = self.get_embedding(self.X[self.idxs_lb], self.Y[self.idxs_lb])

		alpha_cap = 0.
		for i in range(self.args.alpha_alpha_scales if self.args.alpha_alpha_growth_method == 'exponential' else int(pow(2, self.args.alpha_alpha_scales) - 1)):
			if self.args.alpha_alpha_growth_method == 'exponential':
				alpha_cap = self.args.alpha_cap / (pow(2, self.args.alpha_alpha_scales - i - 1))
			else:
				#alpha_cap *= float(n * (1 if self.args.alpha_max_changes <= 0 else self.args.alpha_max_changes)) / pred_change.sum().item()
				alpha_cap += self.args.alpha_cap / (pow(2, self.args.alpha_alpha_scales - 1))

			tmp_pred_change, tmp_pred_change_sum, tmp_min_alphas, tmp_min_added_feats, tmp_cf_probs, _, tmp_min_mixing_labels, tmp_min_cf_feats = \
				self.find_candidate_set(
					lb_embedding, ulb_embedding, pred_1, ulb_probs, probs_sort_idxs, alpha_cap=alpha_cap,
					Y=self.Y[self.idxs_lb])

			if tmp_pred_change.sum() > 0:
				print('selected alpha_max %f' % alpha_cap)
				self.writer.add_scalar('stats/alpha_cap', alpha_cap, self.query_count)

				return alpha_cap

		print('no labelled sample change!!!')
		return 0.5

	def find_candidate_set(self, lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap, Y, grads):

		unlabeled_size = ulb_embedding.size(0)
		embedding_size = ulb_embedding.size(1)

		min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
		pred_change = torch.zeros(unlabeled_size, dtype=torch.bool)

		if self.args.alpha_closed_form_approx:
			alpha_cap /= math.sqrt(embedding_size)
			grads = grads.to(self.device)
			
		for i in range(self.args.n_label):
			emb = lb_embedding[Y == i]
			if emb.size(0) == 0:
				emb = lb_embedding
			anchor_i = emb.mean(dim=0).view(1, -1).repeat(unlabeled_size, 1)

			if self.args.alpha_closed_form_approx:
				embed_i, ulb_embed = anchor_i.to(self.device), ulb_embedding.to(self.device)
				alpha = self.calculate_optimum_alpha(alpha_cap, embed_i, ulb_embed, grads)

				embedding_mix = (1 - alpha) * ulb_embed + alpha * embed_i
				out, _ = self.model.clf(embedding_mix, embedding=True)
				out = out.detach().cpu()
				alpha = alpha.cpu()

				pc = out.argmax(dim=1) != pred_1
			else:
				alpha = self.generate_alpha(unlabeled_size, embedding_size, alpha_cap)
				if self.args.alpha_opt:
					alpha, pc = self.learn_alpha(ulb_embedding, pred_1, anchor_i, alpha, alpha_cap,
												 log_prefix=str(i))
				else:
					embedding_mix = (1 - alpha) * ulb_embedding + alpha * anchor_i
					out, _ = self.model.clf(embedding_mix.to(self.device), embedding=True)
					out = out.detach().cpu()

					pc = out.argmax(dim=1) != pred_1

			torch.cuda.empty_cache()
			self.writer.add_scalar('stats/inconsistencies_%d' % i, pc.sum().item(), self.query_count)

			alpha[~pc] = 1.
			pred_change[pc] = True
			is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)
			min_alphas[is_min] = alpha[is_min]

			self.writer.add_scalar('stats/inconsistencies_%d' % i, pc.sum().item(), self.query_count)
			
		return pred_change, min_alphas

	def calculate_optimum_alpha(self, eps, lb_embedding, ulb_embedding, ulb_grads):
		z = (lb_embedding - ulb_embedding) #* ulb_grads
		alpha = (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (z + 1e-8)

		return alpha

	def sample(self, n, feats):
		feats = feats.numpy()
		cluster_learner = KMeans(n_clusters=n)
		cluster_learner.fit(feats)

		cluster_idxs = cluster_learner.predict(feats)
		centers = cluster_learner.cluster_centers_[cluster_idxs]
		dis = (feats - centers) ** 2
		dis = dis.sum(axis=1)
		return np.array(
			[np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) if
			 (cluster_idxs == i).sum() > 0])

	def retrieve_anchor(self, embeddings, count):
		return embeddings.mean(dim=0).view(1, -1).repeat(count, 1)

	def generate_alpha(self, size, embedding_size, alpha_cap):
		alpha = torch.normal(
			mean=alpha_cap / 2.0,
			std=alpha_cap / 2.0,
			size=(size, embedding_size))

		alpha[torch.isnan(alpha)] = 1

		return self.clamp_alpha(alpha, alpha_cap)

	def clamp_alpha(self, alpha, alpha_cap):
		return torch.clamp(alpha, min=1e-8, max=alpha_cap)

	def learn_alpha(self, org_embed, labels, anchor_embed, alpha, alpha_cap, log_prefix=''):
		labels = labels.to(self.device)
		min_alpha = torch.ones(alpha.size(), dtype=torch.float)
		pred_changed = torch.zeros(labels.size(0), dtype=torch.bool)

		loss_func = torch.nn.CrossEntropyLoss(reduction='none')

		self.model.clf.eval()

		for i in range(self.args.alpha_learning_iters):
			tot_nrm, tot_loss, tot_clf_loss = 0., 0., 0.
			for b in range(math.ceil(float(alpha.size(0)) / self.args.alpha_learn_batch_size)):
				self.model.clf.zero_grad()
				start_idx = b * self.args.alpha_learn_batch_size
				end_idx = min((b + 1) * self.args.alpha_learn_batch_size, alpha.size(0))

				l = alpha[start_idx:end_idx]
				l = torch.autograd.Variable(l.to(self.device), requires_grad=True)
				opt = torch.optim.Adam([l], lr=self.args.alpha_learning_rate / (1. if i < self.args.alpha_learning_iters * 2 / 3 else 10.))
				e = org_embed[start_idx:end_idx].to(self.device)
				c_e = anchor_embed[start_idx:end_idx].to(self.device)
				embedding_mix = (1 - l) * e + l * c_e

				out, _ = self.model.clf(embedding_mix, embedding=True)

				label_change = out.argmax(dim=1) != labels[start_idx:end_idx]

				tmp_pc = torch.zeros(labels.size(0), dtype=torch.bool).to(self.device)
				tmp_pc[start_idx:end_idx] = label_change
				pred_changed[start_idx:end_idx] += tmp_pc[start_idx:end_idx].detach().cpu()

				tmp_pc[start_idx:end_idx] = tmp_pc[start_idx:end_idx] * (l.norm(dim=1) < min_alpha[start_idx:end_idx].norm(dim=1).to(self.device))
				min_alpha[tmp_pc] = l[tmp_pc[start_idx:end_idx]].detach().cpu()

				clf_loss = loss_func(out, labels[start_idx:end_idx].to(self.device))

				l2_nrm = torch.norm(l, dim=1)

				clf_loss *= -1

				loss = self.args.alpha_clf_coef * clf_loss + self.args.alpha_l2_coef * l2_nrm
				loss.sum().backward(retain_graph=True)
				opt.step()

				l = self.clamp_alpha(l, alpha_cap)

				alpha[start_idx:end_idx] = l.detach().cpu()

				tot_clf_loss += clf_loss.mean().item() * l.size(0)
				tot_loss += loss.mean().item() * l.size(0)
				tot_nrm += l2_nrm.mean().item() * l.size(0)

				del l, e, c_e, embedding_mix
				torch.cuda.empty_cache()

		count = pred_changed.sum().item()
		if pred_changed.sum() > 0:
			self.writer.add_scalar('stats/inconsistencies_%s' % (log_prefix), count, self.query_count)
			self.writer.add_scalar('alpha_change/mean_%s' % (log_prefix), min_alpha[pred_changed].mean().item(), self.query_count)
			self.writer.add_scalar('alpha_change/std_%s' % (log_prefix), min_alpha[pred_changed].std().item(), self.query_count)
			self.writer.add_scalar('alpha_change/norm_%s' % (log_prefix), min_alpha[pred_changed].norm(dim=1).mean(), self.query_count)
			self.writer.add_scalar('alpha_change/clf_loss_%s' % (log_prefix), tot_clf_loss / min_alpha.size(0), self.query_count)
			self.writer.add_scalar('alpha_change/loss_%s' % (log_prefix), tot_loss / min_alpha.size(0), self.query_count)
		else:
			self.writer.add_scalar('stats/inconsistencies_%s' % (log_prefix), 0, self.query_count)
			self.writer.add_scalar('alpha_change/mean_%s' % (log_prefix), 0, self.query_count)
			self.writer.add_scalar('alpha_change/std_%s' % (log_prefix), 0, self.query_count)
			self.writer.add_scalar('alpha_change/norm_%s' % (log_prefix), 0, self.query_count)
			self.writer.add_scalar('alpha_change/clf_loss_%s' % (log_prefix), 0, self.query_count)
			self.writer.add_scalar('alpha_change/loss_%s' % (log_prefix), 0, self.query_count)

		return min_alpha.cpu(), pred_changed.cpu()

