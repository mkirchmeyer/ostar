import ot
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from utils_local import extract_prototypes, extract_input, compute_proportion
import torch.nn.functional as F


def extract_input_estim(self, is_phi=False, is_full=False):
    x_full_s, y_full_s = extract_input(self.logger, self.source_data_loader)
    x_full_t, y_full_t = extract_input(self.logger, self.target_data_loader)
    if is_full:
        x_set_s, y_set_s, x_set_t, y_set_t = x_full_s, y_full_s, x_full_t, y_full_t
    else:
        index = min(int(min(len(x_full_s), len(x_full_t))), 2000)
        idx_s, idx_t = np.random.choice(x_full_s.size(0), index, replace=False), np.random.choice(x_full_t.size(0), index, replace=False)
        x_set_s, y_set_s, x_set_t, y_set_t = x_full_s[idx_s], y_full_s[idx_s], x_full_t[idx_t], y_full_t[idx_t]
    with torch.no_grad():
        z_set = self.feat_extractor(torch.cat((x_set_s, x_set_t), 0))
        z_set_s, z_set_t = z_set[:x_set_s.shape[0]], z_set[x_set_s.shape[0]:]
        if is_phi:
            z_set_s = self.phi(z_set_s)[0]
    return z_set_s, y_set_s, z_set_t, y_set_t


def extract_pred(self, z_set_s, z_set_t, is_phi=False):
    y_s_pred = self.data_classifier(z_set_s) if not is_phi else self.data_classifier_t(z_set_s)
    y_t_pred = self.data_classifier(z_set_t) if not is_phi else self.data_classifier_t(z_set_t)
    ptyh_soft = (torch.sum(F.softmax(y_t_pred, dim=1), dim=0).view(-1, 1).detach() / z_set_t.size(0)).reshape(-1)
    ptyh_hard = compute_proportion(self, y_t_pred.data.max(1)[1])
    return y_s_pred, ptyh_hard, ptyh_soft.cpu()


def extract_cov(self, y_s_pred, y_set_s):
    ys_onehot = torch.zeros((y_set_s.shape[0], self.n_class)).to(self.device)
    ys_onehot.scatter_(1, y_set_s.view(-1, 1), 1)
    return (torch.mm(F.softmax(y_s_pred, dim=1).transpose(1, 0), ys_onehot).detach() / y_set_s.size(0)).cpu().detach().numpy()


def estimate_proportion(self, k_prop, proportion_T=None, is_phi=False, exp_ave=True, comment=""):
    self.feat_extractor.eval()
    self.data_classifier.eval()
    if is_phi:
        self.phi.eval()
        self.data_classifier_t.eval()
    z_set_s, y_set_s, z_set_t, y_set_t = extract_input_estim(self, is_phi=is_phi, is_full=(self.proportion_method == "confusion"))
    psy, pty = compute_proportion(self, y_set_s), compute_proportion(self, y_set_t)
    y_s_pred, ptyh_hard, ptyh_soft = extract_pred(self, z_set_s, z_set_t, is_phi=is_phi)
    if self.proportion_method == "cluster":
        prop_estimate = im_weights_cluster(z_set_s.detach().cpu().numpy(), y_set_s.detach().cpu().numpy(), z_set_t.detach().cpu().numpy(), self)
    elif self.proportion_method == "gmm":
        prop_estimate = im_weights_gmm(z_set_s.detach().cpu().numpy(), y_set_s.detach().cpu().numpy(), z_set_t.detach().cpu().numpy(), self)
    elif self.proportion_method == "confusion":
        psy, ptyh_soft = psy.detach().cpu().numpy(), ptyh_soft.detach().numpy()
        cov_mat = extract_cov(self, y_s_pred, y_set_s)
        prop_estimate = im_weights_conf(psy, ptyh_soft, cov_mat, self)
    else:
        raise Exception(f"{self.proportion_method} not known")

    if exp_ave:
        proportion_T = proportion_T * (k_prop - 1) / k_prop + prop_estimate / k_prop
    else:
        if k_prop != 1:
            proportion_T = (1 - self.prop_factor) * proportion_T + self.prop_factor * prop_estimate
        else:
            proportion_T = prop_estimate

    self.feat_extractor.train()
    self.data_classifier.train()
    if is_phi:
        self.phi.train()
        self.data_classifier_t.train()
    return proportion_T


"""
Ground truth estimator
"""


def estimate_source_proportion(data_loader, n_clusters):
    x, y = torch.Tensor().cuda(), torch.LongTensor().cuda()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            x = torch.cat((x, data))
            y = torch.cat((y, target))
    nb_sample_S = torch.tensor([torch.sum(y == i) for i in range(n_clusters)]).float()
    return nb_sample_S / torch.sum(nb_sample_S)


"""
MARSc, MARSg estimator
"""


def im_weights_cluster(z_s, y_s, z_t, self):
    # Clustering target samples in the latent space and extracting prototypes (means of clusters)
    cluster = AgglomerativeClustering(n_clusters=self.n_class, linkage=self.cluster_param)
    label_t = cluster.fit_predict(z_t)
    mean_mat_S, num_in_class_S = extract_prototypes(z_s, y_s, self.n_class)
    mean_mat_T, num_in_class_T = extract_prototypes(z_t, label_t, self.n_class)

    # We assume that prototypes of classes have been transported in some in the feature space
    M = ot.dist(mean_mat_S, mean_mat_T)
    M /= M.max()

    n_1 = self.n_class
    a = np.ones((n_1,)) / n_1
    b = np.ones((n_1,)) / n_1

    gamma = ot.emd(a, b, M)
    proportion_T = num_in_class_T / np.sum(num_in_class_T)
    assignement_source_to_target = gamma.argmax(axis=1)

    # proportions are arranged directly per class
    proportion_T = torch.from_numpy(proportion_T[assignement_source_to_target]).float()

    return proportion_T


def im_weights_gmm(z_s, y_s, z_t, self):
    # Clustering target samples in the latent space and extracting prototypes (means of clusters)
    gmm = GaussianMixture(n_components=self.n_class, n_init=20)
    y_t = gmm.fit_predict(z_t)

    mean_mat_S, num_in_class_S = extract_prototypes(z_s, y_s, self.n_class)
    mean_mat_T, num_in_class_T = extract_prototypes(z_t, y_t, self.n_class)

    # We assume that prototypes of classes have been transported in some in the feature space
    M = ot.dist(mean_mat_S, mean_mat_T)
    M /= M.max()
    a = np.ones((self.n_class,)) / self.n_class
    b = np.ones((self.n_class,)) / self.n_class

    gamma = ot.emd(a, b, M)
    proportion_T = num_in_class_T / np.sum(num_in_class_T)
    assignement_source_to_target = gamma.argmax(axis=1)

    # proportions are arranged directly per class
    proportion_T = torch.from_numpy(proportion_T[assignement_source_to_target]).float()
    if self.logger is not None:
        self.logger.info(f"GMM proportion_T: {proportion_T}, assignment_S2T: {assignement_source_to_target}")

    return proportion_T


"""
Lipton estimator
The algorithm is based on parallel proximal algorithm by Pustelnik et al.
https://hal.archives-ouvertes.fr/hal-00574531v2/document
"""


def prox_pos(x):
    return np.maximum(x, 0)


def prox_b_one(x, A, b):
    aux = np.linalg.inv(A @ A.T)
    return x - A.T @ aux @ A @ x + A.T @ (aux) @ b


def im_weights_conf(psy, ptyh, C, self):
    D_S = np.sum(C, axis=0)
    n_prox = 2
    Z = np.zeros((n_prox, self.n_class))
    z = np.mean(Z, axis=0)
    A = np.expand_dims(D_S, 0)
    b = np.ones(1)
    step = 1 / np.linalg.norm(C)

    prox_sum = lambda x: prox_b_one(x, A, b)
    list_prox = [prox_pos, prox_sum]
    for i in range(100):
        grad = - (ptyh - C @ z)
        for j, prox in enumerate(list_prox):
            aux = 2 * z - Z[j, :] - step * grad
            Z[j, :] = Z[j, :] + prox(aux) - z
        z = np.mean(Z, axis=0)

    return torch.from_numpy(z * psy).float() / torch.sum(torch.from_numpy(z * psy).float())
