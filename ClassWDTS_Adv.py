import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils_local import evaluate_data_classifier, loop_iterable, set_requires_grad, gradient_penalty, format_list, \
    compute_diff_label, set_lr
from proportion_estimators import estimate_proportion


class WDTS_Adv(object):
    def __init__(self, feat_extractor, data_classifier, domain_classifier, source_data_loader, target_data_loader,
                 grad_scale=1.0, cuda=False, logger=None, n_class=10, ts=1, dataset="digits", compute_cluster_every=5,
                 epoch_start_align=11, cluster_param="ward", epoch_start_g=30, n_epochs=100, gamma=10, init_lr=0.001, iter=0,
                 iter_domain_classifier=10, factor_f=1, lr_g_weight=1.0, lr_f_weight=1.0, lr_d_weight=1.0, factor_g=1.0,
                 eval_data_loader=None, proportion_T_gt=None, setting=10, beta_ratio=-1, proportion_S=None):
        self.feat_extractor = feat_extractor
        self.data_classifier = data_classifier
        self.setting = setting
        self.iter = iter
        self.proportion_T_gt = proportion_T_gt
        self.domain_classifier = domain_classifier
        self.source_data_loader = source_data_loader
        self.target_data_loader = target_data_loader
        self.eval_data_loader = eval_data_loader
        self.n_class = n_class
        self.lr_g_weight = lr_g_weight
        self.lr_f_weight = lr_f_weight
        self.lr_d_weight = lr_d_weight
        self.proportion_S = proportion_S
        self.cuda = cuda
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.epoch_start_align = epoch_start_align
        self.epoch_start_g = epoch_start_g
        self.iter_domain_classifier = iter_domain_classifier
        self.gamma = gamma
        self.grad_scale = grad_scale
        self.proportion_method = "confusion"
        self.logger = logger
        self.cluster_step = compute_cluster_every
        self.init_lr = init_lr
        self.ts = ts
        self.dataset = dataset
        self.cluster_param = cluster_param
        self.prop_factor = 0.5
        self.factor_f = factor_f
        self.factor_g = factor_g
        self.optimizer_feat_extractor = optim.SGD(self.feat_extractor.parameters(), lr=0.001)
        self.optimizer_data_classifier = optim.SGD(self.data_classifier.parameters(), lr=0.001)
        self.optimizer_domain_classifier = optim.SGD(self.domain_classifier.parameters(), lr=0.01)
        self.beta_ratio = beta_ratio

    def fit(self):
        if self.cuda:
            self.feat_extractor.cuda()
            self.data_classifier.cuda()
            self.domain_classifier.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        k_critic = self.iter_domain_classifier
        k_prop = 1
        gamma = self.gamma
        self.print_start = True
        self.print_start_g = True

        proportion_T = torch.ones(self.n_class) / self.n_class
        if self.beta_ratio == -1:
            self.hist_proportion = proportion_T.numpy()

        # Train latent space
        self.logger.info("--Initialize f, g--")
        for epoch in range(self.n_epochs):
            self.recluster = ((self.epoch_start_g > epoch > self.epoch_start_align and ((epoch - self.epoch_start_align) % 2) == 0) or
                              (epoch == self.epoch_start_g) or (epoch >= self.epoch_start_g and (epoch % self.cluster_step) == 0))
            self.align = (epoch >= self.epoch_start_align)

            S_batches = loop_iterable(self.source_data_loader)
            batch_iterator = zip(S_batches, loop_iterable(self.target_data_loader))
            batch_iterator_w = zip(S_batches, loop_iterable(self.target_data_loader))
            iterations = len(self.source_data_loader)

            if self.align:
                if self.print_start:
                    self.logger.info("--Start Alignment--")
                    self.print_start = False

            dist_loss_tot, clf_loss_tot_s, loss_tot, wass_loss_tot = 0, 0, 0, 0
            self.feat_extractor.train()
            self.data_classifier.train()
            if self.align:
                self.domain_classifier.train()

            if self.recluster and self.beta_ratio == -1:
                # Estimate proportion
                if self.proportion_method == "gt":
                    proportion_T = self.proportion_T_gt
                else:
                    self.logger.info(f"k_prop: {k_prop}")
                    proportion_T = estimate_proportion(self, k_prop=k_prop, proportion_T=proportion_T, comment=f"{self.ts}_adv_estim_{epoch}")
                    if epoch >= self.epoch_start_g:
                        k_prop += 1
                    elif epoch > self.epoch_start_align + 1:
                        k_prop = 2
                self.hist_proportion = np.vstack((self.hist_proportion, proportion_T.numpy()))
                compute_diff_label(self, self.proportion_T_gt, comment=f"pT(Y) {self.proportion_method}")

            for batch_idx in range(iterations):
                (x_s, y_s), (x_t, y_t) = next(batch_iterator)
                x_s, x_t, y_s, y_t = x_s.to(self.device), x_t.to(self.device), y_s.to(self.device), y_t.to(self.device)

                dist_loss, clf_s_loss = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
                if self.align:
                    # Set lr
                    p = (batch_idx + (epoch - self.epoch_start_align) * len(self.source_data_loader)) / (
                            len(self.source_data_loader) * (self.n_epochs - self.epoch_start_align))
                    lr = float(self.init_lr / (1. + 10 * p) ** 0.75)
                    set_lr(self.optimizer_domain_classifier, lr * self.lr_d_weight)
                    set_lr(self.optimizer_data_classifier, lr * self.lr_f_weight)
                    set_lr(self.optimizer_feat_extractor, lr * self.lr_g_weight)

                    if self.beta_ratio == -1:
                        source_weight_un = torch.zeros((y_s.size(0), 1)).to(self.device)
                    Ns_class = torch.zeros((self.n_class, 1)).to(self.device)
                    Nt_class = torch.zeros((self.n_class, 1)).to(self.device)
                    for j in range(self.n_class):
                        nb_sample = y_s.eq(j).nonzero().size(0)
                        if self.beta_ratio == -1:
                            source_weight_un[y_s == j] = proportion_T[j].to(self.device) / nb_sample
                        Ns_class[j] = nb_sample
                        Nt_class[j] = y_t.eq(j).nonzero().size(0)

                    #######################
                    # Train discriminator #
                    #######################
                    set_requires_grad(self.feat_extractor, requires_grad=False)
                    set_requires_grad(self.domain_classifier, requires_grad=True)
                    for kk in range(k_critic):
                        (x_s_w, y_s_w), (x_t_w, _) = next(batch_iterator_w)
                        x_s_w, x_t_w, y_s_w = x_s_w.to(self.device), x_t_w.to(self.device), y_s_w.to(self.device)
                        if self.beta_ratio == -1:
                            source_weight_un_w = torch.zeros((y_s_w.size(0), 1)).to(self.device)
                            for j in range(self.n_class):
                                nb_sample = y_s_w.eq(j).nonzero().size(0)
                                source_weight_un_w[y_s_w == j] = proportion_T[j].to(self.device) / nb_sample
                        with torch.no_grad():
                            z_w = self.feat_extractor(torch.cat((x_s_w, x_t_w), 0))
                            s_w = z_w[:x_s_w.shape[0]]
                            t_w = z_w[x_s_w.shape[0]:]
                        gp = gradient_penalty(self.domain_classifier, s_w, t_w, self.cuda)
                        critic_w = self.domain_classifier(torch.cat((s_w, t_w), 0))
                        critic_s_w, critic_t_w = critic_w[:x_s.shape[0]], critic_w[x_s.shape[0]:]
                        if self.beta_ratio == -1:
                            wasserstein_distance_w = (critic_s_w * source_weight_un_w.detach()).sum() - critic_t_w.mean()
                        else:
                            wasserstein_distance_w = (critic_s_w.mean() - (1 + self.beta_ratio) * critic_t_w.mean())
                        critic_cost = - wasserstein_distance_w + gamma * gp
                        self.optimizer_domain_classifier.zero_grad()
                        critic_cost.backward()
                        self.optimizer_domain_classifier.step()
                        wass_loss_tot += wasserstein_distance_w.item()

                    ##############
                    # Train f, g #
                    ##############
                    set_requires_grad(self.data_classifier, requires_grad=True)
                    set_requires_grad(self.feat_extractor, requires_grad=True)
                    set_requires_grad(self.domain_classifier, requires_grad=False)
                    z = self.feat_extractor(torch.cat((x_s, x_t), 0))

                    # Classif
                    if self.beta_ratio == -1:
                        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(proportion_T / self.proportion_S).to(self.device))
                    else:
                        self.criterion = nn.CrossEntropyLoss()
                    clf_s_loss = self.criterion(self.data_classifier(z[:x_s.shape[0]]), y_s)

                    # Alignment
                    critic = self.domain_classifier(z)
                    critic_s, critic_t = critic[:x_s.shape[0]], critic[x_s.shape[0]:]
                    if self.beta_ratio == -1:
                        dist_loss = self.grad_scale * ((critic_s * source_weight_un.detach()).sum() - critic_t.mean())
                    else:
                        dist_loss = self.grad_scale * (critic_s.mean() - (1 + self.beta_ratio) * critic_t.mean())

                    loss = clf_s_loss + dist_loss
                    self.optimizer_data_classifier.zero_grad()
                    self.optimizer_feat_extractor.zero_grad()
                    loss.backward()
                    self.optimizer_data_classifier.step()
                    self.optimizer_feat_extractor.step()
                else:
                    set_requires_grad(self.data_classifier, requires_grad=True)
                    set_requires_grad(self.feat_extractor, requires_grad=True)
                    z = self.feat_extractor(torch.cat((x_s, x_t), 0))
                    self.criterion = nn.CrossEntropyLoss()
                    clf_s_loss = self.criterion(self.data_classifier(z[:x_s.shape[0]]), y_s)

                    loss = clf_s_loss

                    self.optimizer_feat_extractor.zero_grad()
                    self.optimizer_data_classifier.zero_grad()
                    loss.backward()
                    self.optimizer_feat_extractor.step()
                    self.optimizer_data_classifier.step()

            loss_tot += loss.item()
            clf_loss_tot_s += clf_s_loss.item()
            dist_loss_tot += dist_loss.item()

            comment = f"WDTS_ADV {self.proportion_method}" if self.beta_ratio == -1 else f"WDGRL b={self.beta_ratio}"
            self.logger.info('{} {} {} s{} Iter {} Epoch {}/{} \tTotal: {:.6f} L_S: {:.6f} DistL:{:.6f} WassL:{:.6f}'.format(self.ts,
                comment, self.dataset, self.setting, self.iter, epoch, self.n_epochs, loss_tot, clf_loss_tot_s, dist_loss_tot, wass_loss_tot))
            if (epoch + 1) % 5 == 0:
                evaluate_data_classifier(self, self.source_data_loader, is_target=False, verbose=False)
                evaluate_data_classifier(self, self.eval_data_loader, is_target=True)
                if self.beta_ratio == -1:
                    self.logger.info(f"pT(Y) {self.proportion_method}: {format_list(proportion_T, 4)}")
