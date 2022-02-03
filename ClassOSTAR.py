import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils_local import evaluate_data_classifier, loop_iterable, set_requires_grad, gradient_penalty, format_list, \
    compute_diff_label, set_lr, entropy_loss
from proportion_estimators import estimate_proportion


class OSTAR(object):
    def __init__(self, feat_extractor, data_classifier, phi, source_data_loader, target_data_loader, div_weight=0.1,
                 grad_scale=1.0, cuda=False, logger=None, n_class=10, ts=1, dataset="digits", cluster_step=5,
                 ot_weight=1.0, domain_classifier=None, epoch_start_align=11, cluster_param="ward", epoch_start_g=30,
                 n_epochs=100, gamma=10, init_lr=0.001, proportion_S=None, iter_domain_classifier=10, lr_g_weight=1.0,
                 lr_f_weight=1.0, lr_d_weight=1.0, lr_phi_weight=1.0, eval_data_loader=None, proportion_T_gt=None, iter=0,
                 data_class_t=None, ent_weight=0.1, clf_t_weight=0.3, setting=10, use_div=False):
        self.feat_extractor = feat_extractor
        self.data_classifier = data_classifier
        self.data_classifier_t = data_class_t
        self.phi = phi
        self.setting = setting
        self.iter = iter
        self.proportion_T_gt = proportion_T_gt
        self.domain_classifier = domain_classifier
        self.source_data_loader = source_data_loader
        self.target_data_loader = target_data_loader
        self.eval_data_loader = eval_data_loader
        self.proportion_S = proportion_S
        self.n_class = n_class
        self.ent_weight = ent_weight
        self.lr_g_weight = lr_g_weight
        self.lr_f_weight = lr_f_weight
        self.lr_d_weight = lr_d_weight
        self.lr_phi_weight = lr_phi_weight
        self.ot_weight = ot_weight
        self.ot_weight_init = ot_weight
        self.grad_scale = grad_scale
        self.clf_t_weight = clf_t_weight
        self.cuda = cuda
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.epoch_start_align = epoch_start_align
        self.epoch_start_g = epoch_start_g
        self.iter_domain_classifier = iter_domain_classifier
        self.gamma = gamma
        self.proportion_method = "confusion"
        self.logger = logger
        self.cluster_step = cluster_step
        self.init_lr = init_lr
        self.ts = ts
        self.dataset = dataset
        self.cluster_param = cluster_param
        self.prop_factor = 0.5
        self.optimizer_feat_extractor = optim.SGD(self.feat_extractor.parameters(), lr=0.001)
        self.optimizer_data_classifier = optim.SGD(self.data_classifier.parameters(), lr=0.001)
        self.optimizer_data_classifier_t = optim.SGD(self.data_classifier_t.parameters(), lr=0.001)
        self.optimizer_domain_classifier = optim.SGD(self.domain_classifier.parameters(), lr=0.01)
        self.optimizer_phi = optim.SGD(self.phi.parameters(), lr=0.001)
        self.use_div = use_div
        self.div_weight = div_weight

    def fit(self):
        if self.cuda:
            self.feat_extractor.cuda()
            self.data_classifier.cuda()
            self.data_classifier_t.cuda()
            self.domain_classifier.cuda()
            self.phi.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        k_critic = self.iter_domain_classifier
        k_prop = 1
        gamma = self.gamma
        self.print_start = True
        self.print_start_g = True
        self.use_phi = False

        proportion_T = torch.ones(self.n_class) / self.n_class
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
                    self.logger.info("--Train phi--")
                    self.print_start = False
                if self.print_start_g and epoch >= self.epoch_start_g:
                    self.logger.info("--Train g--")
                    self.print_start_g = False

            dist_loss_tot, ot_loss_tot, clf_loss_tot_s, clf_loss_tot_t, loss_tot, wass_loss_tot, ent_loss_tot = \
                0, 0, 0, 0, 0, 0, 0

            self.feat_extractor.train()
            if self.align:
                self.phi.train()
                self.domain_classifier.train()
                self.data_classifier_t.train()
            else:
                self.data_classifier.train()

            if epoch == self.epoch_start_align:
                self.data_classifier_t.load_state_dict(self.data_classifier.state_dict())

            if self.recluster:
                # Estimate proportion
                self.logger.info(f"is_phi: {self.use_phi} / k_prop: {k_prop}")
                proportion_T = estimate_proportion(self, k_prop=k_prop, proportion_T=proportion_T, is_phi=self.use_phi, comment=f"{self.ts}_conf_estim_{epoch}")
                if epoch >= self.epoch_start_g:
                    k_prop += 1
                elif epoch > self.epoch_start_align + 1:
                    k_prop = 2
                self.hist_proportion = np.vstack((self.hist_proportion, proportion_T.numpy()))
                compute_diff_label(self, self.proportion_T_gt, comment="pT(Y)")

            self.use_phi = self.align

            for batch_idx in range(iterations):
                (x_s, y_s), (x_t, y_t) = next(batch_iterator)
                x_s, x_t, y_s, y_t = x_s.to(self.device), x_t.to(self.device), y_s.to(self.device), y_t.to(self.device)

                ent_loss, dist_loss, ot_loss, clf_s_loss, clf_t_loss, pl_loss, N_tpl = torch.zeros(1).to(self.device),\
                    torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), \
                    torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), 0

                if self.align:
                    # Set lr
                    p = (batch_idx + (epoch - self.epoch_start_align) * len(self.source_data_loader)) / (
                            len(self.source_data_loader) * (self.n_epochs - self.epoch_start_align))
                    lr = float(self.init_lr / (1. + 10 * p) ** 0.75)
                    set_lr(self.optimizer_domain_classifier, lr * self.lr_d_weight)
                    set_lr(self.optimizer_data_classifier_t, lr * self.lr_f_weight)
                    set_lr(self.optimizer_feat_extractor, lr * self.lr_g_weight)
                    set_lr(self.optimizer_phi, lr * self.lr_phi_weight)

                    source_weight_un = torch.zeros((y_s.size(0), 1)).to(self.device)
                    weight_class = torch.zeros((y_s.size(0), 1)).to(self.device)
                    Ns_class = torch.zeros((self.n_class, 1)).to(self.device)
                    Nt_class = torch.zeros((self.n_class, 1)).to(self.device)
                    for j in range(self.n_class):
                        nb_sample = y_s.eq(j).nonzero().size(0)
                        source_weight_un[y_s == j] = proportion_T[j].to(self.device) / nb_sample
                        weight_class[y_s == j] = 1 / nb_sample if nb_sample != 0 else 0
                        Ns_class[j] = nb_sample
                        Nt_class[j] = y_t.eq(j).nonzero().size(0)

                    #######################
                    # Train discriminator #
                    #######################
                    set_requires_grad(self.phi, requires_grad=False)
                    set_requires_grad(self.feat_extractor, requires_grad=False)
                    set_requires_grad(self.domain_classifier, requires_grad=True)
                    for _ in range(k_critic):
                        (x_s_w, y_s_w), (x_t_w, _) = next(batch_iterator_w)
                        x_s_w, x_t_w, y_s_w = x_s_w.to(self.device), x_t_w.to(self.device), y_s_w.to(self.device)
                        source_weight_un_w = torch.zeros((y_s_w.size(0), 1)).to(self.device)
                        for j in range(self.n_class):
                            nb_sample = y_s_w.eq(j).nonzero().size(0)
                            source_weight_un_w[y_s_w == j] = proportion_T[j].to(self.device) / nb_sample
                        with torch.no_grad():
                            z_w = self.feat_extractor(torch.cat((x_s_w, x_t_w), 0))
                            s_w = self.phi(z_w[:x_s_w.shape[0]])[0]
                            t_w = z_w[x_s_w.shape[0]:]
                        gp = gradient_penalty(self.domain_classifier, s_w, t_w, self.cuda)
                        critic_w = self.domain_classifier(torch.cat((s_w, t_w), 0))
                        wasserstein_distance_w = (critic_w[:x_s.shape[0]] * source_weight_un_w.detach()).sum() - critic_w[x_s.shape[0]:].mean()
                        critic_cost = - wasserstein_distance_w + gamma * gp
                        self.optimizer_domain_classifier.zero_grad()
                        critic_cost.backward()
                        self.optimizer_domain_classifier.step()
                        wass_loss_tot += wasserstein_distance_w.item()

                    #############
                    # Train phi #
                    #############
                    set_requires_grad(self.phi, requires_grad=True)
                    set_requires_grad(self.feat_extractor, requires_grad=False)
                    set_requires_grad(self.domain_classifier, requires_grad=False)
                    with torch.no_grad():
                        z_nograd = self.feat_extractor(torch.cat((x_s, x_t), 0))
                        zt_nograd = z_nograd[x_s.shape[0]:]
                    phi_z_s_nograd, rs_nograd = self.phi(z_nograd[:x_s.shape[0]].detach())

                    # Alignment
                    critic = self.domain_classifier(torch.cat((phi_z_s_nograd, zt_nograd.detach()), 0))
                    critic_s = critic[:x_s.shape[0]]
                    critic_t = critic[x_s.shape[0]:]
                    dist_loss = self.grad_scale * ((critic_s * source_weight_un.detach()).sum() - critic_t.mean())

                    # OT penalization
                    ot_loss = sum([torch.sum((torch.abs(r) ** 2) * weight_class.detach()) for r in rs_nograd])

                    loss = dist_loss + ot_loss * self.ot_weight
                    self.optimizer_phi.zero_grad()
                    loss.backward()
                    self.optimizer_phi.step()

                    ###############
                    # Train g, fT #
                    ###############
                    self.train_g = False if (self.epoch_start_g > epoch >= self.epoch_start_align) else True
                    set_requires_grad(self.data_classifier, requires_grad=False)
                    set_requires_grad(self.data_classifier_t, requires_grad=True)
                    set_requires_grad(self.feat_extractor, requires_grad=self.train_g)
                    set_requires_grad(self.phi, requires_grad=False)

                    z = self.feat_extractor(torch.cat((x_s, x_t), 0))
                    z_s, z_t = z[:x_s.shape[0]], z[x_s.shape[0]:]
                    phi_z_s = self.phi(z_s)[0]

                    # Phi
                    self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(proportion_T / self.proportion_S).to(self.device))
                    clf_t_loss = self.criterion(self.data_classifier_t(phi_z_s.detach()), y_s)
                    clf_t_loss *= self.clf_t_weight

                    # Entropy on target
                    output_class_t = self.data_classifier_t(z_t)
                    ent_loss = self.ent_weight * entropy_loss(output_class_t)
                    if self.use_div:
                        msoftmax = nn.Softmax(dim=1)(output_class_t).mean(dim=0)
                        ent_loss -= self.div_weight * torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

                    if self.train_g:
                        # Source
                        self.criterion = nn.CrossEntropyLoss()
                        source_preds = self.data_classifier(z_s)
                        clf_s_loss = self.criterion(source_preds, y_s)

                    loss = clf_s_loss + clf_t_loss + ent_loss
                    if self.train_g:
                        self.optimizer_feat_extractor.zero_grad()
                    self.optimizer_data_classifier_t.zero_grad()
                    loss.backward()
                    if self.train_g:
                        self.optimizer_feat_extractor.step()
                    self.optimizer_data_classifier_t.step()
                else:
                    set_requires_grad(self.data_classifier, requires_grad=True)
                    set_requires_grad(self.feat_extractor, requires_grad=True)
                    z = self.feat_extractor(torch.cat((x_s, x_t), 0))
                    source_preds = self.data_classifier(z[:x_s.shape[0]])
                    self.criterion = nn.CrossEntropyLoss()
                    clf_s_loss = self.criterion(source_preds, y_s)

                    loss = clf_s_loss

                    self.optimizer_feat_extractor.zero_grad()
                    self.optimizer_data_classifier.zero_grad()
                    loss.backward()
                    self.optimizer_feat_extractor.step()
                    self.optimizer_data_classifier.step()

            loss_tot += loss.item()
            clf_loss_tot_s += clf_s_loss.item()
            clf_loss_tot_t += clf_t_loss.item()
            dist_loss_tot += dist_loss.item()
            ot_loss_tot += ot_loss.item()
            ent_loss_tot += ent_loss.item()

            self.logger.info(
                '{} OSTAR {} s{} Iter {} Epoch {}/{} \tTotal: {:.6f} L_S: {:.6f} L_T: {:.6f} DistL:{:.6f} WassL:{:.6f} '
                'OTL:{:.6f} H:{:.6f}'.format(self.ts, self.dataset, self.setting, self.iter, epoch, self.n_epochs,
                loss_tot, clf_loss_tot_s, clf_loss_tot_t, dist_loss_tot, wass_loss_tot, ot_loss_tot, ent_loss_tot))
            if (epoch + 1) % 5 == 0:
                evaluate_data_classifier(self, self.source_data_loader, is_target=False, verbose=False)
                evaluate_data_classifier(self, self.eval_data_loader, is_target=True, is_ft=self.align)
                self.logger.info(f"pT(Y) {self.proportion_method}: {format_list(proportion_T, 4)}")
