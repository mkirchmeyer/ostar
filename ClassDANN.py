import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils_local import loop_iterable, build_label_domain, evaluate_data_classifier, evaluate_domain_classifier, set_lr
from time import clock as tick


class DANN(object):
    def __init__(self, feat_extractor, data_classifier, domain_classifier, source_data_loader, target_data_loader,
                 grad_scale=1, cuda=False, logger_file=None, logger=None, n_epochs=100, epoch_start_align=10,
                 n_class=10, eval_data_loader=None, init_lr=0.001, lr_g_weight=1.0, lr_f_weight=1.0, lr_d_weight=1.0,
                 setting=10, dataset="digits", ts=1, proportion_T_gt=None, iter=0):
        self.eval_data_loader = eval_data_loader
        self.feat_extractor = feat_extractor
        self.data_classifier = data_classifier
        self.source_data_loader = source_data_loader
        self.target_data_loader = target_data_loader
        self.source_domain_label = 1
        self.test_domain_label = 0
        self.cuda = cuda
        self.iter = iter
        self.n_epochs = n_epochs
        self.logger = logger_file
        self.criterion = nn.CrossEntropyLoss()
        self.init_lr = init_lr
        self.epoch_start_align = epoch_start_align
        self.grad_scale = grad_scale
        self.logger = logger
        self.n_class = n_class
        self.lr_g_weight = lr_g_weight
        self.lr_f_weight = lr_f_weight
        self.lr_d_weight = lr_d_weight
        self.setting = setting
        self.dataset = dataset
        self.proportion_T_gt = proportion_T_gt
        self.ts = ts
        _parent_class = self

        class GradReverse(torch.autograd.Function):
            @staticmethod
            def forward(self, x):
                return x.clone()

            @staticmethod
            def backward(self, grad_output):
                return grad_output.neg() * _parent_class.grad_scale

        class GRLDomainClassifier(nn.Module):
            def __init__(self, domain_classifier):
                super(GRLDomainClassifier, self).__init__()
                self.domain_classifier = domain_classifier

            def forward(self, input):
                x = GradReverse.apply(input)
                x = self.domain_classifier.forward(x)
                return x

        self.grl_domain_classifier = GRLDomainClassifier(domain_classifier)
        self.optimizer_feat_extractor = optim.SGD(self.feat_extractor.parameters(), lr=0.001)
        self.optimizer_data_classifier = optim.SGD(self.data_classifier.parameters(), lr=0.001)
        self.optimizer_domain_classifier = optim.SGD(self.grl_domain_classifier.parameters(), lr=0.1)

    def fit(self):
        if self.cuda:
            self.feat_extractor.cuda()
            self.data_classifier.cuda()
            self.grl_domain_classifier.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        for epoch in range(self.n_epochs):
            self.align = (epoch >= self.epoch_start_align)
            self.feat_extractor.train()
            self.data_classifier.train()
            if self.align:
                self.grl_domain_classifier.train()

            tic = tick()
            batch_iterator = zip(loop_iterable(self.source_data_loader), loop_iterable(self.target_data_loader))
            iterations = len(self.source_data_loader)
            for batch_idx in range(iterations):
                (x_s, y_s), (x_t, y_t) = next(batch_iterator)
                x_s, x_t, y_s, y_t = x_s.to(self.device), x_t.to(self.device), y_s.to(self.device), y_t.to(self.device)
                dist_loss = torch.zeros(1).to(self.device)

                # Classif
                z = self.feat_extractor(torch.cat((x_s, x_t), 0))
                z_s, z_t = z[:x_s.shape[0]], z[x_s.shape[0]:]
                output_class_source = self.data_classifier(z_s)
                clf_s_loss = F.cross_entropy(output_class_source, y_s)

                if self.align:
                    # Set lr
                    p = (batch_idx + (epoch - self.epoch_start_align) * len(self.source_data_loader)) / (
                            len(self.source_data_loader) * (self.n_epochs - self.epoch_start_align))
                    lr = float(self.init_lr / (1. + 10 * p) ** 0.75)
                    set_lr(self.optimizer_domain_classifier, lr * self.lr_d_weight)
                    set_lr(self.optimizer_data_classifier, lr * self.lr_f_weight)
                    set_lr(self.optimizer_feat_extractor, lr * self.lr_g_weight)

                    output_domain = self.grl_domain_classifier(torch.cat((z_s, z_t), 0))
                    label_domain_s = build_label_domain(self, x_s.size(0), self.source_domain_label).to(self.device)
                    error_source_data = F.cross_entropy(output_domain[:x_s.shape[0]], label_domain_s)

                    label_domain_t = build_label_domain(self, x_t.size(0), self.test_domain_label).to(self.device)
                    error_test_data = F.cross_entropy(output_domain[x_s.shape[0]:], label_domain_t)
                    dist_loss = (error_source_data + error_test_data)

                loss = clf_s_loss + dist_loss
                self.optimizer_feat_extractor.zero_grad()
                self.optimizer_data_classifier.zero_grad()
                if self.align:
                    self.optimizer_domain_classifier.zero_grad()
                loss.backward()
                self.optimizer_feat_extractor.step()
                self.optimizer_data_classifier.step()
                if self.align:
                    self.optimizer_domain_classifier.step()

            toc = tick() - tic
            self.logger.info('{} DANN {} s{} Iter {} Epoch: {}/{} {:2.2f}s Loss: {:.6f} clf_s_loss: {:.6f} dist:{:.6f}'.format(
                self.ts, self.dataset, self.setting, self.iter, epoch, self.n_epochs, toc, loss.item(), clf_s_loss.item(), dist_loss.item()))
            if epoch % 5 == 0:
                evaluate_data_classifier(self, self.source_data_loader, is_target=False, verbose=False)
                evaluate_data_classifier(self, self.eval_data_loader, is_target=True)
                evaluate_domain_classifier(self)
