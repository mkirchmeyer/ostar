import torch
import logging
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch import nn
from torch.nn import init
from torch.autograd import grad
from functools import partial
from logging.handlers import RotatingFileHandler
from scipy.spatial import distance
import torch.nn.functional as F


#################
# Various utils #
#################


def loop_iterable(iterable):
    while True:
        yield from iterable


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def extract_prototypes(X, y, n_clusters):
    n_hidden = X.shape[1]
    mean_mat = np.zeros((n_clusters, n_hidden))
    number_in_class = np.zeros(n_clusters)
    for i in range(n_clusters):
        mean_mat[i] = np.mean(X[y == i, :], axis=0)
        number_in_class[i] = np.sum(y == i)
    return mean_mat, number_in_class


def extract_input(logger, data_loader):
    x, y = torch.Tensor().cuda(), torch.LongTensor().cuda()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            x = torch.cat((x, data))
            y = torch.cat((y, target))
    return x, y


def format_list(list, higher_pres=2):
    if higher_pres == 4:
        return [float("%.4f" % item) for item in list]
    if higher_pres == 3:
        return [float("%.3f" % item) for item in list]
    if higher_pres == 1:
        return [float("%.1f" % item) for item in list]
    return [float("%.2f" % item) for item in list]


def compute_diff_label(self, proportion_gt, comment=""):
    proportion_T = torch.from_numpy(self.hist_proportion[-1, :]).float()
    mse_label = torch.dist(proportion_gt, proportion_T, 2).cpu().detach().numpy()
    js_label = distance.jensenshannon(proportion_gt.cpu().detach().numpy(), proportion_T.cpu().detach().numpy())
    if comment != "":
        self.logger.info(f"{comment}: {format_list(proportion_T, 4)} / MSE: {mse_label:.4f} / JS: {js_label:.4f}")
    return mse_label, js_label


def gradient_penalty(critic, h_s, h_t, cuda):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    alpha = torch.rand(h_s.size(0), 1)
    alpha = (alpha.expand(h_s.size())).to(device)
    differences = h_t - h_s

    interpolates = (h_s + (alpha * differences))
    interpolates = torch.cat((interpolates, h_s, h_t), dim=0).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates, grad_outputs=torch.ones_like(preds), retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


def create_logger(folder, outfile):
    try:
        os.makedirs(folder)
        print(f"Directory {folder} created")
    except FileExistsError:
        print(f"Directory {folder} already exists replacing files in this notebook")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = RotatingFileHandler(outfile, "w")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.DEBUG)
    logger.addHandler(steam_handler)

    return logger


def m_std(value, it=None, is_list=False):
    if it is None:
        if is_list:
            M_v, S_v = 100 * value.mean(), 100 * value.std()
        else:
            M_v, S_v = 100 * value.mean(axis=0), 100 * value.std(axis=0)
    else:
        if is_list:
            M_v, S_v = 100 * value[:it + 1, :].mean(axis=0), 100 * value[:it + 1, :].std(axis=0)
        else:
            M_v, S_v = 100 * value[:it + 1].mean(), 100 * value[:it + 1].std()
    return M_v, S_v


def weights_init_digits(net, name='normal', gain=0.02, gain_bn=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if name == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif name == 'xavier-n':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif name == 'xavier-u':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif name == 'kaiming-n':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif name == 'kaiming-u':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif name == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif name == "phi":
                init.uniform_(m.weight, -1e-3, 1e-3)
                init.uniform_(m.bias, -1e-3, 1e-3)
            else:
                raise NotImplementedError(f'initialization method {name} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            init.constant_(m.weight.data, gain_bn)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'batch1d':
        norm_layer = partial(nn.BatchNorm1d, affine=True)
    elif norm_type == 'instance':
        norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'instance1d':
        norm_layer = partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        return partial(nn.ReLU, inplace=True)
    if layer_type == 'lrelu':
        return partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    if layer_type == 'elu':
        return partial(nn.ELU, inplace=True)
    if layer_type == 'tanh':
        return nn.Tanh
    if layer_type == 'sigmoid':
        return partial(nn.Sigmoid)
    else:
        raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)

##############
# Evaluation #
##############


def evaluate_data_classifier(self, data_loader, verbose=True, is_target=False, is_ft=False):
    self.feat_extractor.eval()
    self.data_classifier.eval()
    if is_ft:
        self.data_classifier_t.eval()
    accur, MAP = compute_scores(self, data_loader, verbose, is_target, is_ft=is_ft)
    return accur, MAP


def compute_scores(self, data_loader, verbose=True, is_target=False, is_ft=False):
    comments = "T" if is_target else "S"
    test_loss, correct, ot_loss = 0, 0, 0
    y_pred, y_true = torch.Tensor(), torch.Tensor()
    if verbose:
        confusion_matrix_array = np.zeros((self.n_class, self.n_class))
    for data, target in data_loader:
        data, target = data.to(self.device), target.to(self.device)
        output_feat = self.feat_extractor(data)
        output = self.data_classifier_t(output_feat) if (is_ft and is_target) else self.data_classifier(output_feat)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        y_pred = torch.cat((y_pred, pred.float().cpu()))
        y_true = torch.cat((y_true, target.float().cpu()))
        test_loss += self.criterion(output, target).item()
        if verbose:
            try:
                confusion_matrix_array += confusion_matrix(target.data.cpu().numpy(), pred.cpu().numpy(),
                                                           labels=np.arange(self.n_class))
            except Exception as e:
                self.logger.info(f"Error in confusion matrix: {e}")
        if not is_target and is_ft:
            weight_class = torch.zeros((target.size(0), 1)).to(self.device)
            for j in range(self.n_class):
                nb_sample = target.eq(j).nonzero().size(0)
                weight_class[target == j] = 1 / nb_sample if nb_sample != 0 else 0
            _, rs_nograd = self.phi(output_feat)
            ot_loss += sum([torch.sum((torch.abs(r) ** 2) * weight_class.detach()) for r in rs_nograd]).item()

    MAP = balanced_accuracy_score(y_true, y_pred)
    test_loss /= len(data_loader)  # loss function already averages over batch size
    accur = correct.item() / len(data_loader.dataset)
    if not is_target and is_ft:
        self.logger.info(f"OT: {ot_loss} / OT batch: {ot_loss / len(data_loader)} / OT dataset: {ot_loss / len(data_loader.dataset)}")

    if self.logger is not None:
        self.logger.info(
            '{} Mean Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), MAP:{:.4f}'.format(
                comments, test_loss, correct, len(data_loader.dataset), 100 * accur, MAP))
        if verbose:
            try:
                self.logger.info('{} Per Class Acc: {}'.format(
                    comments, format_list(confusion_matrix_array.diagonal() / confusion_matrix_array.sum(axis=1))))
                self.logger.info('P vs R p(Y): {} / {}'.format(
                    format_list(confusion_matrix_array.sum(axis=0) / confusion_matrix_array.sum()),
                    format_list(confusion_matrix_array.sum(axis=1) / confusion_matrix_array.sum())))
            except Exception:
                pass
    return accur, MAP


def build_label_domain(self, size, label):
    label_domain = torch.LongTensor(size)
    if self.cuda:
        label_domain = label_domain.cuda()

    label_domain.data.resize_(size).fill_(label)
    return label_domain


def evaluate_domain_classifier_class(self, data_loader, domain_label):
    self.feat_extractor.eval()
    self.data_classifier.eval()
    self.grl_domain_classifier.eval()

    loss = 0
    correct = 0
    for data, _ in data_loader:
        target = build_label_domain(self, data.size(0), domain_label)
        if self.cuda:
            data, target = data.cuda(), target.cuda()
        output_feat = self.feat_extractor(data)
        output = self.grl_domain_classifier(output_feat)
        loss += self.criterion(output, target).item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    return loss, correct


def evaluate_domain_classifier(self):
    self.feat_extractor.eval()
    self.data_classifier.eval()
    self.grl_domain_classifier.eval()

    test_loss, correct = evaluate_domain_classifier_class(self, self.source_data_loader, self.source_domain_label)
    loss, correct_a = evaluate_domain_classifier_class(self, self.eval_data_loader, self.test_domain_label)
    test_loss += loss
    correct += correct_a
    nb_source = len(self.source_data_loader.dataset)
    nb_target = len(self.eval_data_loader.dataset)
    nb_tot = nb_source + nb_target
    print('Domain: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, (nb_source + nb_target),
        100. * correct / (nb_source + nb_target)))
    if self.logger is not None:
        self.logger.info('Domain: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, (nb_tot),
            100. * correct / nb_tot))
    return correct / nb_tot

###################
# Setter / getter #
###################


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return optimizer


def set_optimizer_data_classifier(self, optimizer):
    self.optimizer_data_classifier = optimizer


def set_optimizer_data_classifier_t(self, optimizer):
    self.optimizer_data_classifier_t = optimizer


def set_optimizer_domain_classifier(self, optimizer):
    self.optimizer_domain_classifier = optimizer


def set_optimizer_phi(self, optimizer):
    self.optimizer_phi = optimizer


def set_optimizer_feat_extractor(self, optimizer):
    self.optimizer_feat_extractor = optimizer


def set_proportion_method(self, method):
    self.proportion_method = method
    print('cluster_method', method)


def entropy_loss(v):
    """
    Entropy loss for probabilistic prediction vectors
    """
    return torch.mean(torch.sum(- F.softmax(v, dim=1) * F.log_softmax(v, dim=1), 1))


def compute_proportion(self, y_set, comment=""):
    number_in_class = torch.zeros(self.n_class)
    for i in range(self.n_class):
        number_in_class[i] = torch.sum(y_set == i)
    proportion = number_in_class / torch.sum(number_in_class)
    if comment != "":
        self.logger.info(f"{comment}={format_list(proportion, 4)}")
    return proportion
