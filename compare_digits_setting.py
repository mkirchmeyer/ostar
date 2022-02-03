from compare_digits_models import FeatureExtractorDigits, DomainClassifierDANNDigits, DomainClassifierDigits, \
    DataClassifierDigits, ResidualPhi
from data_shift.mnist_shift import get_mnist_shift
from data_shift.usps_shift import get_usps_shift
import numpy as np


def digits_expe_setting(setting, logger=None):
    opt = {}
    n_hidden = 128
    opt['feat_extract'] = FeatureExtractorDigits(channel=1, output_dim=n_hidden)
    opt['data_classifier'] = DataClassifierDigits(n_class=10, input_size=n_hidden)
    opt['data_classifier_t'] = DataClassifierDigits(n_class=10, input_size=n_hidden)
    opt['domain_classifier'] = DomainClassifierDigits(input_size=n_hidden)
    opt['domain_classifier_dann'] = DomainClassifierDANNDigits(input_size=n_hidden)
    opt['phi'] = ResidualPhi(nblocks=10, dim=n_hidden, nl_layer="relu", norm_layer="batch1d")

    image_size = 32
    is_full = True
    batch_size = 200
    opt['lr'] = opt['lr_beta_0'] = opt['lr_beta_1'] = opt['lr_dann'] = 0.0001
    opt['grad_scale'] = opt['grad_scale_dann'] = opt['grad_scale_beta_0'] = opt['grad_scale_beta_1'] = 0.01
    opt['wdgrl_grad_down_factor'] = 10
    opt['n_epochs_alg'] = 100
    opt['nb_epoch_alg'] = 100
    opt['iter_domain'] = 5
    opt['batch_size'] = batch_size
    opt['cluster_param'] = 'ward'
    opt['nb_class'] = 10
    opt['cluster_every'] = 5
    opt['gamma_wdts'] = 10
    opt['start_align'] = 11

    # -------------------------------------------------------------------------
    #                       MNIST-USPS
    # -------------------------------------------------------------------------
    if setting < 10:
        n_sample_mnist = 10000 if is_full else 5000
        n_sample_usps = get_ratio_digits("usps", ratio=[0.07, 0.07, 0.07, 0.07, 0.22, 0.22, 0.07, 0.07, 0.07, 0.07])[0] if is_full else 3000

        if setting == 1:  # balanced
            ratio_s = ratio_t = [0.1] * 10
        if setting == 2:  # mild
            ratio_s = [0.1] * 10
            ratio_t = [0.06, 0.06, 0.06, 0.06, 0.2, 0.2, 0.06, 0.1, 0.1, 0.1]
        if setting == 3:  # high
            ratio_s = [0.1] * 10
            ratio_t = [0.07, 0.07, 0.07, 0.07, 0.22, 0.22, 0.07, 0.07, 0.07, 0.07]

        opt['source_loader'] = get_mnist_shift(train=True, batch_size=batch_size, ratio=ratio_s, num_channel=1,
                                               image_size=image_size, logger=logger, total_sample=n_sample_mnist)
        opt['target_loader'] = get_usps_shift(train=True, batch_size=batch_size, total_sample=n_sample_usps, ratio=ratio_t,
                                              image_size=image_size, logger=logger)
        opt['eval_loader'] = get_usps_shift(train=True, batch_size=batch_size, total_sample=n_sample_usps, ratio=ratio_t,
                                            image_size=image_size, logger=logger, drop_last=False)
        return opt, np.array(ratio_s), np.array(ratio_t)

    # -------------------------------------------------------------------------
    #                       USPS-MNIST
    # -------------------------------------------------------------------------
    elif 10 <= setting <= 20:  # USPS - MNIST
        opt['lr_dann'] = 0.001
        n_sample_usps = get_ratio_digits("usps", ratio=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])[0] if is_full else 2000
        n_sample_mnist = 20000 if is_full else 10000

        if setting == 10:  # balanced
            ratio_s = ratio_t = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        if setting == 11:  # mild
            ratio_s = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            ratio_t = [0.06, 0.06, 0.06, 0.06, 0.2, 0.2, 0.06, 0.1, 0.1, 0.1]
        if setting == 12:  # high
            ratio_s = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            ratio_t = [0.07, 0.07, 0.07, 0.07, 0.22, 0.22, 0.07, 0.07, 0.07, 0.07]

        opt['source_loader'] = get_usps_shift(train=True, batch_size=batch_size, drop_last=True, ratio=ratio_s,
                                              image_size=image_size, logger=logger, total_sample=n_sample_usps)
        opt['target_loader'] = get_mnist_shift(train=True, batch_size=batch_size, drop_last=True, ratio=ratio_t,
                                               num_channel=1, total_sample=n_sample_mnist, image_size=image_size, logger=logger)
        opt['eval_loader'] = get_mnist_shift(train=True, batch_size=batch_size, drop_last=False, ratio=ratio_t,
                                             num_channel=1, total_sample=n_sample_mnist, image_size=image_size, logger=logger)
        return opt, np.array(ratio_s), np.array(ratio_t)


def get_ratio_digits(data_name, ratio=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    usps = [1229, 1019, 733, 673, 680, 574, 677, 632, 570, 651]  # 7438
    mnist = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]  # 60000

    if data_name is 'usps':
        data = usps
    elif data_name is 'mnist':
        data = mnist
    initial_total_sample = np.sum(data)

    total_sample = initial_total_sample
    for i in range(len(data)):
        total_sample = np.minimum(int(data[i] / ratio[i]), total_sample)
    nb_sample = np.array(ratio) * total_sample
    print(f"total_sample {data_name}: {total_sample}")
    print(f"nb_sample {data_name}: {nb_sample} / sum={np.sum(nb_sample)}")
    print(f"ratio={nb_sample / np.sum(nb_sample)}")

    return total_sample, nb_sample