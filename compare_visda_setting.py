from compare_digits_models import ResidualPhi
from compare_visda_models import DataClassifierVisda, FeatureExtractorVisda, DomainClassifierVisda, DomainClassifierDANNVisda
from data_shift.office31_shift import get_ratio
from data_shift.visda_shift import get_visda_shift
import numpy as np


# -------------------------------------------------------------------------
#                       VisDA12
# -------------------------------------------------------------------------


def visda_expe_setting(setting, logger):
    opt = {}
    n_blocks = 10
    n_hidden = 100
    opt['phi'] = ResidualPhi(nblocks=n_blocks, dim=n_hidden, nl_layer="relu", norm_layer="batch1d")
    is_full = False
    batch_size = 200
    input_sample_s = input_sample_t = 9600
    opt['lr'] = opt['lr_beta_0'] = 0.0005
    opt['lr_beta_1'] = 0.001
    opt['grad_scale'] = 0.005
    opt['grad_scale_beta_0'] = 1.0
    opt['grad_scale_beta_1'] = 0.001
    opt['lr_dann'] = 0.001
    opt['grad_scale_dann'] = 0.01
    opt['feat_extract'] = FeatureExtractorVisda(n_hidden)
    opt['data_classifier'] = DataClassifierVisda(n_hidden)
    opt['data_classifier_t'] = DataClassifierVisda(n_hidden)
    opt['domain_classifier'] = DomainClassifierVisda(n_hidden)
    opt['domain_classifier_dann'] = DomainClassifierDANNVisda(n_hidden)
    opt['nb_class'] = 12
    opt['nb_epoch_alg'] = 100
    opt['wdgrl_grad_down_factor'] = 10
    opt['iter_domain'] = 5
    opt['start_align'] = 5
    opt['batch_size'] = batch_size
    opt['cluster_param'] = 'ward'
    opt['cluster_every'] = 5
    opt['gamma_wdts'] = 10

    if setting == 1:
        ratio_s, total_sample_s, nb_sample_r = get_ratio('visda_train', r_min=1.0, r_max=1.0, cut=6)
        ratio_t, total_sample_t, nb_sample_r = get_ratio('visda_val', r_min=1.0, r_max=1.0, cut=6)
    if setting == 2:
        ratio_s, total_sample_s, nb_sample_r = get_ratio('visda_train', r_min=0.3, r_max=1.0, cut=6)
        ratio_t, total_sample_t, nb_sample_r = get_ratio('visda_val', r_min=1.0, r_max=1.0, cut=6)
    opt['source_loader'] = get_visda_shift(train=True, batch_size=batch_size, drop_last=True,
                                           total_sample=total_sample_s if is_full else input_sample_s, ratio=ratio_s,
                                           classe_vec=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], logger=logger)
    opt['target_loader'] = get_visda_shift(train=False, batch_size=batch_size, drop_last=True,
                                           total_sample=total_sample_t if is_full else input_sample_t, ratio=ratio_t,
                                           classe_vec=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], logger=logger)
    opt['eval_loader'] = opt['target_loader']

    return opt, np.array(ratio_s), np.array(ratio_t)
