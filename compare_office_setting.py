from compare_digits_models import ResidualPhi
from compare_office_models import DomainClassifierOffice, DomainClassifierDANNOffice, DataClassifierOffice, \
    FeatureExtractorOffice
from data_shift.office31_shift import get_office31_shift, get_ratio
from data_shift.officehome_shift import get_officehome_shift, get_ratio_home


def office_expe_setting(setting, logger):
    opt = {}
    n_class = 31 if setting < 10 else 65
    r_min = 0.3
    r_max = 1.0
    batch_size = 200
    n_hidden = 256
    opt['lr'] = opt['lr_dann'] = 0.001
    opt['grad_scale'] = opt['grad_scale_beta_0'] = opt['grad_scale_beta_1'] = opt['grad_scale_dann'] = 0.001
    opt['lr_beta_0'] = opt['lr_beta_1'] = 0.001
    opt['nb_epoch_alg'] = 100
    opt['start_align'] = 21
    opt['feat_extract'] = FeatureExtractorOffice()
    opt['data_classifier'] = DataClassifierOffice(class_num=n_class)
    opt['data_classifier_t'] = DataClassifierOffice(class_num=n_class)
    opt['domain_classifier'] = DomainClassifierOffice(input_size=n_hidden)
    opt['domain_classifier_dann'] = DomainClassifierDANNOffice(input_size=n_hidden)
    opt['phi'] = ResidualPhi(nblocks=10, dim=n_hidden, nl_layer="relu", norm_layer="batch1d")
    opt['wdgrl_grad_down_factor'] = 10
    opt['iter_domain'] = 5
    opt['batch_size'] = batch_size
    opt['cluster_param'] = 'ward'
    opt['nb_class'] = n_class
    opt['cluster_every'] = 10
    opt['gamma_wdts'] = 10

    if 10 <= setting < 13:
        opt['grad_scale_beta_1'] = 0.005
        opt['grad_scale_beta_0'] = 0.01
        opt['grad_scale_dann'] = 0.05
        opt['lr'] = 0.0005
        opt['lr_beta_0'] = opt['lr_beta_1'] = 0.005
    elif 13 <= setting < 16:
        opt['grad_scale'] = 0.005
        opt['grad_scale_beta_0'] = opt['grad_scale_beta_1'] = 0.01
        opt['grad_scale_dann'] = 0.05
        opt['lr'] = 0.0001
        opt['lr_beta_1'] = opt['lr_beta_0'] = opt['lr_dann'] = 0.005
    elif 16 <= setting < 19:
        opt['grad_scale'] = 0.01
        opt['grad_scale_dann'] = 0.05
        opt['lr'] = 0.0005
        opt['lr_beta_1'] = opt['lr_beta_0'] = 0.0005
        opt['lr_dann'] = 0.005
    elif 19 <= setting < 22:
        opt['lr'] = opt['lr_dann'] = 0.005
        opt['grad_scale'] = 0.005

    # -------------------------------------------------------------------------
    #                       Office31
    # -------------------------------------------------------------------------
    if setting == 1:
        ratio_s, total_sample_s, nb_sample_r = get_ratio('amazon', r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio('dslr', r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_office31_shift('amazon_amazon', batch_size=batch_size, drop_last=True,
                                                  total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_office31_shift('amazon_dslr', batch_size=batch_size, drop_last=True,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_office31_shift('amazon_dslr', batch_size=batch_size, drop_last=False,
                                                total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        return opt, ratio_s, ratio_t

    if setting == 2:
        ratio_s, total_sample_s, nb_sample_r = get_ratio('dslr', r_min=r_min, r_max=r_max)  # , r_min=0.3, r_max=0.85)
        ratio_t, total_sample_t, nb_sample_r = get_ratio('webcam', r_min=1.0, r_max=1.0)  # , r_min=0.85, r_max=0.4)

        opt['source_loader'] = get_office31_shift('dslr_dslr', batch_size=batch_size, drop_last=True,
                                                  total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_office31_shift('dslr_webcam', batch_size=batch_size, drop_last=True,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_office31_shift('dslr_webcam', batch_size=batch_size, drop_last=False,
                                                total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        return opt, ratio_s, ratio_t

    if setting == 3:
        ratio_s, total_sample_s, nb_sample_r = get_ratio('webcam', r_min=r_min, r_max=r_max)  # , r_min=0.3, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio('amazon', r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_office31_shift('webcam_webcam', batch_size=batch_size, drop_last=True,
                                                  total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_office31_shift('webcam_amazon', batch_size=batch_size, drop_last=True,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_office31_shift('webcam_amazon', batch_size=batch_size, drop_last=False,
                                                total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        return opt, ratio_s, ratio_t

    if setting == 4:
        ratio_s, total_sample_s, nb_sample_r = get_ratio('webcam', r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio('dslr', r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_office31_shift('webcam_webcam', batch_size=batch_size, drop_last=True,
                                                  total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_office31_shift('webcam_dslr', batch_size=batch_size, drop_last=True,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_office31_shift('webcam_dslr', batch_size=batch_size, drop_last=False,
                                                total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        return opt, ratio_s, ratio_t

    if setting == 5:
        ratio_s, total_sample_s, nb_sample_r = get_ratio('dslr', r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio('amazon', r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_office31_shift('dslr_dslr', batch_size=batch_size, drop_last=True,
                                                  total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_office31_shift('dslr_amazon', batch_size=batch_size, drop_last=True,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_office31_shift('dslr_amazon', batch_size=batch_size, drop_last=False,
                                                total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        return opt, ratio_s, ratio_t

    if setting == 6:
        ratio_s, total_sample_s, nb_sample_r = get_ratio('amazon', r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio('webcam', r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_office31_shift('amazon_amazon', batch_size=batch_size, drop_last=True,
                                                  total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_office31_shift('amazon_webcam', batch_size=batch_size, drop_last=True,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_office31_shift('amazon_webcam', batch_size=batch_size, drop_last=False,
                                                total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        return opt, ratio_s, ratio_t

    # -------------------------------------------------------------------------
    #                       OfficeHome
    # -------------------------------------------------------------------------
    if setting == 10:
        source = 'Art_Art'
        target = 'Art_Clipart'
        ratio_s, total_sample_s, nb_sample_r = get_ratio_home(source, r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio_home(target, r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_officehome_shift(source, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=False,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        return opt, ratio_s, ratio_t

    if setting == 11:
        source = 'Art_Art'
        target = 'Art_Product'
        ratio_s, total_sample_s, nb_sample_r = get_ratio_home(source, r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio_home(target, r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_officehome_shift(source, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=False,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['nb_epoch_alg'] = 200
        return opt, ratio_s, ratio_t

    if setting == 12:
        source = 'Art_Art'
        target = 'Art_RealWorld'
        ratio_s, total_sample_s, nb_sample_r = get_ratio_home(source, r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio_home(target, r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_officehome_shift(source, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=False,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['start_align'] = 11
        return opt, ratio_s, ratio_t

    if setting == 13:
        source = 'Clipart_Clipart'
        target = 'Clipart_Art'
        ratio_s, total_sample_s, nb_sample_r = get_ratio_home(source, r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio_home(target, r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_officehome_shift(source, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=False,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        return opt, ratio_s, ratio_t

    if setting == 14:
        source = 'Clipart_Clipart'
        target = 'Clipart_Product'
        ratio_s, total_sample_s, nb_sample_r = get_ratio_home(source, r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio_home(target, r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_officehome_shift(source, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=False,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        return opt, ratio_s, ratio_t

    if setting == 15:
        source = 'Clipart_Clipart'
        target = 'Clipart_RealWorld'
        ratio_s, total_sample_s, nb_sample_r = get_ratio_home(source, r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio_home(target, r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_officehome_shift(source, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=False,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['start_align'] = 11
        return opt, ratio_s, ratio_t

    if setting == 16:
        source = 'Product_Product'
        target = 'Product_Art'
        ratio_s, total_sample_s, nb_sample_r = get_ratio_home(source, r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio_home(target, r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_officehome_shift(source, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=False,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        return opt, ratio_s, ratio_t

    if setting == 17:
        source = 'Product_Product'
        target = 'Product_Clipart'
        ratio_s, total_sample_s, nb_sample_r = get_ratio_home(source, r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio_home(target, r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_officehome_shift(source, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=False,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['nb_epoch_alg'] = 200
        return opt, ratio_s, ratio_t

    if setting == 18:
        source = 'Product_Product'
        target = 'Product_RealWorld'
        ratio_s, total_sample_s, nb_sample_r = get_ratio_home(source, r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio_home(target, r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_officehome_shift(source, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=False,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['start_align'] = 11
        return opt, ratio_s, ratio_t

    if setting == 19:
        source = 'RealWorld_RealWorld'
        target = 'RealWorld_Product'
        ratio_s, total_sample_s, nb_sample_r = get_ratio_home(source, r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio_home(target, r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_officehome_shift(source, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=False,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        return opt, ratio_s, ratio_t

    if setting == 20:
        source = 'RealWorld_RealWorld'
        target = 'RealWorld_Art'
        ratio_s, total_sample_s, nb_sample_r = get_ratio_home(source, r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio_home(target, r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_officehome_shift(source, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=False,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['nb_epoch_alg'] = 200
        return opt, ratio_s, ratio_t

    if setting == 21:
        source = 'RealWorld_RealWorld'
        target = 'RealWorld_Clipart'
        ratio_s, total_sample_s, nb_sample_r = get_ratio_home(source, r_min=r_min, r_max=r_max)
        ratio_t, total_sample_t, nb_sample_r = get_ratio_home(target, r_min=1.0, r_max=1.0)

        opt['source_loader'] = get_officehome_shift(source, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_s, ratio=ratio_s, logger=logger)
        opt['target_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=True,
                                                    total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['eval_loader'] = get_officehome_shift(target, batch_size=batch_size, drop_last=False,
                                                  total_sample=total_sample_t, ratio=ratio_t, logger=logger)
        opt['start_align'] = 11
        return opt, ratio_s, ratio_t