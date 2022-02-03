import copy
import getopt
import logging
import random
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from ClassDANN import DANN
from ClassOSTAR import OSTAR
from ClassWDTS_Adv import WDTS_Adv
from compare_digits_setting import digits_expe_setting
from compare_office_setting import office_expe_setting
from compare_visda_setting import visda_expe_setting
from proportion_estimators import estimate_source_proportion
from utils_local import weight_init, set_optimizer_feat_extractor, set_optimizer_data_classifier, set_optimizer_phi, \
    set_optimizer_domain_classifier, set_proportion_method, evaluate_data_classifier, weights_init_digits, create_logger, \
    m_std, compute_diff_label, format_list, set_optimizer_data_classifier_t

logging.getLogger("matplotlib.font_manager").disabled = True

setting = 10
gpu = 0
modeltorun = '000000000001'
nb_iter = 5
dataset = "digits"
output_npz = True

opts, args = getopt.getopt(sys.argv[1:], "d:g:i:r:s:t:")
for opt, arg in opts:
    if opt == "-d":
        dataset = arg
    if opt == "-g":
        gpu = int(arg)
    if opt == "-i":
        nb_iter = int(arg)
    if opt == "-s":
        setting = int(arg)
    if opt == "-t":
        modeltorun = arg

now = datetime.now()
ts = int(time.mktime(now.timetuple()))
print(f"run_id: {ts}")
cuda = torch.cuda.is_available()
if cuda:
    gpu_id = gpu
    torch.cuda.set_device(gpu_id)
filename = f"{str(ts)}"
path_results = f'./results/{dataset}/'
logger = create_logger(path_results, f"{path_results}/{filename}")

# Dataset
is_usps_mnist = (setting >= 10) and (dataset == "digits")
is_mnist_usps = (setting < 10) and (dataset == "digits")
is_visda12 = (dataset == "visda")
is_office_31 = (dataset == "office")
is_office_home = (setting >= 10) and (dataset == "office")

if dataset == "visda":
    opt, ratio_s, ratio_t = visda_expe_setting(setting, logger=logger)
elif dataset == "digits":
    opt, ratio_s, ratio_t = digits_expe_setting(setting, logger=logger)
elif dataset == "office":
    opt, ratio_s, ratio_t = office_expe_setting(setting, logger=logger)
else:
    raise Exception("Dataset does not exist")

# Hyperparameters
proportion_method = "confusion"
beta_vec = [0, 1, 2, 3, 4]
lr_f, lr_g, lr_phi, lr_d = opt["lr"], opt["lr"], opt["lr"], opt["lr"]
lr_g_weight = lr_d_weight = lr_phi_weight = lr_f_weight = lr_f_weight_star = 1.0
ent_weight = clf_t_weight = div_weight = 0.1
n_epochs = n_epochs_star = opt['nb_epoch_alg']
epoch_start_g = opt["start_align"] + 9
iter_domain_classifier = 10
use_div = (dataset == "office" or dataset == "visda")

if is_usps_mnist:
    n_epochs = n_epochs_star = 50
    epoch_start_g = opt["start_align"] + 9
    lr_g = opt["lr"] * 100
    lr_f = opt["lr"] / 10
    lr_f_weight = 0.1
    lr_d_weight = lr_g_weight = lr_f_weight_star = 10
    ot_weight = 0.01
    clf_t_weight = ent_weight = 0.1
elif is_mnist_usps:
    n_epochs = n_epochs_star = 30
    epoch_start_g = opt["start_align"] + 9
    lr_g = opt["lr"] * 100
    lr_f = opt["lr"] / 10
    lr_d_weight = lr_g_weight = lr_f_weight = lr_f_weight_star = lr_phi_weight = 10
    ot_weight = 0.01
    clf_t_weight = ent_weight = 0.1
elif is_visda12:
    n_epochs, n_epochs_star = 150, 30
    epoch_start_g = opt["start_align"] + 5
    ot_weight = 1e-5
    clf_t_weight = ent_weight = div_weight = 1.0
    iter_domain_classifier = 20
elif is_office_31:
    n_epochs = n_epochs_star = 100
    epoch_start_g = opt["start_align"] + 9
    lr_d_weight = 10
    ot_weight = 1e-5
    clf_t_weight = ent_weight = div_weight = 1.0
    iter_domain_classifier = 20
elif is_office_home:
    n_epochs = n_epochs_star = 100
    epoch_start_g = opt["start_align"] + 5
    ot_weight = 1e-5
    clf_t_weight = ent_weight = div_weight = 1.0
    iter_domain_classifier = 20

logger.info(f"run_id: {ts}")
logger.info(f"dataset: {dataset}")
logger.info(f"setting: {setting}")
logger.info(f"model to run: {modeltorun}")
logger.info(f"proportion_method: {proportion_method}")
logger.info(f"cluster_step: {opt['cluster_every']}")
logger.info(f"lr_f_weight_baseline: {lr_f_weight}")
logger.info(f"lr_f_weight_star: {lr_f_weight_star}")
logger.info(f"lr_g_weight: {lr_g_weight}")
logger.info(f"lr_d_weight: {lr_d_weight}")
logger.info(f"lr_phi_weight: {lr_phi_weight}")
logger.info(f"clf_t_weight: {clf_t_weight}")
logger.info(f"div_weight: {div_weight}")
logger.info(f"entropy_weight: {ent_weight}")
logger.info(f"ot_weight: {ot_weight}")
logger.info(f"epoch_start_g: {epoch_start_g}")
logger.info(f"use_div: {use_div}")
logger.info(f"output_npz: {output_npz}")
if cuda:
    logger.info(f"gpu_id: {gpu_id}")

# Init acc
bc_source_t, MAP_source_t = np.zeros(nb_iter), np.zeros(nb_iter)
bc_dann_t, MAP_dann_t = np.zeros(nb_iter), np.zeros(nb_iter)
bc_wd_t, MAP_wd_t = np.zeros((nb_iter, len(beta_vec))), np.zeros((nb_iter, len(beta_vec)))
bc_marsg_t, MAP_marsg_t = np.zeros(nb_iter), np.zeros(nb_iter)
bc_wdgt_t, MAP_wdgt_t = np.zeros(nb_iter), np.zeros(nb_iter)
bc_marsc_t, MAP_marsc_t = np.zeros(nb_iter), np.zeros(nb_iter)
bc_iwwd_t, MAP_iwwd_t = np.zeros(nb_iter), np.zeros(nb_iter)
bc_ostar_t, MAP_ostar_t = np.zeros(nb_iter), np.zeros(nb_iter)
hist_prop_gmm, hist_prop_conf, hist_prop_clus, hist_prop_gt, hist_prop_ot = np.empty(opt['nb_class']), \
    np.empty(opt['nb_class']), np.empty(opt['nb_class']), np.empty(opt['nb_class']), np.empty(opt['nb_class'])
mse_label_t_marsc, js_label_t_marsc = np.zeros(nb_iter), np.zeros(nb_iter)
mse_label_t_marsg, js_label_t_marsg = np.zeros(nb_iter), np.zeros(nb_iter)
mse_label_t_iwwd, js_label_t_iwwd = np.zeros(nb_iter), np.zeros(nb_iter)
mse_label_t_ostar, js_label_t_ostar = np.zeros(nb_iter), np.zeros(nb_iter)

for it in range(nb_iter):
    logger.info(f"====Iter {it}====")
    np.random.seed(it)
    random.seed(it)
    torch.manual_seed(it)
    torch.cuda.manual_seed_all(it)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    source_loader = opt['source_loader']
    target_loader = opt['target_loader']
    eval_loader = opt['eval_loader']

    feat_extract_init = opt['feat_extract']
    data_class_init = opt['data_classifier']
    data_class_t_init = opt['data_classifier_t']
    domain_class_init = opt['domain_classifier']
    domain_class_dann_init = opt['domain_classifier_dann']
    phi_init = opt["phi"]

    if (is_usps_mnist or is_mnist_usps):
        weights_init_digits(feat_extract_init, name="normal")
        weights_init_digits(data_class_init, name="normal")
        weights_init_digits(data_class_t_init, name="normal")
        weights_init_digits(domain_class_init, name="normal")
        weights_init_digits(domain_class_dann_init, name="normal")
    else:
        feat_extract_init.apply(weight_init)
        data_class_init.apply(weight_init)
        data_class_t_init.apply(weight_init)
        domain_class_init.apply(weight_init)
        domain_class_dann_init.apply(weight_init)
    weights_init_digits(phi_init, name="orthogonal")

    proportion_S = estimate_source_proportion(source_loader, opt['nb_class'])
    logger.info(f"pS(Y): {format_list(proportion_S, 4)}")
    proportion_S = torch.from_numpy(ratio_s).float()
    logger.info(f"ratio_s: {format_list(proportion_S, 4)}")

    proportion_T_gt = estimate_source_proportion(eval_loader, opt['nb_class'])
    logger.info(f"pT(Y) gt: {format_list(proportion_T_gt, 4)}")
    proportion_T_gt = torch.from_numpy(ratio_t).float()
    logger.info(f"ratio_t: {format_list(proportion_T_gt, 4)}")

    # ------------------------------------------------------------------------
    # Source
    # ------------------------------------------------------------------------
    if int(modeltorun[0]) == 1:
        logger.info("\n---Source---")
        feat_extract_source = copy.deepcopy(feat_extract_init)
        data_class_source = copy.deepcopy(data_class_init)
        domain_class_source = copy.deepcopy(domain_class_dann_init)

        source = DANN(feat_extract_source, data_class_source, domain_class_source, source_loader, target_loader,
                      cuda=cuda, logger=logger, n_epochs=n_epochs, grad_scale=opt['grad_scale_dann'],
                      epoch_start_align=opt['nb_epoch_alg'], n_class=opt['nb_class'], eval_data_loader=eval_loader,
                      proportion_T_gt=proportion_T_gt, setting=setting, dataset=dataset, ts=ts, iter=it)
        set_optimizer_feat_extractor(source, optim.Adam(source.feat_extractor.parameters(), lr=opt['lr_dann'], betas=(0.5, 0.999)))
        set_optimizer_data_classifier(source, optim.Adam(source.data_classifier.parameters(), lr=opt['lr_dann'], betas=(0.5, 0.999)))
        set_optimizer_domain_classifier(source, optim.Adam(source.grl_domain_classifier.parameters(), lr=opt['lr_dann'], betas=(0.5, 0.999)))
        source.fit()

        bc_source_t[it], MAP_source_t[it] = evaluate_data_classifier(source, eval_loader, is_target=True)

    # ------------------------------------------------------------------------
    # DANN
    # ------------------------------------------------------------------------
    if int(modeltorun[1]) == 1:
        logger.info("\n---DANN---")
        feat_extract_dann = copy.deepcopy(feat_extract_init)
        data_class_dann = copy.deepcopy(data_class_init)
        domain_class_dann = copy.deepcopy(domain_class_dann_init)

        dann = DANN(feat_extract_dann, data_class_dann, domain_class_dann, source_loader, target_loader,
                    cuda=cuda, logger=logger, n_epochs=n_epochs, grad_scale=opt['grad_scale_dann'], init_lr=opt['lr_dann'],
                    epoch_start_align=opt['start_align'], n_class=opt['nb_class'], eval_data_loader=eval_loader, iter=it,
                    lr_g_weight=lr_g_weight, lr_f_weight=lr_f_weight, lr_d_weight=lr_d_weight, setting=setting, dataset=dataset, ts=ts,
                    proportion_T_gt=proportion_T_gt)
        set_optimizer_feat_extractor(dann, optim.Adam(dann.feat_extractor.parameters(), lr=opt['lr_dann'], betas=(0.5, 0.999)))
        set_optimizer_data_classifier(dann, optim.Adam(dann.data_classifier.parameters(), lr=opt['lr_dann'], betas=(0.5, 0.999)))
        set_optimizer_domain_classifier(dann, optim.Adam(dann.grl_domain_classifier.parameters(), lr=opt['lr_dann'], betas=(0.5, 0.999)))
        dann.fit()

        bc_dann_t[it], MAP_dann_t[it] = evaluate_data_classifier(dann, eval_loader, is_target=True)

    # ------------------------------------------------------------------------
    # Weighted adversarial wasserstein WD_beta - Wu2019
    # ------------------------------------------------------------------------
    list_run = [int(i) for i in modeltorun[2:6]]
    if sum(list_run) > 0:
        for k, todo in enumerate(list_run):
            if todo == 1:
                beta = beta_vec[k]
                logger.info(f"\n---WD beta:{beta}---")
                if beta == 0:
                    grad_scale_wd = opt['grad_scale_beta_0']
                    lr = opt['lr_beta_0']
                if beta == 2 or beta == 1:
                    grad_scale_wd = opt['grad_scale_beta_1']
                    lr = opt['lr_beta_1']
                else:
                    grad_scale_wd = opt['grad_scale_beta_1'] / opt['wdgrl_grad_down_factor']
                    lr = opt['lr_beta_1']
                feat_extract_wd = copy.deepcopy(feat_extract_init)
                data_class_wd = copy.deepcopy(data_class_init)
                domain_class_wd = copy.deepcopy(domain_class_init)

                wdgrl = WDTS_Adv(feat_extract_wd, data_class_wd, domain_class_wd, source_loader, target_loader, init_lr=lr,
                                 cuda=cuda, grad_scale=grad_scale_wd, logger=logger, n_epochs=n_epochs, beta_ratio=beta,
                                 eval_data_loader=eval_loader, epoch_start_align=opt['start_align'], lr_d_weight=lr_d_weight,
                                 iter_domain_classifier=opt['iter_domain'], lr_g_weight=lr_g_weight, lr_f_weight=lr_f_weight, ts=ts,
                                 setting=setting, dataset=dataset, iter=it, n_class=opt['nb_class'],
                                 proportion_T_gt=proportion_T_gt)
                set_optimizer_feat_extractor(wdgrl, optim.Adam(wdgrl.feat_extractor.parameters(), lr=lr_g, betas=(0.5, 0.999)))
                set_optimizer_data_classifier(wdgrl, optim.Adam(wdgrl.data_classifier.parameters(), lr=lr_f, betas=(0.5, 0.999)))
                set_optimizer_domain_classifier(wdgrl, optim.Adam(wdgrl.domain_classifier.parameters(), lr=lr_d, betas=(0.5, 0.999)))
                wdgrl.fit()

                bc_wd_t[it, k], MAP_wd_t[it, k] = evaluate_data_classifier(wdgrl, eval_loader, is_target=True)

    # ------------------------------------------------------------------------
    # Wasserstein Distance with Reweighting methods
    # ------------------------------------------------------------------------
    if int(modeltorun[7]) == 1:
        logger.info("\n---MARSg---")
        feat_extract_wdtsadv = copy.deepcopy(feat_extract_init)
        data_class_wdtsadv = copy.deepcopy(data_class_init)
        domain_class_wdtsadv = copy.deepcopy(domain_class_init)

        marsg = WDTS_Adv(feat_extract_wdtsadv, data_class_wdtsadv, domain_class_wdtsadv, source_loader, target_loader,
                         cuda=cuda, grad_scale=opt['grad_scale'], logger=logger, ts=ts, dataset=dataset,
                         n_class=opt['nb_class'], compute_cluster_every=opt['cluster_every'], cluster_param=opt['cluster_param'],
                         n_epochs=n_epochs, gamma=opt['gamma_wdts'], init_lr=opt["lr"], epoch_start_align=opt['start_align'],
                         iter_domain_classifier=opt['iter_domain'], eval_data_loader=eval_loader,
                         proportion_T_gt=proportion_T_gt, lr_g_weight=lr_g_weight,  lr_f_weight=lr_f_weight,
                         lr_d_weight=lr_d_weight, setting=setting, epoch_start_g=epoch_start_g, proportion_S=proportion_S, iter=it)
        set_proportion_method(marsg, 'gmm')
        set_optimizer_feat_extractor(marsg, optim.Adam(marsg.feat_extractor.parameters(), lr=lr_g, betas=(0.5, 0.999)))
        set_optimizer_data_classifier(marsg, optim.Adam(marsg.data_classifier.parameters(), lr=lr_f, betas=(0.5, 0.999)))
        set_optimizer_domain_classifier(marsg, optim.Adam(marsg.domain_classifier.parameters(), lr=lr_d, betas=(0.5, 0.999)))
        marsg.fit()

        hist_prop_gmm = np.vstack((hist_prop_gmm, marsg.hist_proportion[-1, :]))
        bc_marsg_t[it], MAP_marsg_t[it] = evaluate_data_classifier(marsg, eval_loader, is_target=True)
        mse_label_t_marsg[it], js_label_t_marsg[it] = compute_diff_label(marsg, proportion_T_gt)

    if int(modeltorun[8]) == 1:
        logger.info("\n---MARSc---")
        feat_extract_wdtsadv = copy.deepcopy(feat_extract_init)
        data_class_wdtsadv = copy.deepcopy(data_class_init)
        domain_class_wdtsadv = copy.deepcopy(domain_class_init)

        marsc = WDTS_Adv(feat_extract_wdtsadv, data_class_wdtsadv, domain_class_wdtsadv, source_loader,
                         target_loader, cuda=cuda, grad_scale=opt['grad_scale'], logger=logger, ts=ts, dataset=dataset,
                         n_class=opt['nb_class'], compute_cluster_every=opt['cluster_every'], cluster_param=opt['cluster_param'],
                         n_epochs=n_epochs, gamma=opt['gamma_wdts'], init_lr=opt["lr"], epoch_start_align=opt['start_align'],
                         iter_domain_classifier=opt['iter_domain'], eval_data_loader=eval_loader,
                         iter=it, proportion_T_gt=proportion_T_gt, setting=setting, lr_g_weight=lr_g_weight,
                         lr_f_weight=lr_f_weight, lr_d_weight=lr_d_weight, epoch_start_g=epoch_start_g, proportion_S=proportion_S)
        set_proportion_method(marsc, 'cluster')
        set_optimizer_feat_extractor(marsc, optim.Adam(marsc.feat_extractor.parameters(), lr=lr_g, betas=(0.5, 0.999)))
        set_optimizer_data_classifier(marsc, optim.Adam(marsc.data_classifier.parameters(), lr=lr_f, betas=(0.5, 0.999)))
        set_optimizer_domain_classifier(marsc, optim.Adam(marsc.domain_classifier.parameters(), lr=lr_d, betas=(0.5, 0.999)))

        marsc.fit()
        hist_prop_clus = np.vstack((hist_prop_clus, marsc.hist_proportion[-1, :]))
        bc_marsc_t[it], MAP_marsc_t[it] = evaluate_data_classifier(marsc, eval_loader, is_target=True)
        mse_label_t_marsc[it], js_label_t_marsc[it] = compute_diff_label(marsc, proportion_T_gt)

    if int(modeltorun[9]) == 1:
        logger.info("\n---IW-WD---")
        feat_extract_wdtsadv = copy.deepcopy(feat_extract_init)
        data_class_wdtsadv = copy.deepcopy(data_class_init)
        domain_class_wdtsadv = copy.deepcopy(domain_class_init)

        iwwd = WDTS_Adv(feat_extract_wdtsadv, data_class_wdtsadv, domain_class_wdtsadv, source_loader,
                        target_loader, cuda=cuda, grad_scale=opt['grad_scale'], logger=logger, ts=ts, dataset=dataset,
                        n_class=opt['nb_class'], compute_cluster_every=opt['cluster_every'],
                        cluster_param=opt['cluster_param'], n_epochs=n_epochs, gamma=opt['gamma_wdts'],
                        epoch_start_align=opt['start_align'], iter_domain_classifier=opt['iter_domain'], init_lr=opt["lr"],
                        eval_data_loader=eval_loader, proportion_T_gt=proportion_T_gt, lr_g_weight=lr_g_weight,
                        lr_f_weight=lr_f_weight, lr_d_weight=lr_d_weight, setting=setting,
                        epoch_start_g=epoch_start_g, proportion_S=proportion_S, iter=it)
        set_proportion_method(iwwd, 'confusion')
        set_optimizer_feat_extractor(iwwd, optim.Adam(iwwd.feat_extractor.parameters(), lr=lr_g, betas=(0.5, 0.999)))
        set_optimizer_data_classifier(iwwd, optim.Adam(iwwd.data_classifier.parameters(), lr=lr_f, betas=(0.5, 0.999)))
        set_optimizer_domain_classifier(iwwd, optim.Adam(iwwd.domain_classifier.parameters(), lr=lr_d, betas=(0.5, 0.999)))
        iwwd.fit()

        hist_prop_conf = np.vstack((hist_prop_conf, iwwd.hist_proportion[-1, :]))
        bc_iwwd_t[it], MAP_iwwd_t[it] = evaluate_data_classifier(iwwd, eval_loader, is_target=True)
        mse_label_t_iwwd[it], js_label_t_iwwd[it] = compute_diff_label(iwwd, proportion_T_gt)

    if int(modeltorun[10]) == 1:
        logger.info("\n---WD-GT---")
        feat_extract_wdtsadv = copy.deepcopy(feat_extract_init)
        data_class_wdtsadv = copy.deepcopy(data_class_init)
        domain_class_wdtsadv = copy.deepcopy(domain_class_init)

        wd_gt = WDTS_Adv(feat_extract_wdtsadv, data_class_wdtsadv, domain_class_wdtsadv, source_loader,
                           target_loader, cuda=cuda, grad_scale=opt['grad_scale'], logger=logger, ts=ts,
                           dataset=dataset, eval_data_loader=eval_loader, n_class=opt['nb_class'], n_epochs=n_epochs, iter=it,
                           epoch_start_align=opt['start_align'], compute_cluster_every=opt['cluster_every'],
                           gamma=opt['gamma_wdts'], iter_domain_classifier=opt['iter_domain'], init_lr=opt["lr"],
                           cluster_param=opt['cluster_param'], proportion_T_gt=proportion_T_gt, proportion_S=proportion_S,
                           lr_g_weight=lr_g_weight, lr_f_weight=lr_f_weight, lr_d_weight=lr_d_weight, setting=setting)
        set_proportion_method(wd_gt, 'gt')
        set_optimizer_feat_extractor(wd_gt, optim.Adam(wd_gt.feat_extractor.parameters(), lr=lr_g, betas=(0.5, 0.999)))
        set_optimizer_data_classifier(wd_gt, optim.Adam(wd_gt.data_classifier.parameters(), lr=lr_f, betas=(0.5, 0.999)))
        set_optimizer_domain_classifier(wd_gt, optim.Adam(wd_gt.domain_classifier.parameters(), lr=lr_d, betas=(0.5, 0.999)))
        wd_gt.fit()

        hist_prop_gt = np.vstack((hist_prop_gt, wd_gt.hist_proportion[-1, :]))
        bc_wdgt_t[it], MAP_wdgt_t[it] = evaluate_data_classifier(wd_gt, eval_loader, is_target=True)

    if int(modeltorun[11]) == 1:
        logger.info("\n---OSTAR---")
        feat_extract_ostar = copy.deepcopy(feat_extract_init)
        data_class_ostar = copy.deepcopy(data_class_init)
        data_class_t_ostar = copy.deepcopy(data_class_t_init)
        phi_ostar = copy.deepcopy(phi_init)
        domain_class_ostar = copy.deepcopy(domain_class_init)

        ostar = OSTAR(feat_extract_ostar, data_class_ostar, phi_ostar, source_loader, target_loader, cuda=cuda,
                      logger=logger, ot_weight=ot_weight, n_class=opt["nb_class"], ts=ts, dataset=dataset,
                      cluster_step=opt['cluster_every'], eval_data_loader=eval_loader, domain_classifier=domain_class_ostar,
                      epoch_start_align=opt["start_align"], init_lr=opt["lr"], cluster_param=opt['cluster_param'],
                      lr_phi_weight=lr_phi_weight, use_div=use_div, lr_g_weight=lr_g_weight,
                      lr_f_weight=lr_f_weight_star, lr_d_weight=lr_d_weight, n_epochs=n_epochs_star,
                      clf_t_weight=clf_t_weight, iter=it, epoch_start_g=epoch_start_g, gamma=opt['gamma_wdts'],
                      iter_domain_classifier=iter_domain_classifier, proportion_T_gt=proportion_T_gt, div_weight=div_weight,
                      data_class_t=data_class_t_ostar, ent_weight=ent_weight, setting=setting, proportion_S=proportion_S)
        set_optimizer_phi(ostar, optim.Adam(ostar.phi.parameters(), lr=lr_phi, betas=(0.5, 0.999)))
        set_optimizer_data_classifier_t(ostar, optim.Adam(ostar.data_classifier_t.parameters(), lr=lr_f, betas=(0.5, 0.999)))
        set_optimizer_feat_extractor(ostar, optim.Adam(ostar.feat_extractor.parameters(), lr=lr_g, betas=(0.5, 0.999)))
        set_optimizer_data_classifier(ostar, optim.Adam(ostar.data_classifier.parameters(), lr=lr_f, betas=(0.5, 0.999)))
        set_optimizer_domain_classifier(ostar, optim.Adam(ostar.domain_classifier.parameters(), lr=lr_d, betas=(0.5, 0.999)))
        ostar.fit()

        evaluate_data_classifier(ostar, source_loader, is_target=False, verbose=True)
        bc_ostar_t[it], MAP_ostar_t[it] = evaluate_data_classifier(ostar, eval_loader, is_target=True, verbose=True, is_ft=True)
        hist_prop_ot = np.vstack((hist_prop_ot, ostar.hist_proportion[-1, :]))
        mse_label_t_ostar[it], js_label_t_ostar[it] = compute_diff_label(ostar, proportion_T_gt)

    M_bc_marsc_t, S_bc_marsc_t = m_std(bc_marsc_t, it)
    M_bc_marsg_t, S_bc_marsg_t = m_std(bc_marsg_t, it)
    M_bc_iwwd_t, S_bc_iwwd_t = m_std(bc_iwwd_t, it)
    M_bc_wdgt_t, S_bc_wdgt_t = m_std(bc_wdgt_t, it)
    M_bc_wd_t, S_bc_wd_t = m_std(bc_wd_t, it, is_list=True)
    M_bc_dann_t, S_bc_dann_t = m_std(bc_dann_t, it)
    M_bc_source_t, S_bc_source_t = m_std(bc_source_t, it)
    M_bc_ostar_t, S_bc_ostar_t = m_std(bc_ostar_t, it)

    M_MAP_marsc_t, S_MAP_marsc_t = m_std(MAP_marsc_t, it)
    M_MAP_marsg_t, S_MAP_marsg_t = m_std(MAP_marsg_t, it)
    M_MAP_iwwd_t, S_MAP_iwwd_t = m_std(MAP_iwwd_t, it)
    M_MAP_wdgt_t, S_MAP_wdgt_t = m_std(MAP_wdgt_t, it)
    M_MAP_wd_t, S_MAP_wd_t = m_std(MAP_wd_t, it, is_list=True)
    M_MAP_dann_t, S_MAP_dann_t = m_std(MAP_dann_t, it)
    M_MAP_source_t, S_MAP_source_t = m_std(MAP_source_t, it)
    M_MAP_ostar_t, S_MAP_ostar_t = m_std(MAP_ostar_t, it)

    M_mse_label_t_marsg, S_mse_label_t_marsg = m_std(mse_label_t_marsg, it)
    M_js_label_t_marsg, S_js_label_t_marsg = m_std(js_label_t_marsg, it)
    M_mse_label_t_marsc, S_mse_label_t_marsc = m_std(mse_label_t_marsc, it)
    M_js_label_t_marsc, S_js_label_t_marsc = m_std(js_label_t_marsc, it)
    M_mse_label_t_iwwd, S_mse_label_t_iwwd = m_std(mse_label_t_iwwd, it)
    M_js_label_t_iwwd, S_js_label_t_iwwd = m_std(js_label_t_iwwd, it)
    M_mse_label_t_ostar, S_mse_label_t_ostar = m_std(mse_label_t_ostar, it)
    M_js_label_t_ostar, S_js_label_t_ostar = m_std(js_label_t_ostar, it)

    WD_bc = [str(float("%.2f" % elt[0])) + " +- " + str(float("%.1f" % elt[1])) for elt in
             list(zip(M_bc_wd_t, S_bc_wd_t))]
    WD_MAP = [str(float("%.2f" % elt[0])) + " +- " + str(float("%.1f" % elt[1])) for elt in
              list(zip(M_MAP_wd_t, S_MAP_wd_t))]
    logger.info(
        f"Target Test BC {it + 1}/{nb_iter}= Source: {M_bc_source_t:.2f} +- {S_bc_source_t:.1f} / "
        f"DANN: {M_bc_dann_t:.2f} +- {S_bc_dann_t:.1f} / "
        f"WD: {WD_bc} / "
        f"MARSg: {M_bc_marsg_t:.2f} +- {S_bc_marsg_t:.1f} / "
        f"MARSc: {M_bc_marsc_t:.2f} +- {S_bc_marsc_t:.1f} / "
        f"IW-WD: {M_bc_iwwd_t:.2f} +- {S_bc_iwwd_t:.1f} / "
        f"OSTAR: {M_bc_ostar_t:.2f} +- {S_bc_ostar_t:.1f} / "
        f"WD-GT: {M_bc_wdgt_t:.2f} +- {S_bc_wdgt_t:.1f}")
    logger.info(
        f"Target Test MAP {it + 1}/{nb_iter} = Source: {M_MAP_source_t:.2f} +- {S_MAP_source_t:.1f} / "
        f"DANN: {M_MAP_dann_t:.2f} +- {S_MAP_dann_t:.1f} / "
        f"WD: {WD_MAP} / "
        f"MARSg: {M_MAP_marsg_t:.2f} +- {S_MAP_marsg_t:.1f} /  "
        f"MARSc: {M_MAP_marsc_t:.2f} +- {S_MAP_marsc_t:.1f} / "
        f"IW-WD: {M_MAP_iwwd_t:.2f} +- {S_MAP_iwwd_t:.1f}/ "        
        f"OSTAR: {M_MAP_ostar_t:.2f} +- {S_MAP_ostar_t:.1f} / "
        f"WD-GT: {M_MAP_wdgt_t:.2f} +- {S_MAP_wdgt_t:.1f}")
    logger.info(
        f"{M_MAP_source_t:.2f} +- {S_MAP_source_t:.1f} / {M_MAP_dann_t:.2f} +- {S_MAP_dann_t:.1f} / {WD_MAP} / "
        f"{M_MAP_marsg_t:.2f} +- {S_MAP_marsg_t:.1f} / {M_MAP_marsc_t:.2f} +- {S_MAP_marsc_t:.1f} / "
        f"{M_MAP_iwwd_t:.2f} +- {S_MAP_iwwd_t:.1f}/ {M_MAP_ostar_t:.2f} +- {S_MAP_ostar_t:.1f} / "
        f"{M_MAP_wdgt_t:.2f} +- {S_MAP_wdgt_t:.1f}")
    logger.info(f"MSE_marsg: {M_mse_label_t_marsg:.4f} +- {S_mse_label_t_marsg:.4f} / "
                f"JS_marsg: {M_js_label_t_marsg:.4f} +- {S_js_label_t_marsg:.4f} -- "
                f"MSE_marsc: {M_mse_label_t_marsc:.4f} +- {S_mse_label_t_marsc:.4f} / "
                f"JS_marsc: {M_js_label_t_marsc:.4f} +- {S_js_label_t_marsc:.4f} -- "
                f"MSE_iwwd: {M_mse_label_t_iwwd:.4f} +- {S_mse_label_t_iwwd:.4f} / "
                f"JS_iwwd: {M_js_label_t_iwwd:.4f} +- {S_js_label_t_iwwd:.4f} -- "
                f"MSE_ostar: {M_mse_label_t_ostar:.4f} +- {S_mse_label_t_ostar:.4f} / "
                f"JS_ostar: {M_js_label_t_ostar:.4f} +- {S_js_label_t_ostar:.4f}")

    if output_npz:
        np.savez(path_results + filename, MAP_source=MAP_source_t, MAP_dann=MAP_dann_t, MAP_wd=MAP_wd_t,
                 MAP_marsg=MAP_marsg_t, MAP_marsc=MAP_marsc_t, MAP_iwwd=MAP_iwwd_t, MAP_wdgt=MAP_wdgt_t,
                 MAP_star=MAP_ostar_t, hist_prop_gmm=hist_prop_gmm, hist_prop_clus=hist_prop_clus,
                 hist_prop_conf=hist_prop_conf, hist_prop_gt=hist_prop_gt, hist_prop_ot=hist_prop_ot)
