import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


def get_office31_shift(file, batch_size=32, drop_last=True, total_sample=2000, ratio=[1 / 31] * 31, logger=None):
    """Get USPS dataset loader."""
    try:
        path = '../../dann_dataset/office/office31_resnet50/'
        df = pd.read_csv(path + file + '.csv')
    except:
        path = '../dann_dataset/office/office31_resnet50/'
        df = pd.read_csv(path + file + '.csv')

    data = torch.from_numpy(df.values[:, 0:2048]).long()
    label = torch.from_numpy(df.values[:, 2048])

    # ----------------------Subsampling the dataset ---------------------------
    c = len(torch.unique(label))
    n = label.size(0)
    ind = [[j for j in range(n) if label[j] == i] for i in range(c)]
    nb_sample = [len(ind[i]) for i in range(c)]
    logger.info(f'sample per class in data before subsampling: {nb_sample} / sum={np.sum(nb_sample)}')
    logger.info(f'ratio*total: {np.array(ratio) * total_sample}')
    all_index = torch.zeros(0).long()
    for i in range(c):
        perm = torch.randperm(nb_sample[i])
        ind_classe = label.eq(i).nonzero()
        ind = ind_classe[perm[:int(ratio[i] * total_sample)].long()]
        all_index = torch.cat((all_index, ind))

    label = label[all_index].squeeze()
    data = data[all_index].float().squeeze()
    logger.info(data.shape)

    # ------------------------------------------------------------------------

    full_data = torch.utils.data.TensorDataset(data, label.long())
    usps_data_loader = torch.utils.data.DataLoader(
        dataset=full_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last)

    return usps_data_loader


def get_ratio(data_name, cut=15, r_min=0.25, r_max=0.75, subsample=True):
    nb_sample = []
    amazon = [92, 82, 72, 82, 36, 94, 91, 97, 97, 81, 99, 100, 100, 98, 100, 99, 100, 94, 96, 95, 93, 100, 98, 97, 90,
              75, 100, 99, 99, 96, 64]  # 2816
    dslr = [12, 21, 24, 12, 16, 12, 13, 14, 15, 15, 13, 10, 24, 16, 31, 22, 12, 8, 10, 10, 13, 15, 22, 18, 10, 7, 18,
            26, 21, 22, 15]  # 497
    webcam = [29, 21, 28, 12, 16, 31, 40, 18, 21, 19, 27, 27, 30, 19, 30, 43, 30, 27, 28, 32, 16, 20, 29, 27, 40, 11,
              25, 30, 24, 23, 21]  # 794
    visda_train = [14309, 7365, 16639, 12800, 9512, 14240, 17360, 12160, 10731, 11680, 16000, 9600]  # 152396
    visda_val = [3646, 3475, 4690, 10401, 4690, 2075, 5796, 4000, 4549, 2281, 4236, 5548]  # 55387

    if data_name is 'amazon':
        data = amazon
    elif data_name is 'dslr':
        data = dslr
    elif data_name is 'webcam':
        data = webcam
    elif data_name is 'visda_train':
        data = visda_train
    elif data_name is 'visda_val':
        data = visda_val

    if subsample:
        for i, dat in enumerate(data):
            if i < cut:
                r = r_min
            else:
                r = r_max
            nb_sample.append((dat * r))
        nb_sample = np.array(nb_sample).astype('int')
        ratio = nb_sample / np.sum(nb_sample)
        total_sample = np.sum(nb_sample)
    else:
        ratio = data / np.sum(data)
        total_sample = np.sum(data)
        nb_sample = data

    return ratio, total_sample, nb_sample
