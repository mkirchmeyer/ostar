"""
Modified from http://csr.bu.edu/ftp/visda17/clf/
"""

import numpy as np
import torch
import torch.utils.data as data


def get_visda_shift(train, batch_size=32, drop_last=True, total_sample=2000,
                    ratio=[0.3, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                    classe_vec=[0, 4, 11], resample=True, logger=None):
    """Get USPS dataset loader."""
    path = '../dann_dataset/visda/'
    aux = [str(i) for i in classe_vec]
    if train:
        filename = 'visda-train' + ''.join(aux) + '.npz'
    else:
        filename = 'visda-val' + ''.join(aux) + '.npz'
    logger.info(filename)
    res = np.load(path + filename)
    data = torch.from_numpy(res['X'])
    label = torch.from_numpy(res['y'])

    if resample:
        # ----------------------Subsampling the dataset ---------------------------
        c = len(torch.unique(label))
        n = label.size(0)
        ind = [[j for j in range(n) if label[j] == i] for i in range(c)]
        nb_sample = [len(ind[i]) for i in range(c)]
        logger.info(f'sample per class in data before subsampling: {nb_sample} / sum={np.sum(nb_sample)}')
        logger.info(f'ratio*total: {np.array(ratio) * total_sample} / sum={np.sum(np.array(ratio) * total_sample)}')
        all_index = torch.zeros(0).long()
        for i in range(c):
            perm = torch.randperm(nb_sample[i])
            ind_classe = label.eq(i).nonzero()
            ind = ind_classe[perm[:int(ratio[i] * total_sample)].long()]
            all_index = torch.cat((all_index, ind))

        label = label[all_index].squeeze()
        data = data[all_index].float().squeeze()
        logger.info(data.shape)
    else:
        data = data.float()

    # ------------------------------------------------------------------------

    full_data = torch.utils.data.TensorDataset(data, label.long())
    usps_data_loader = torch.utils.data.DataLoader(
        dataset=full_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last)

    return usps_data_loader
