import numpy as np
import torch
import torch.utils.data as data


def get_officehome_shift(file, batch_size=32, drop_last=True, total_sample=2000, ratio=[1 / 31] * 31, logger=None):
    """Get officehome dataset loader."""
    try:
        path = '../../dann_dataset/office/'
        data_dic = np.load(path + 'officehome.npy', allow_pickle=True)
    except:
        path = '../dann_dataset/office/'
        data_dic = np.load(path + 'officehome.npy', allow_pickle=True)

    data = torch.from_numpy(data_dic.item()[file][0]).long()
    label = torch.from_numpy(data_dic.item()[file][1])

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


def get_ratio_home(data_name, cut=32, r_min=0.25, r_max=0.75):
    nb_sample = []

    art = [74, 41, 27, 39, 75, 99, 40, 33, 20, 76, 69, 25, 44, 40, 40, 23, 15, 18, 21, 45, 22, 46,
           90, 20, 46, 40, 40, 79, 46, 18, 72, 49, 51, 20, 42, 32, 18, 49, 21, 20, 19, 19, 20, 26,
           19, 18, 24, 47, 49, 15, 20, 30, 42, 41, 46, 40, 20, 46, 40, 16, 44, 43, 20, 21, 16]
    product = [79, 99, 62, 43, 44, 62, 47, 81, 81, 56, 99, 65, 96, 88, 75, 83, 67, 41, 67, 58, 71,
               99, 91, 90, 41, 67, 57, 90, 72, 99, 41, 54, 99, 56, 98, 72, 96, 41, 93, 68, 70,
               47, 60, 40, 38, 99, 43, 43, 58, 58, 99, 40, 49, 46, 99, 43, 99, 47, 76, 60, 58, 42, 45, 93, 98]
    clipart = [60, 55, 64, 98, 99, 99, 73, 46, 78, 99, 99, 40, 99, 64, 42, 41, 48, 40, 41, 50, 40, 40, 99, 99,
               61, 52, 99, 69, 40, 99, 53, 40, 99, 71, 99, 40, 76, 99, 83, 40, 51, 40, 99, 99, 41, 87, 40, 46,
               40, 67, 99, 75, 42, 42, 43, 61, 90, 60, 99, 80, 99, 39, 99, 53, 40]
    realworld = [86, 99, 64, 83, 99, 78, 80, 73, 68, 99, 96, 65, 64, 76, 73, 62, 51, 43, 81, 60, 58,
                 85, 75, 57, 36, 60, 52, 60, 72, 75, 83, 78, 66, 23, 71, 46, 60, 58, 68, 64, 30, 68,
                 65, 59, 67, 52, 53, 66, 75, 41, 77, 51, 66, 77, 88, 63, 81, 54, 53, 59, 82, 85, 67, 81, 49]

    data_tar = data_name.split('_')[1]
    if data_tar == 'Art':
        data = art
    elif data_tar == 'Product':
        data = product
    elif data_tar == 'Clipart':
        data = clipart
    elif data_tar == 'RealWorld':
        data = realworld

    for i, dat in enumerate(data):
        if i < cut:
            r = r_min
        else:
            r = r_max
        nb_sample.append((dat * r))
    nb_sample = np.array(nb_sample).astype('int')
    ratio = nb_sample / np.sum(nb_sample)
    total_sample = np.sum(nb_sample)

    return ratio, total_sample, nb_sample