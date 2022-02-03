"""
Dataset setting and data loader for MNIST.
Adapted from https://github.com/corenel/pytorch-adda/tree/master/datasets
"""

import torch
from torchvision import datasets, transforms
import numpy as np


def get_mnist_shift(train, batch_size=32, drop_last=True, num_channel=3, image_size=28,
                    total_sample=5000, ratio=[0.3, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], logger=None):
    """Get MNIST dataset loader."""
    # image pre-processing   
    pre_process = transforms.Compose([transforms.Resize(image_size),
                                      transforms.Grayscale(num_channel),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.5] * num_channel,
                                          std=[0.5] * num_channel)])
    # dataset and data loader
    mnist_dataset = datasets.MNIST(root='../dann_dataset/',
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False)

    data = torch.zeros((len(mnist_data_loader), num_channel, image_size, image_size))
    label = torch.zeros(len(mnist_data_loader))
    for i, (data_, target) in enumerate(mnist_data_loader):
        data[i] = data_
        label[i] = target

    # ----------------------Subsampling the dataset ---------------------------
    c = len(torch.unique(label))
    n = label.size(0)
    ind = [[j for j in range(n) if label[j] == i] for i in range(c)]
    nb_sample_class = [len(ind[i]) for i in range(c)]
    logger.info(f'sample per class in data before subsampling: {nb_sample_class} / sum={np.sum(nb_sample_class)}')
    logger.info(f'ratio*total: {np.array(ratio) * total_sample} / sum={np.sum(np.array(ratio) * total_sample)}')
    all_index = torch.zeros(0).long()
    for i in range(c):
        perm = torch.randperm(nb_sample_class[i])
        ind_classe = label.eq(i).nonzero()
        ind = ind_classe[perm[:int(ratio[i] * total_sample)].long()]
        all_index = torch.cat((all_index, ind))

    label = label[all_index].squeeze()
    data = data[all_index][:, 0, :, :, :]

    full_data = torch.utils.data.TensorDataset(data, label.long())
    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=full_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last)

    return mnist_data_loader
