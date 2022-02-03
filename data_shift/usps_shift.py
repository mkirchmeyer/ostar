"""
Dataset setting and data loader for USPS.
Modified from https://github.com/corenel/pytorch-adda/
"""

import gzip
import os
import pickle
import urllib
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms


class USPS(data.Dataset):
    """
    USPS Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC
        print(self.train_labels.shape)
        self.train_labels = self.train_labels

    def __getitem__(self, index):
        """
        Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = int(label)
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels


def get_usps_shift(train, batch_size=32, drop_last=True, total_sample=2000, image_size=28, num_channel=1,
                   ratio=[0.3, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], logger=None):
    """Get USPS dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.5],
                                          std=[0.5])])
    # dataset and data loader
    usps_dataset = USPS(root='../dann_dataset',
                        train=train,
                        transform=pre_process,
                        download=True)
    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False)
    data = torch.zeros((len(usps_data_loader), num_channel, image_size, image_size))
    label = torch.zeros(len(usps_data_loader))
    # preloading te full dataset
    for i, (data_, target) in enumerate(usps_data_loader):
        data[i] = data_[0, 0]
        label[i] = target
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
    data = data[all_index, 0]
    # ------------------------------------------------------------------------

    full_data = torch.utils.data.TensorDataset(data, label.long())
    usps_data_loader = torch.utils.data.DataLoader(
        dataset=full_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last)

    return usps_data_loader
