"""ReBias
Copyright (c) 2024-present Damian Smith
Copyright (c) 2020-2024 NAVER Corp.
MIT license

Python implementation of Biased-MNIST.
Modified to add background only and mixed datasets

Original at https://github.com/clovaai/rebias/blob/master/datasets/colour_mnist.py
Forked to https://github.com/DHLSmith/clovaai-rebias/blob/master/datasets/colour_mnist.py

Cite Original As:
@inproceedings{bahng2019rebias,
    title={Learning De-biased Representations with Biased Representations},
    author={Bahng, Hyojin and Chun, Sanghyuk and Yun, Sangdoo and Choo, Jaegul and Oh, Seong Joon},
    year={2020},
    booktitle={International Conference on Machine Learning (ICML)},
}
"""
import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data

from torchvision import transforms
from torchvision.datasets import MNIST


class BiasedMNIST(MNIST):
    """A base class for Biased-MNIST.
    We manually select ten colours to synthetic colour bias. (See `COLOUR_MAP` for the colour configuration)
    Usage is exactly same as torchvision MNIST dataset class.

    You have two paramters to control the level of bias.

    Parameters
    ----------
    root : str
        path to MNIST dataset.
    data_label_correlation : float, default=1.0
        Here, each class has the pre-defined colour (bias).
        data_label_correlation, or `rho` controls the level of the dataset bias.

        A sample is coloured with
            - the pre-defined colour with probability `rho`,
            - coloured with one of the other colours with probability `1 - rho`.
              The number of ``other colours'' is controlled by `n_confusing_labels` (default: 9).
        Note that the colour is injected into the background of the image (see `_binary_to_colour`).
            - The background colour can optionally be varied by setting bg_noise_level. 
              Each image will still have a flat background colour, but the values will vary slightly from 
              those in the COLOUR_MAP

        Hence, we have
            - Perfectly biased dataset with rho=1.0
            - Perfectly unbiased with rho=0.1 (1/10) ==> our ``unbiased'' setting in the test time.
        In the paper, we explore the high correlations but with small hints, e.g., rho=0.999.

    n_confusing_labels : int, default=9
        In the real-world cases, biases are not equally distributed, but highly unbalanced.
        We mimic the unbalanced biases by changing the number of confusing colours for each class.
        In the paper, we use n_confusing_labels=9, i.e., during training, the model can observe
        all colours for each class. However, you can make the problem harder by setting smaller n_confusing_labels, e.g., 2.
        We suggest to researchers considering this benchmark for future researches.

    bg_noise_level: float, default=0.0 
        Optionally vary the bg colour from the values defined in the colour_map. This applies a flat variation; ie each
        image will have a single colour across its background, but each bg of nominally the same colour will be different
        
    bg_only: bool, default=False 
        Omit the mnist digit and just create an image with bg colour
    
    standard_getitem: bool, default=False  
        The dataloaders by default return a tuple of (image, target, bias_target) which is not compatible with some 
        other code expecting only (image, target). If True, the bias_target information will be omitted
    
    bias_targets_as_targets: bool, default=False 
        The dataloaders by default return a tuple of (image, target, bias_target). This setting allows the tuple to
        be (image, bias_target) and overrides standard_getitem
    """

    COLOUR_MAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [0, 255, 255], [255, 128, 0], [255, 0, 128], [128, 0, 255], [128, 128, 128]]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9,
                 bg_noise_level=0.0, bg_only=False, standard_getitem=False, bias_targets_as_targets=False):
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.random = True

        self.data_label_correlation = data_label_correlation
        self.n_confusing_labels = n_confusing_labels
        self.bg_noise_level = bg_noise_level
        self.bg_only = bg_only
        self.standard_getitem = standard_getitem
        self.bias_targets_as_targets = bias_targets_as_targets
        self.data, self.targets, self.biased_targets = self.build_biased_mnist()
        indices = np.arange(len(self.data))
        self._shuffle(indices)

        self.data = self.data[indices].numpy()
        self.targets = self.targets[indices]
        self.biased_targets = self.biased_targets[indices]

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)

    def _make_biased_mnist(self, indices, label):
        raise NotImplementedError

    def _update_bias_indices(self, bias_indices, label):
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)

        indices = np.where((self.targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)  # a shuffled list of indices to targets where the target value matches the label

        n_samples = len(indices)  # i.e. the number of samples matching the label
        n_correlated_samples = int(n_samples * self.data_label_correlation)
        n_decorrelated_per_class = int(np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels)))

        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])  # first store the matching target indices

        decorrelated_indices = torch.split(indices[n_correlated_samples:], n_decorrelated_per_class)

        other_labels = [_label % 10 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
        self._shuffle(other_labels)  # shuffled list of all labels other than the one given

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])  # put some of the indices matching this label against a different index in bias_indices

    def build_biased_mnist(self):
        """Build biased MNIST.
        """
        n_labels = self.targets.max().item() + 1

        bias_indices = {label: torch.LongTensor() for label in range(n_labels)}
        for label in range(n_labels):
            self._update_bias_indices(bias_indices, label)
        # if correlating then bias_indices is a table of indexes for each target label. If decorrelating then each entry will include a mix of accurate and inaccurate labels
        data = torch.ByteTensor()
        targets = torch.LongTensor()
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_mnist(indices, bias_label)
            data = torch.cat([data, _data])
            targets = torch.cat([targets, _targets])
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = torch.LongTensor(biased_targets)
        return data, targets, biased_targets

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.standard_getitem:
            return img, target

        if self.bias_targets_as_targets:
            return img, int(self.biased_targets[index])
            
        return img, target, int(self.biased_targets[index])


class ColourBiasedMNIST(BiasedMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9,
                 bg_noise_level=0.0, bg_only=False, standard_getitem=False, bias_targets_as_targets=False):
        super(ColourBiasedMNIST, self).__init__(root, train=train, transform=transform,
                                                target_transform=target_transform,
                                                download=download,
                                                data_label_correlation=data_label_correlation,
                                                n_confusing_labels=n_confusing_labels,
                                                bg_noise_level=bg_noise_level, bg_only=bg_only,
                                                standard_getitem=standard_getitem, bias_targets_as_targets=bias_targets_as_targets)
    
    
    def _binary_to_colour(self, data, colour):    
        fg_data = torch.zeros_like(data)
        if not self.bg_only:
            fg_data[data != 0] = 255
            fg_data[data == 0] = 0
        fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1)
    
        if not self.bg_only:
            bg_data = torch.zeros_like(data)
            bg_data[data == 0] = 1
            bg_data[data != 0] = 0
        else:
            bg_data = torch.ones_like(data)
        
        bg_data = bg_data.unsqueeze(3).expand(-1, -1, -1, 3)
        
        colour = torch.tensor(colour, dtype=torch.uint8)
        noise =  self.bg_noise_level * torch.randn(len(bg_data),3).clamp(-1,1)
        noisy_colour =  (colour + noise * 255).clamp(0, 255).byte()
        bg_data = bg_data * noisy_colour.view(data.size(0), 1, 1, 3)
            
        bg_data = bg_data.permute(0, 3, 1, 2)
        
        data = fg_data + bg_data
        return data.permute(0, 2, 3, 1)
    
    def _make_biased_mnist(self, indices, label):
        return self._binary_to_colour(self.data[indices], self.COLOUR_MAP[label]), self.targets[indices]


def get_biased_mnist_dataloader(root, batch_size, data_label_correlation,
                                n_confusing_labels=9, train=True, num_workers=8,
                                bg_noise_level=0.0, bg_only=False, standard_getitem=False, bias_targets_as_targets=False, dl_transform=transforms.Compose([])):
    transform = transforms.Compose([
        dl_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
    dataset = ColourBiasedMNIST(root, train=train, transform=transform,
                                download=True, data_label_correlation=data_label_correlation,
                                n_confusing_labels=n_confusing_labels,
                                bg_noise_level=bg_noise_level, bg_only=bg_only, 
                                standard_getitem=standard_getitem, bias_targets_as_targets=bias_targets_as_targets)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader

'''
Only produce greyscale mnist data
'''
class GreyBiasedMNIST(BiasedMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, data_label_correlation=1.0, n_confusing_labels=9, standard_getitem=False, bias_targets_as_targets=False):
        super(GreyBiasedMNIST, self).__init__(root, train=train, transform=transform,
                                                target_transform=target_transform,
                                                download=download,
                                                data_label_correlation=data_label_correlation,
                                                n_confusing_labels=n_confusing_labels,
                                                standard_getitem=standard_getitem, bias_targets_as_targets=bias_targets_as_targets)
    
    
    def _binary_to_grey(self, data):    
        assert torch.max(data) <= 255
        fg_data = torch.stack([data, data, data], dim=1)
         
        return fg_data.permute(0, 2, 3, 1)
    
    def _make_biased_mnist(self, indices, label):
        return self._binary_to_grey(self.data[indices]), self.targets[indices]


'''
Combines 3 types of dataset in order to train networks to represent colour, shape, and both combined
by sometimes presenting just 
background colour
greyscale mnist, or 
bias(=correlated) colour mnist

'''
def get_mixed_mnist_dataloader(root, batch_size, n_confusing_labels=9, train=True, num_workers=8,
                               bg_noise_level=0.0, standard_getitem=False, bias_targets_as_targets=False, dl_transform=transforms.Compose([])):
    
    transform = transforms.Compose([
        dl_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
    
    # mnist dataset is already mixed classes, so taking first 1/3 should be representative
    data_label_correlation = 1.0  # fully correlated
    bg_only = False  # digit and coloured background together
    dataset_biased = ColourBiasedMNIST(root, train=train, transform=transform,
                                       download=True, data_label_correlation=data_label_correlation,
                                       n_confusing_labels=n_confusing_labels, bg_noise_level=bg_noise_level, bg_only=bg_only, 
                                       standard_getitem=standard_getitem, bias_targets_as_targets=bias_targets_as_targets)
    dataset_biased = torch.utils.data.Subset(dataset_biased, range(0, len(dataset_biased)//3))

    # For the bg_only, take middle 1/3 to avoid repetition
    data_label_correlation = 1.0  # fully correlated
    bg_only = True  # digit and coloured background together
    dataset_bgonly = ColourBiasedMNIST(root, train=train, transform=transform,
                                       download=True, data_label_correlation=data_label_correlation,
                                       n_confusing_labels=n_confusing_labels, bg_noise_level=bg_noise_level, bg_only=bg_only, 
                                       standard_getitem=standard_getitem, bias_targets_as_targets=bias_targets_as_targets)
    dataset_bgonly = torch.utils.data.Subset(dataset_bgonly, range(len(dataset_bgonly)//3, 2*len(dataset_bgonly)//3))

    # For the natural, take last 1/3 to avoid repetition
    data_label_correlation = 1.0  # fully correlated
    bg_only = False  # digit and coloured background together
    dataset_natural = GreyBiasedMNIST(root, train=train, transform=transform,
                                      download=True, data_label_correlation=data_label_correlation,
                                      n_confusing_labels=n_confusing_labels, 
                                      standard_getitem=standard_getitem, bias_targets_as_targets=bias_targets_as_targets)
    dataset_natural = torch.utils.data.Subset(dataset_natural, range(2*len(dataset_natural)//3, len(dataset_natural)))


    dataloader = data.DataLoader(dataset=torch.utils.data.ConcatDataset([dataset_natural, dataset_biased, dataset_bgonly]),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return dataloader
