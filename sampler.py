import math

import torch
import torch.utils.data
import torchvision

import data_util

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        dataset_type = type(dataset)
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if type(label) is list:
                for lab in label:
                    if lab in label_to_count:
                        label_to_count[lab] += 1
                    else:
                        label_to_count[lab] = 1
            else:
                if label in label_to_count:
                    label_to_count[label] += 1
                else:
                    label_to_count[label] = 1

        # weight for each sample
        if dataset_type is data_util.pdFilesDataset:
            label_to_weight = {label: count**0.6 for label, count in label_to_count.items()}
            max_weight = max(label_to_weight.values())
            label_to_weight = {label: max_weight / weight for label, weight in label_to_weight.items()}
            # import pdb; pdb.set_trace()
            weights = []
            for idx in self.indices:
                label = self._get_label(dataset, idx)
                w = sum([label_to_weight[l] for l in label])
                weights.append(w)
        else:
            weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                       for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        elif dataset_type is data_util.pdFilesDataset:
            return dataset.get_raw_label(idx)
        else:
            raise NotImplementedError(f'{dataset_type} not supported')

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
