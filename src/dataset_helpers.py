import torch
import bisect
from collections import Counter
from torch.utils.data import Dataset, IterableDataset


class MyConcatDataset(Dataset):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.targets = []
        for dataset in self.datasets:
            assert not isinstance(dataset, IterableDataset), "ConcatDataset does not support IterableDataset"
            self.targets.extend(dataset.targets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.classes = datasets[0].classes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class MySubset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.targets = [target if idx in self.indices else len(self.dataset.classes) for idx, target in enumerate(self.dataset.targets)]

        # for weighted sampling
        class_count = dict(Counter(target for target in self.targets if target != len(self.dataset.classes)))
        class_count = dict(sorted(class_count.items()))
        class_count = list(class_count.values())
        class_weights = [len(self.indices)/cls_count for cls_count in class_count]
        class_weights.append(0)
        class_weights = torch.FloatTensor(class_weights)
        self.image_weights = class_weights[self.targets]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image = self.dataset[self.indices[index]][0]
        target = self.targets[self.indices[index]]
        return (image, target)


class MapDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function to its items.

    Note that data is not cloned/copied from the initial dataset
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        img_tensor = self.map(img)
        img = img_tensor.cpu().numpy()
        return img, target