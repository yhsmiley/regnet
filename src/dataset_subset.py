import torch
from collections import Counter
from torch.utils.data import Dataset


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
