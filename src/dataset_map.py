"""
@author: Signatrix GmbH
Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)
"""
from torch.utils.data import Dataset


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
