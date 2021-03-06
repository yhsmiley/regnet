"""
@author: Signatrix GmbH
Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)
"""
import os
import re
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from src.transforms_fixres import get_transforms


class Imagenet(Dataset):
    def __init__(self, root_dir, mode="train"):
        self.data_dir = os.path.join(root_dir, mode)
        self.mode = mode
        self.set_name = set
        self.load_categories()

    def load_categories(self):
        # self.raw_category_ids = sorted(file_ for file_ in os.listdir(self.data_dir) if re.match(r"^n[0-9]+$", file_))
        self.raw_category_ids = sorted(file_ for file_ in os.listdir(self.data_dir))
        self.fine_category_ids = {value: key for key, value in enumerate(self.raw_category_ids)}
        self.images = []
        self.targets = []
        for raw_id in self.raw_category_ids:
            fine_id = self.fine_category_ids[raw_id]
            dir = os.path.join(self.data_dir, raw_id)
            self.images.extend([{"image": os.path.join(dir, image), "category": fine_id} for image in os.listdir(dir)])
            self.targets.extend([fine_id for image in os.listdir(dir)])

    def transform(self, img, finetune=True):
        transf = get_transforms(kind='full', crop=True, need=('train', 'val'))

        if self.mode == "train":
            if finetune:
                transformation = transf['val_train']
            else:
                transformation = transf['train']
        else:
            transformation = transf['val']

        pil_img = Image.fromarray(img)
        img_tensor = transformation(pil_img)
        img = img_tensor.cpu().numpy()

        return img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        try:
            img = cv2.imread(self.images[index]["image"])
            img = img.astype(np.uint8, copy=False)
            img = self.transform(img)
            category = self.images[index]["category"]
        except:
            return None
        return img, category
