import os.path as osp
from PIL import Image
import torch.utils.data as data
import numpy as np


class OurDataset(data.Dataset):
    def __init__(
            self,
            root,
            img_dir,
            anno_file,
            transform=None,
        ):
        self.transform = transform
        self.root = root
        self.img_dir = img_dir
        img_paths = []
        labels = []
        with open(osp.join(root, anno_file)) as f:
            for line in f:
                path, label = line.strip().split(' ')
                img_paths.append(path)
                labels.append(int(label))
        self.img_paths = img_paths
        self.labels = labels

    def __getitem__(self, index):
        img_path = osp.join(self.root, self.img_dir, self.img_paths[index])
        label = self.labels[index] - 1
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_paths)
