import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
import glob
from PIL import Image


class ImageListDataset(Dataset):
    def __init__(self, data_root, data_list, transform=None):
        super(ImageListDataset, self).__init__()
        self.data_root = data_root
        self.transform = transform
        self.img_list = []

        with open(data_list, 'r', encoding='utf-8') as f:
            for line in f:
                self.img_list.append(line.strip())

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        label = int(self.img_list[index].split('_', 1)[0])
        img_fn = os.path.join(self.data_root, self.img_list[index])

        img = Image.open(img_fn)
        '''
        if img.mode != 'RGB':
            img = img.convert('RGB')
        '''

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == '__main__':
    dataset = ImageListDataset('/workspace/personal/classification/dataset',
                               '/workspace/personal/classification/dataset/train.txt')
    img, label = dataset[10]
