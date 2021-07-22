import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

ASCII = [chr(i) for i in range(48, 58)] + [chr(j) for j in range(97, 123)]
ASCII = dict(zip(ASCII, range(10 + 26)))


def default_loader(path):
    return Image.open(path).convert('RGB')


def one_hot(chars):
    tmp = [0 for i in range(36 * 4)]
    for step, char in enumerate(chars):
        tmp[ASCII[char] + 36 * step] = 1
    return tmp


class VerifyCode(Dataset):

    def __init__(self, path, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        files = os.listdir(path)
        for file in files:
            chars = os.path.splitext(file)[0]
            imgs.append(
                [os.path.join(path, file), one_hot(chars)]
            )
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.imgs)


def build_dataloader(filepath):
    transform = transforms.Compose(
        [transforms.ToTensor(), ]
    )

    ds = datasets.ImageNet(
        filepath,
        transform=transform,
        target_transform=lambda t: torch.tensor([t]).float()
    )

    dl = DataLoader(ds, batch_size=50, shuffle=True)

    return dl


if __name__ == "__main__":
    train_data = VerifyCode(path="../data/testing", transform=transforms.ToTensor())
    data_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    print(len(data_loader))

    print("DONE!!!")
