import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as np


class PetImagesDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        train=True,
        transform=None,
        target_transform=None,
    ):
        self.img_labels = pd.read_csv(annotations_file)[["Id", "Pawpularity"]]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".jpg")
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def plot_random_image(dataset):
    figure = plt.figure(figsize=(8, 8))
    sample_idx = torch.randint(len(dataset), size=(1,)).item()
    img, label = dataset[sample_idx]
    plt.title(label)
    plt.axis("off")
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":

    pet_images = PetImagesDataset("data/train.csv", "data/train")

    plot_random_image(
        pet_images,
    )

    lengths = [int(len(pet_images) * 0.8), int(len(pet_images) * 0.2) + 1]
    print(lengths)
    print(len(pet_images))
    training_dataset, test_dataset = random_split(pet_images, lengths)

    batch_size = 256

    train_dataloader = DataLoader(
        training_dataset, batch_size=batch_size, num_workers=8
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    
