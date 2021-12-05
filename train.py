import os
from PIL.Image import new
import pandas as pd
from torchvision.io import image, read_image
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.transform import resize


class PetImagesDataset(Dataset):
    def __init__(self, annotations_file, img_dir, downsize_output):
        self.img_labels = pd.read_csv(annotations_file)[["Id", "Pawpularity"]]
        self.img_dir = img_dir
        self.downsize_output = downsize_output

    def downsampling(self, image, new_shape):
        image = image / 255
        image_resized = resize(image, (new_shape[0], new_shape[1]), anti_aliasing=True)
        return image_resized

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".jpg")
        image = np.transpose(read_image(img_path), (1, 2, 0))
        image = self.downsampling(image, self.downsize_output)
        label = self.img_labels.iloc[idx, 1]
        return image, label


def plot_random_image(dataset):
    figure = plt.figure(figsize=(8, 8))
    sample_idx = torch.randint(len(dataset), size=(1,)).item()
    img, label = dataset[sample_idx]
    plt.title(label)
    plt.axis("off")
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.sequential_stack = nn.Sequential(
            nn.Linear(28 * 28 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.sequential_stack(x)


def compare_images(i1, i2):
    f, axarr = plt.subplots(1, 2, figsize=(16, 8))
    axarr[0].imshow(i1)
    axarr[0].set_axis_off()
    axarr[1].imshow(i2)
    axarr[1].set_axis_off()
    plt.show()


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        print(batch)
        X = X.float()
        y = y.float()
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    loss = loss.item()
    print(f"training loss is: {loss}")
    return loss


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":

    pet_images = PetImagesDataset("data/train.csv", "data/train", (28, 28))

    lengths = [int(len(pet_images) * 0.8), int(len(pet_images) * 0.2) + 1]
    print(lengths)
    print(len(pet_images))
    training_dataset, test_dataset = random_split(pet_images, lengths)

    batch_size = 64

    train_dataloader = DataLoader(
        training_dataset, batch_size=batch_size, num_workers=8
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device)

    print(model)

    loss_fn = nn.MSELoss()

    learning_rate = 0.003
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
