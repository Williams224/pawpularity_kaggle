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
import timm
import math
from datetime import datetime


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
        image = np.transpose(image, (2, 1, 0))
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
    return figure


def compare_images(i1, i2):
    f, axarr = plt.subplots(1, 2, figsize=(16, 8))
    axarr[0].imshow(i1)
    axarr[0].set_axis_off()
    axarr[1].imshow(i2)
    axarr[1].set_axis_off()
    plt.show()
    return f


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = timm.create_model(
            "efficientnet_b3", pretrained=True, num_classes=100
        )
        self.fc1 = nn.Linear(100, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.model(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer, device):
    running_loss = 0.0
    n_batches = 0
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        print(batch)
        X = X.float().to(device)
        y = y.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
        current_training_loss = running_loss / n_batches
        print(f"Done {batch} of {num_batches}")
        print(f"batch loss = {loss}")
        print(f"current average training loss = {current_training_loss}")

    training_loss = running_loss / n_batches
    print(f"training loss is: {training_loss}")
    rmse = math.sqrt(training_loss)
    print(f" avg rmse: {rmse}")
    return loss


def test_loop(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    test_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.float().to(device)
            y = y.float().unsqueeze(1).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    avg_test_loss = test_loss / num_batches
    print(f"Avg loss: {avg_test_loss:>8f} \n")
    rmse = math.sqrt(avg_test_loss)
    print(f" avg rmse: {rmse}")


if __name__ == "__main__":

    target_resize = (28, 28)

    pet_images = PetImagesDataset("data/train.csv", "data/train", target_resize)

    lengths = [int(len(pet_images) * 0.8), int(len(pet_images) * 0.2) + 1]
    training_dataset, test_dataset = random_split(pet_images, lengths)

    batch_size = 256

    train_dataloader = DataLoader(
        training_dataset, batch_size=batch_size, num_workers=14
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=14)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device = {device}")
    model = Net().to(device)

    print(model)

    loss_fn = nn.MSELoss()

    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    epochs = 5
    for t in range(epochs):
        start_time = datetime.now()
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
        end_time = datetime.now()
        time_diff = (end_time - start_time).seconds / 60.0
        print(f"Epoch took {time_diff} minute")
    print("Done!")

    timestamp = datetime.utcnow().timestamp()

    torch.save(model.state_dict(), f"experiments/models/model_state_dict_{timestamp}")
