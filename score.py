from train import Net, PetImagesDataset
from torch.utils.data import Dataset, random_split, DataLoader
import torch

if __name__ == "__main__":
    model = Net()
    model.load_state_dict(
        torch.load("experiments/models/model_state_dict_1638914648.861017")
    )
    model.eval()
    print(model)

    pet_images_submission = DataLoader(
        PetImagesDataset("data/test.csv", "data/test", (28, 28)), batch_size=1
    )

    X, y = next(iter(pet_images_submission))

    print(X)
    print(y)

    pred = model(X)
    actual = y.item()
    print(f"pred = {pred.item()}, actual = {actual}")
