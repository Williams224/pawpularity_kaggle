from train import Net, PetImagesDataset
import torch

if __name__ == "__main__":
    model = Net()
    model.load_state_dict(
        torch.load("experiments/models/model_state_dict_1638914648.861017")
    )
    model.eval()
    print(model)

    pet_images = PetImagesDataset("data/train.csv", "data/train", (28, 28))

    pred = model(torch.from_numpy(pet_images[0][0]).float())
    actual = pet_images[0][1]
    print(f"pred = {pred}, actual = {actual}")
