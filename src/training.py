import torch
import numpy as np
from torch.utils.data import DataLoader
import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt
from model import FNO1d
import os
from dataset import PDEDataset
import json
import random
import shutil


def relative_l2_error(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(output - target) / torch.norm(target)


torch.manual_seed(0)
np.random.seed(0)

BATCH_SIZE = 32
DEVICE = "mps"
training_id = random.randint(0, 1000000)

training_notes = """Test"""


dataset_train = PDEDataset(
    "data/train_allen_cahn_fourier.npy",
    device=DEVICE,
    # time_pairs=[(0, 4)],
)
dataset_validation = PDEDataset(
    "data/test_allen_cahn_fourier.npy",
    device=DEVICE,
    # time_pairs=[(0, 4)],
)

dataset_test = PDEDataset(
    "data/test_allen_cahn_fourier.npy",
    device=DEVICE,
    # time_pairs=[(0, 4)],
)

train_data_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(dataset_validation, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)


learning_rate = 0.001
epochs = 50
step_size = 2
gamma = 0.5

modes = 16
width = 64
fno = FNO1d(modes, width).to(DEVICE)  # model

optimizer = Adam(fno.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


training_losses = []
validation_losses = []
learning_rates = []


progress_bar = tqdm.tqdm(range(epochs))
for epoch in progress_bar:
    fno.train()
    train_loss = 0.0
    for i, (dt, input, target) in enumerate(train_data_loader):
        optimizer.zero_grad()
        prediction = fno(input, dt).squeeze(-1)

        loss = relative_l2_error(prediction, target)
        # loss += 0.01 * fourier_loss(prediction, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_data_loader)
    training_losses.append(train_loss)

    learning_rates.append(scheduler.get_last_lr())
    scheduler.step()

    with torch.no_grad():
        fno.eval()
        validation_relative_l2 = 0.0
        for i, (dt, input, target) in enumerate(val_data_loader):
            prediction = fno(input, dt).squeeze(-1)

            loss = relative_l2_error(prediction, target)
            validation_relative_l2 += loss.item()

        validation_relative_l2 /= len(val_data_loader)
        validation_losses.append(validation_relative_l2)

    progress_bar.set_postfix(
        {"train_loss": train_loss, "val_loss": validation_relative_l2}
    )

# validate model

fno.eval()
progress_bar = tqdm.tqdm(test_data_loader)

with torch.no_grad():
    test_relative_l2 = 0.0
    for i, (dt, input, target) in enumerate(progress_bar):
        prediction = fno(input, dt).squeeze(-1)

        loss = relative_l2_error(prediction, target)
        test_relative_l2 += loss.item()
    test_relative_l2 /= len(test_data_loader)


print("#" * 20)
print(f"Test relative L2 error: {test_relative_l2}")
print(f"Training ID: {training_id}")


# create dir for model
os.makedirs(f"models/{training_id}", exist_ok=True)

# plot losses
plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
# plt.plot(learning_rates, label="Learning Rate")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(f"models/{training_id}/loss.png")


# save model to disk
torch.save(fno.state_dict(), f"models/{training_id}/fno_model.pth")

# save training metadata to file
with open(f"models/{training_id}/metadata.json", "w") as f:
    json.dump(
        {
            "training_id": training_id,
            "training_losses": training_losses,
            "validation_losses": validation_losses,
            "test_relative_l2": test_relative_l2,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "step_size": step_size,
            "gamma": gamma,
            "modes": modes,
            "width": width,
            "notes": training_notes,
            "batch_size": BATCH_SIZE,
            "device": DEVICE,
        },
        f,
    )


# add copy of training, model and layers to the folder
shutil.copy("src/model/training.py", f"models/{training_id}/training.py")
shutil.copy("src/model/model.py", f"models/{training_id}/model.py")
shutil.copy("src/model/layers.py", f"models/{training_id}/layers.py")
