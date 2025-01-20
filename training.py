import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tqdm
from lib.dataset import PDEDatasetAll2All


def train_model(
    model,
    train_data,
    epsilon_values,
    time_points,
    batch_size=32,
    epochs=100,
    device="cuda",
    learning_rate=1e-3,
    curriculum_steps=None,
):
    """
    Training loop with curriculum learning on epsilon values.

    curriculum_steps: list of (epoch, epsilon_subset) tuples defining when to introduce each epsilon value
    """
    dataset_train = (
        PDEDatasetAll2All("data/train_allen_cahn_fourier.npz", device=device)
        + PDEDatasetAll2All("data/train_allen_cahn_gmm.npz", device=device)
        + PDEDatasetAll2All("data/train_allen_cahn_piecewise.npz", device=device)
    )

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset_train, [int(0.8 * len(dataset_train)), int(0.2 * len(dataset_train))]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

    model = model.to(device)
    best_val_loss = float("inf")

    progress_bar = tqdm.tqdm(range(epochs))

    for epoch in progress_bar:
        # Update curriculum if needed
        if curriculum_steps:
            for step_epoch, eps_subset in curriculum_steps:
                if epoch == step_epoch:
                    train_dataset = AllenCahnDataset(
                        train_data, eps_subset, time_points
                    )
                    train_loader = DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True
                    )
                    print(
                        f"Curriculum update: now training on epsilon values {eps_subset}"
                    )

        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            # Forward pass - implement your model to handle these inputs
            pred = model(batch["initial"], batch["epsilon"], batch["times"])

            # Compute loss - you might want to modify this
            loss = nn.MSELoss()(pred, batch["target"])

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(batch["initial"], batch["epsilon"], batch["times"])
                val_loss += nn.MSELoss()(pred, batch["target"]).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

    return model


# Example curriculum steps
curriculum_steps = [
    (0, [0.1]),  # Start with largest epsilon
    (20, [0.1, 0.05]),  # Add medium epsilon
    (40, [0.1, 0.05, 0.02]),  # Add smallest epsilon
]
