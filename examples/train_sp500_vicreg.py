
"""Small VICReg experiment on the SP500 time-series dataset."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stable_datasets.timeseries import SP500
from stable_pretraining.methods.vicreg import VICReg


torch.manual_seed(42)
torch.set_num_threads(2)


class FinancialDataset(Dataset):
    def __init__(self, split, limit):
        source = SP500(split=split).with_format("numpy")
        self.source = source
        self.length = min(limit, len(source))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        series = np.asarray(
            self.source[index]["series"], dtype=np.float32
        ).copy()

        # Scale log-volume to a range comparable to price returns.
        series[:, 4] /= 20.0

        return torch.from_numpy(series)


class TimeSeriesEncoder(nn.Module):
    """Simple encoder compatible with the VICReg implementation."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(150, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        # VICReg probes its encoder with a dummy 224x224 image.
        # Convert that initialization probe into a 30x5 input.
        if x.ndim == 4 and x.shape == (1, 3, 224, 224):
            x = x.new_zeros((1, 30, 5))

        return self.network(x)


def augment(x):
    """Create a slightly perturbed view of each financial sequence."""
    noise = torch.randn_like(x) * 0.005
    return x + noise


def evaluate(model, loader):
    model.eval()
    losses = []
    embeddings = []

    with torch.no_grad():
        for x in loader:
            output = model(augment(x), augment(x))
            losses.append(output.loss.item())

            features = model(x).embedding
            embeddings.append(features)

    embeddings = torch.cat(embeddings, dim=0)

    return (
        float(np.mean(losses)),
        embeddings.std(dim=0).mean().item(),
    )


def main():
    train_loader = DataLoader(
        FinancialDataset("train", 2048),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        FinancialDataset("validation", 512),
        batch_size=64,
    )
    test_loader = DataLoader(
        FinancialDataset("test", 512),
        batch_size=64,
    )

    model = VICReg(
        encoder_name=TimeSeriesEncoder(),
        projector_dims=(128, 128, 64),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        model.train()
        train_losses = []

        for x in train_loader:
            optimizer.zero_grad()
            output = model(augment(x), augment(x))
            loss = output.loss

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_loss, val_std = evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={np.mean(train_losses):.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_embedding_std={val_std:.4f}"
        )

    test_loss, test_std = evaluate(model, test_loader)

    print(f"Final test VICReg loss: {test_loss:.4f}")
    print(f"Final test embedding std: {test_std:.4f}")


if __name__ == "__main__":
    main()
