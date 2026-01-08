"""
CLI to run MLP experiments from the SelfNormalizingNetworks notebook.

Usage example:
    python scripts/run_experiment.py --exp-name test_exp \
        --activation relu --hidden-dim 512 --epochs 20 --wandb

This script:
 - loads MNIST (from ./dataset)
 - constructs an MLP with two hidden layers (hidden_dim, hidden_dim//2)
 - trains the model and evaluates on validation/test splits
 - logs metrics to wandb when enabled
 - saves best model and metrics to local logs/<exp_name>/
"""

import argparse
import json
import os
import copy
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms

from sklearn.metrics import accuracy_score

# optional import
try:
    import swanlab as wandb  # type: ignore
except Exception:
    wandb = None


device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Model and activations (copied/adapted from notebook) ----------
class ExpActivation(nn.Module):
    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, input):
        return torch.exp(input)


class ReEUActivation(nn.Module):
    def __init__(self):
        super(ReEUActivation, self).__init__()

    def forward(self, input):
        return torch.exp(nn.functional.relu(input)) - 1.0


class SoftmaxActivation(nn.Module):
    def __init__(self):
        super(SoftmaxActivation, self).__init__()

    def forward(self, input):
        return nn.functional.softmax(input, dim=-1)


class LinearActivation(nn.Module):
    """
    Applies the Softmax activation function elementwise
    """

    def __init__(self):
        super(LinearActivation, self).__init__()

    def forward(self, input):
        return input


class SparseActivation(nn.Module):
    """
    Applies the Softmax activation function elementwise
    topk
    = -1: no sparsity
    = 0: half sparsity
    > 0: number as sparsity
    """

    def __init__(self, topk=0):
        super(SparseActivation, self).__init__()
        self.topk = topk

    def forward(self, input):
        if self.topk == 0:
            topk = int(input.shape[-1] / 2)
        elif self.topk == -1:
            topk = input.shape[-1]
        else:
            topk = self.topk
        res = torch.zeros_like(input)
        with torch.no_grad():
            indices = torch.topk(input, topk).indices
        res = res.scatter(-1, indices, 1)
        return torch.mul(input, res)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: List[int],
        p_drop=0.2,
        act_fn: str = "relu",
        topk=-1,
        use_layer_norm: bool = True,
    ):
        super(MLP, self).__init__()

        if act_fn == "selu":
            activation = nn.SELU()
        elif act_fn == "relu":
            activation = nn.ReLU()
        elif act_fn == "exp":
            activation = ExpActivation()
        elif act_fn == "softmax":
            activation = SoftmaxActivation()
        elif act_fn == "reu":
            activation = ReEUActivation()
        elif act_fn == "elu":
            activation = nn.ELU()
        elif act_fn == "linear":
            activation = LinearActivation()
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

        dropout = nn.Dropout(p=p_drop)
        sparse = SparseActivation(topk=topk)

        layers = [nn.Flatten()]
        prev_dim = in_features
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            # layers.extend([activation, dropout])
            layers.extend([sparse, activation, dropout])
            # layers.extend([sparse, dropout])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_features))

        self.net = nn.Sequential(*layers)

        if act_fn == "selu":
            for param in self.net.parameters():
                if len(param.shape) == 1:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.kaiming_normal_(
                        param, mode="fan_in", nonlinearity="linear"
                    )

    def forward(self, x):
        return self.net(x)


class Accuracy(nn.Module):
    def forward(self, x, y):
        y_pred = F.softmax(x, dim=1).argmax(dim=1).cpu().numpy()
        y = y.cpu().numpy()
        return accuracy_score(y_true=y, y_pred=y_pred)


# ---------- Training utilities ----------


def _forward(network: nn.Module, data: DataLoader, metric: callable):
    for x, y in data:
        x = x.to(next(network.parameters()).device)
        y_hat = network.forward(x).cpu()
        loss = metric(y_hat, y)
        yield loss


@torch.enable_grad()
def update(
    network: nn.Module,
    data: DataLoader,
    loss: nn.Module,
    opt: torch.optim.Optimizer,
) -> list:
    network.train()
    errs = []
    for err in _forward(network, data, loss):
        errs.append(err.item())
        opt.zero_grad()
        err.backward()
        opt.step()
    return errs


@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric: callable) -> float:
    network.eval()
    performance = []
    for p in _forward(network, data, metric):
        p = np.array(p)
        performance.append(p.item())
    return np.mean(performance).item()


def fit(
    network: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    testloader: DataLoader,
    epochs: int,
    lr: float,
    exp_logger=None,
):
    # optimizer = torch.optim.SGD(params=network.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(
        params=network.parameters(), lr=lr, weight_decay=1e-3
    )
    ce = nn.CrossEntropyLoss()
    accuracy = Accuracy()

    train_losses, val_losses, accuracies = [], [], []

    val_losses.append(evaluate(network=network, data=valloader, metric=ce))

    best_model = copy.deepcopy(network)
    best_acc = 0.0

    pbar = tqdm(range(epochs))
    for ep in pbar:
        tl = update(network=network, data=trainloader, loss=ce, opt=optimizer)
        train_losses.extend(tl)
        vl = evaluate(network=network, data=valloader, metric=ce)
        val_losses.append(vl)
        ac = evaluate(network=network, data=valloader, metric=accuracy)

        if ac > best_acc:
            best_acc = ac
            best_model = copy.deepcopy(network)

        accuracies.append(ac)

        epoch_train_loss = float(np.mean(tl))
        epoch_val_loss = float(vl)
        epoch_val_acc = float(ac)

        # logging
        if exp_logger is not None:
            exp_logger.log_epoch(
                epoch=ep + 1,
                train_loss=epoch_train_loss,
                val_loss=epoch_val_loss,
                val_acc=epoch_val_acc,
            )

        print(
            f"Epoch {ep+1}: train loss: {epoch_train_loss:.4f}, val loss: {
                epoch_val_loss:.4f}, val acc: {epoch_val_acc*100:.2f}%"
        )
        pbar.set_description_str(desc=f"Epoch {ep+1}")

    acc = evaluate(network=best_model, data=testloader, metric=accuracy)
    print(f"Final accuracy on testset: {round(acc*100, 2):.2f}%")

    return train_losses, val_losses, accuracies, acc, best_model


# ---------- Experiment logger (wandb + local) ----------
class ExperimentLogger:
    def __init__(
        self,
        outdir: Path,
        exp_name: str,
        config: dict,
        use_wandb: bool = False,
        wandb_project: str = "activations",
    ):
        self.outdir = Path(outdir)
        self.exp_name = exp_name
        self.use_wandb = use_wandb and (wandb is not None)
        self.wandb_run = None
        self.config = config
        self.metrics = []

        os.makedirs(self.outdir, exist_ok=True)
        # save config
        with open(self.outdir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        if use_wandb and wandb is None:
            print(
                "wandb requested but package failed to import.",
                "Continuing without wandb.",
            )

        if self.use_wandb:
            self.wandb_run = wandb.init(
                project=wandb_project, name=self.exp_name, config=self.config
            )

    def log_epoch(
        self, epoch: int, train_loss: float, val_loss: float, val_acc: float
    ):
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        self.metrics.append(rec)
        # write latest metrics to disk
        with open(self.outdir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        # append csv
        import csv

        csv_path = self.outdir / "metrics.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc"]
            )
            if write_header:
                writer.writeheader()
            writer.writerow(rec)

        if self.use_wandb and self.wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

    def finalize(
        self,
        best_model: nn.Module,
        final_test_acc: float,
        train_losses,
        val_losses,
        accuracies,
    ):
        # save model
        model_path = self.outdir / "best_model.pt"
        torch.save(best_model.state_dict(), model_path)

        # save arrays
        np.save(self.outdir / "train_losses.npy", np.asarray(train_losses))
        np.save(self.outdir / "val_losses.npy", np.asarray(val_losses))
        np.save(self.outdir / "accuracies.npy", np.asarray(accuracies))

        # save final summary
        summary = {"final_test_acc": float(final_test_acc)}
        with open(self.outdir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if self.use_wandb and self.wandb_run is not None:
            wandb.log({"final_test_acc": final_test_acc})
            wandb.finish()


# ---------- Main and CLI ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "selu", "exp", "reu", "elu", "softmax", "linear"],
        help="activation to use",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="size of first hidden layer (second will be hidden_dim//2)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="DefaultTestRun",
        help="experiment name for logging and saving",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--p-drop", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layer-norm",
        action="store_true",
        default=True,
        help="enable layer normalization (enabled by default)",
    )
    parser.add_argument(
        "--no-layer-norm",
        dest="layer_norm",
        action="store_false",
        help="disable layer normalization",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=True,
        help="enable wandb logging (enabled by default)",
    )
    parser.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="disable wandb logging",
    )
    parser.add_argument("--wandb-project", type=str, default="activations")
    parser.add_argument("--outdir", type=str, default="logs")

    args = parser.parse_args()

    # system logger setup
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
    )
    sys_logger = logging.getLogger("run_experiment")

    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sys_logger.info(f"Random seed set to {args.seed}")

    # data
    path = os.path.join(".", "dataset", "mnist")
    os.makedirs(path, exist_ok=True)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = torchvision.datasets.MNIST(
        path, download=True, train=True, transform=transform
    )
    test = torchvision.datasets.MNIST(
        path, download=True, train=False, transform=transform
    )
    sys_logger.info(
        f"MNIST dataset loaded: train={len(train)}, test={len(test)}"
    )

    # val split
    rng = np.random.default_rng(seed=args.seed)
    val_inds = rng.choice(
        np.arange(len(train)), size=len(train) // 6, replace=False
    )
    train_inds = np.delete(np.arange(len(train)), val_inds)

    trainloader = DataLoader(
        Subset(train, indices=train_inds),
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valloader = DataLoader(
        Subset(train, indices=val_inds),
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers,
    )
    testloader = DataLoader(
        test,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
    )
    sys_logger.info(
        f"Dataloaders created: train={len(trainloader.dataset)}, val={
            len(valloader.dataset)}, test={
                len(testloader.dataset)}, batch_size={args.batch_size}"
    )

    # model
    hidden_sizes = [args.hidden_dim, max(1, args.hidden_dim // 2)]
    model = MLP(
        in_features=784,
        out_features=10,
        hidden_sizes=hidden_sizes,
        p_drop=args.p_drop,
        act_fn=args.activation,
        use_layer_norm=args.layer_norm,
        topk=args.topk,
    ).to(device)
    sys_logger.info(
        f"Model initialized: activation={args.activation}, hidden_sizes={
            hidden_sizes}, p_drop={args.p_drop}, layer_norm={
                args.layer_norm}"
    )

    # logger
    if args.wandb and wandb is None:
        sys_logger.warning("wandb requested but the package failed to import")
    outdir = Path(args.outdir) / args.exp_name
    config = vars(args)
    logger = ExperimentLogger(
        outdir=outdir,
        exp_name=args.exp_name,
        config=config,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )
    sys_logger.info(
        f"Experiment logger initialized at {outdir}. Wandb requested: {
            args.wandb}; Wandb active: {logger.use_wandb}"
    )

    # run training
    sys_logger.info(f"Starting training for {args.epochs} epochs...")
    train_losses, val_losses, accuracies, test_acc, best_model = fit(
        model,
        trainloader,
        valloader,
        testloader,
        epochs=args.epochs,
        lr=args.lr,
        exp_logger=logger,
    )
    sys_logger.info(f"Training finished. Test accuracy: {test_acc:.4f}")

    # finalize
    logger.finalize(
        best_model=best_model,
        final_test_acc=test_acc,
        train_losses=train_losses,
        val_losses=val_losses,
        accuracies=accuracies,
    )
    sys_logger.info(
        f"Experiment finalized. Outputs saved to: {Path(outdir).resolve()}"
    )


if __name__ == "__main__":
    main()
