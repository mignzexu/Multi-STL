import torch
from torch import optim

try:
    from ..Model_system import System
    from .drift_network import get_model as _drift_get_model
    from .loss import DriftFeatureLoss
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from models.Drift.drift_network import get_model as _drift_get_model
    from models.Drift.loss import DriftFeatureLoss
    from models.Model_system import System


class Model(System):
    def __init__(self, configs):
        self._negative_weight = float(getattr(configs, "drift_negative_weight", 0.0))
        self._max_tokens = int(getattr(configs, "drift_max_tokens", 1024))
        self._pos_bank_size = int(getattr(configs, "drift_pos_bank_size", 64))
        self._neg_bank_size = int(getattr(configs, "drift_neg_bank_size", 128))
        self._pos_samples = int(getattr(configs, "drift_pos_samples", 4))
        self._neg_samples = int(getattr(configs, "drift_neg_samples", 4))
        self._inference_reduce = getattr(configs, "drift_inference_reduce", "mean")
        super().__init__(configs)
        self.feature_loss = DriftFeatureLoss(
            negative_weight=self._negative_weight,
            max_tokens=self._max_tokens,
        )
        self._positive_bank = []
        self._negative_bank = []

    def get_model(self):
        return _drift_get_model(self.configs)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.configs.learning_rate,
            weight_decay=getattr(self.configs, "weight_decay", 0.0),
        )
        self._last_configured_optimizer = optimizer
        return {"optimizer": optimizer}

    def _predict_candidates(self, x0):
        deltas = self.model(x0)
        return x0.unsqueeze(1) + deltas

    def _reduce_candidates(self, candidates):
        if self._inference_reduce == "first":
            return candidates[:, 0]
        if self._inference_reduce != "mean":
            raise ValueError(f"Unknown drift_inference_reduce: {self._inference_reduce}")
        return candidates.mean(dim=1)

    def _predict_next_frame(self, x0):
        return self._reduce_candidates(self._predict_candidates(x0))

    def _sample_bank(self, bank, batch_size, sample_count, device, dtype):
        if sample_count <= 0 or len(bank) == 0:
            return None
        indices = torch.randint(len(bank), (batch_size * sample_count,))
        samples = torch.stack([bank[int(index)] for index in indices], dim=0)
        samples = samples.to(device=device, dtype=dtype)
        return samples.reshape(batch_size, sample_count, *samples.shape[1:])

    def _make_positive_refs(self, target):
        refs = [target.unsqueeze(1)]
        bank_refs = self._sample_bank(
            self._positive_bank,
            target.shape[0],
            self._pos_samples,
            target.device,
            target.dtype,
        )
        if bank_refs is not None:
            refs.append(bank_refs)
        return torch.cat(refs, dim=1)

    def _make_negative_refs(self, target):
        refs = []
        bank_refs = self._sample_bank(
            self._negative_bank,
            target.shape[0],
            self._neg_samples,
            target.device,
            target.dtype,
        )
        if bank_refs is not None:
            refs.append(bank_refs)
        if self._neg_samples > 0 and target.shape[0] > 1:
            refs.append(target.roll(shifts=1, dims=0).unsqueeze(1))
        if not refs:
            return None
        return torch.cat(refs, dim=1)

    def _push_bank(self, bank, samples, max_size):
        if max_size <= 0:
            return
        for sample in samples.detach().cpu():
            bank.append(sample)
        overflow = len(bank) - max_size
        if overflow > 0:
            del bank[:overflow]

    def _update_banks(self, target):
        self._push_bank(self._positive_bank, target, self._pos_bank_size)
        self._push_bank(self._negative_bank, target, self._neg_bank_size)

    def _loss(self, x0, x1, use_memory=True):
        candidates = self._predict_candidates(x0)
        positive_refs = self._make_positive_refs(x1) if use_memory else x1.unsqueeze(1)
        negative_refs = self._make_negative_refs(x1) if use_memory else None
        loss = self.feature_loss(candidates, positive_refs, negative_refs)
        return loss, self._reduce_candidates(candidates)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        x0 = batch_x[:, -1]
        x1 = batch_y[:, 0]
        loss, _ = self._loss(x0, x1)
        self._update_banks(x1)

        return {
            "loss": loss,
            "train_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
        }

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        x0 = batch_x[:, -1]
        x1 = batch_y[:, 0]
        loss, pred = self._loss(x0, x1, use_memory=False)

        lo, hi = self.label_idx[0], self.label_idx[1]
        return {
            "val_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
            "output": pred.unsqueeze(1)[:, :, lo:hi, :, :],
            "label": x1.unsqueeze(1)[:, :, lo:hi, :, :],
        }

    def forward(self, batch_x, batch_y=None, **kwargs):
        current = batch_x[:, -1]
        n_pred = min(self.test_seq, self.aft_seq_length)
        frames = []

        for _ in range(n_pred):
            current = self._predict_next_frame(current)
            frames.append(current)

        lo, hi = self.label_idx[0], self.label_idx[1]
        return torch.stack(frames, dim=1)[:, :, lo:hi, :, :]


if __name__ == "__main__":
    from pathlib import Path
    from types import SimpleNamespace

    torch.manual_seed(0)

    debug_dir = Path(__file__).resolve().parent / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    configs = SimpleNamespace(
        total_seq=[2, 3],
        test_seq=2,
        label_idx=[0, 1],
        in_category=["tp"],
        out_category=["tp"],
        img_size=[16, 16],
        learning_rate=1e-3,
        weight_decay=0.0,
        epoch=1,
        std_method="z_score",
        std_params={
            "dataset": {"mean": [[[[0.0]]]], "std": [[[[1.0]]]]},
            "metric": {"mean": [[[[[0.0]]]]], "std": [[[[[1.0]]]]]},
        },
        threshold=[[0.5]],
        metrics=["mae", "mse"],
        batch_size=2,
        obj_dir=str(debug_dir),
        drift_channels=1,
        drift_unet_config="auto",
        drift_gen_per_input=4,
        drift_pos_bank_size=8,
        drift_neg_bank_size=8,
        drift_pos_samples=2,
        drift_neg_samples=2,
        drift_negative_weight=0.1,
        drift_max_tokens=256,
        drift_inference_reduce="mean",
    )

    try:
        model = Model(configs)
        model.train()
        batch_x = torch.randn(2, 2, 1, 16, 16)
        batch_y = torch.randn(2, 3, 1, 16, 16)

        train_out = model.training_step((batch_x, batch_y), 0)
        train_out["loss"].backward()
        grad_sum = sum(
            p.grad.detach().abs().sum().item()
            for p in model.model.parameters()
            if p.grad is not None
        )
        if grad_sum <= 0.0:
            raise RuntimeError("no model gradients")

        model.eval()
        with torch.no_grad():
            val_out = model.validation_step((batch_x, batch_y), 0)
            pred = model.forward(batch_x)

        expected_shape = (2, 2, 1, 16, 16)
        if tuple(pred.shape) != expected_shape:
            raise RuntimeError(f"forward shape {tuple(pred.shape)} != {expected_shape}")
        if tuple(val_out["output"].shape) != (2, 1, 1, 16, 16):
            raise RuntimeError("validation output shape mismatch")

        print(
            "Drift smoke test PASS "
            f"loss={train_out['loss'].item():.6f} "
            f"val={val_out['val_loss'].item():.6f} "
            f"forward={list(pred.shape)}"
        )
    except Exception as exc:
        print(f"Drift smoke test FAIL {exc}")
        raise
