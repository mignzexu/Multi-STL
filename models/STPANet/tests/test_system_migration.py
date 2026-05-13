import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ConfigNode(dict[str, object]):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def build_configs(test_seq=2):
    obj_dir = tempfile.mkdtemp(prefix="stpanet-system-migration-")
    return ConfigNode(
        total_seq=[2, 2],
        test_seq=test_seq,
        in_category=["tp"],
        out_category=["tp"],
        label_idx=[0, 1],
        img_size=[8, 8],
        N_S=2,
        N_T=2,
        hid_S=4,
        hid_T=8,
        mlp_ratio=2,
        drop=0.0,
        drop_path=0.0,
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        learning_rate=5e-4,
        weight_decay=0.0,
        lr_min=1e-6,
        warmup_t=0,
        k_decay=1.0,
        epoch=3,
        batch_size=2,
        metrics=["mae", "mse"],
        threshold=[[0.5]],
        std_method="z_score",
        std_params={
            "dataset": {"mean": [[[[0.0]]]], "std": [[[[1.0]]]]},
            "metric": {"mean": [[[[[0.0]]]]], "std": [[[[[1.0]]]]]},
        },
        obj_dir=obj_dir,
    )


class STPANetSystemMigrationTests(unittest.TestCase):
    def test_package_exports_model_entrypoint(self):
        stpanet = importlib.import_module("models.STPANet")

        self.assertIsNotNone(stpanet.Model)
        self.assertIsNotNone(stpanet.Main)
        self.assertIsNotNone(stpanet.STPANet_Config)
        self.assertIsNotNone(stpanet.loss_fn)

    def test_model_subclasses_system(self):
        system_module = importlib.import_module("models.Model_system")
        stpanet = importlib.import_module("models.STPANet")

        self.assertTrue(issubclass(stpanet.Model, system_module.System))

    def test_forward_uses_test_seq_as_public_horizon(self):
        stpanet = importlib.import_module("models.STPANet")

        batch_x = torch.randn(2, 2, 1, 8, 8)

        equal_module = stpanet.Model(build_configs(test_seq=2))
        equal_output = equal_module(batch_x)
        self.assertEqual(equal_output.shape, (2, 2, 1, 8, 8))

        truncate_module = stpanet.Model(build_configs(test_seq=1))
        truncate_output = truncate_module(batch_x)
        self.assertEqual(truncate_output.shape, (2, 1, 1, 8, 8))

        rollout_module = stpanet.Model(build_configs(test_seq=5))
        rollout_output = rollout_module(batch_x)
        self.assertEqual(rollout_output.shape, (2, 5, 1, 8, 8))

    def test_training_and_validation_step_match_system_contract(self):
        stpanet = importlib.import_module("models.STPANet")

        module = stpanet.Model(build_configs(test_seq=2))
        batch_x = torch.randn(2, 2, 1, 8, 8)
        batch_y = torch.randn(2, 2, 1, 8, 8)

        train_out = module.training_step((batch_x, batch_y), 0)
        self.assertIn("loss", train_out)
        self.assertIn("train_loss", train_out)
        self.assertTrue(torch.is_tensor(train_out["loss"]))
        self.assertIsInstance(train_out["train_loss"], float)

        val_out = module.validation_step((batch_x, batch_y), 0)
        self.assertIn("val_loss", val_out)
        self.assertIn("output", val_out)
        self.assertIn("label", val_out)
        self.assertEqual(val_out["output"].shape, val_out["label"].shape)
        self.assertIsInstance(val_out["val_loss"], float)

    def test_configure_optimizers_returns_lightning_structure(self):
        stpanet = importlib.import_module("models.STPANet")

        module = stpanet.Model(build_configs(test_seq=2))
        optim_config = module.configure_optimizers()

        self.assertIn("optimizer", optim_config)
        self.assertIsInstance(optim_config["optimizer"], torch.optim.Adam)
        if "lr_scheduler" in optim_config:
            self.assertIn("scheduler", optim_config["lr_scheduler"])


if __name__ == "__main__":
    unittest.main()
