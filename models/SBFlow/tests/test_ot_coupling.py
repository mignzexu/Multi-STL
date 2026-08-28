import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

from models.SBFlow.config import SBFlow_Config
from models.SBFlow import sbflow_math


class FakePOT(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.calls = []

    def unif(self, n):
        return np.ones(n) / n

    def emd(self, a, b, cost, numThreads=1):
        self.calls.append(("emd", a, b, cost, numThreads))
        return np.eye(cost.shape[0], cost.shape[1]) / cost.shape[0]

    def sinkhorn(self, a, b, cost, reg):
        self.calls.append(("sinkhorn", a, b, cost, reg))
        return np.eye(cost.shape[0], cost.shape[1]) / cost.shape[0]


class SBFlowOTCouplingTest(unittest.TestCase):
    def test_default_config_disables_ot(self):
        cfg = SBFlow_Config("lite").model_config
        self.assertFalse(cfg["sbflow_use_ot"])
        self.assertEqual(cfg["sbflow_ot_method"], "sinkhorn")

    def test_ot_config_enables_sinkhorn(self):
        cfg = SBFlow_Config("ot").model_config
        self.assertTrue(cfg["sbflow_use_ot"])
        self.assertEqual(cfg["sbflow_ot_method"], "sinkhorn")

    def test_importing_math_does_not_import_pot(self):
        sys.modules.pop("ot", None)
        self.assertNotIn("ot", sys.modules)
        self.assertIs(sbflow_math.SBFlowOTPlanSampler.__module__, sbflow_math.__name__)
        self.assertNotIn("ot", sys.modules)

    def test_sinkhorn_uses_squared_flattened_cost_and_sigma_reg(self):
        fake = FakePOT()
        x0 = torch.zeros(2, 1, 2, 2)
        x1 = torch.ones(2, 1, 2, 2)
        x1[1] = 2.0

        with mock.patch.dict(sys.modules, {"ot": fake}):
            sampler = sbflow_math.SBFlowOTPlanSampler(
                method="sinkhorn", reg=2.0, num_threads=3
            )
            sampler.get_map(x0, x1)

        name, a, b, cost, reg = fake.calls[-1]
        self.assertEqual(name, "sinkhorn")
        self.assertEqual(cost.shape, (2, 2))
        self.assertAlmostEqual(reg, 2.0)
        self.assertTrue(np.allclose(a, np.ones(2) / 2))
        self.assertTrue(np.allclose(b, np.ones(2) / 2))

    def test_exact_uses_num_threads(self):
        fake = FakePOT()
        x0 = torch.zeros(2, 1)
        x1 = torch.ones(2, 1)

        with mock.patch.dict(sys.modules, {"ot": fake}):
            sampler = sbflow_math.SBFlowOTPlanSampler(
                method="exact", reg=2.0, num_threads=7
            )
            sampler.get_map(x0, x1)

        name, _, _, _, num_threads = fake.calls[-1]
        self.assertEqual(name, "emd")
        self.assertEqual(num_threads, 7)

    def test_sample_plan_reindexes_both_batches(self):
        fake = FakePOT()
        x0 = torch.tensor([[0.0], [1.0]])
        x1 = torch.tensor([[10.0], [20.0]])

        with mock.patch.dict(sys.modules, {"ot": fake}):
            sampler = sbflow_math.SBFlowOTPlanSampler(method="sinkhorn", reg=2.0)
            with mock.patch.object(
                sampler,
                "sample_map",
                return_value=(np.array([1, 0]), np.array([0, 1])),
            ):
                coupled_x0, coupled_x1 = sampler.sample_plan(x0, x1)

        self.assertTrue(torch.equal(coupled_x0, torch.tensor([[1.0], [0.0]])))
        self.assertTrue(torch.equal(coupled_x1, torch.tensor([[10.0], [20.0]])))


if __name__ == "__main__":
    unittest.main()
