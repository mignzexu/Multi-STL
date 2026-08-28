import importlib
import sys
import unittest
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MIGRATED_MODELS = {
    "convlstm": ("models.ConvLSTM", "ConvLSTM_Config"),
    "e3dlstm": ("models.E3DLSTM", "E3DLSTM_Config"),
    "mau": ("models.MAU", "MAU_Config"),
    "mim": ("models.MIM", "MIM_Config"),
    "mmvp": ("models.MMVP", "MMVP_Config"),
    "predrnnpp": ("models.PerdRNNpp", "PredRNNpp_Config"),
    "phydnet": ("models.PhyDNet", "PhyDNet_Config"),
    "poolformer": ("models.PoolFormer", "Poolformer_Config"),
    "prediff": ("models.PreDiff", "PreDiff_Config"),
    "predrnn": ("models.PredRNN", "PredRNN_Config"),
    "predrnnv2": ("models.PredRNNv2", "PredRNNv2_Config"),
    "simvp_incepu": ("models.SimVP_IncepU", "IncepU_Config"),
    "swinlstm": ("models.SwinLSTM", "SwinLSTM_Config"),
    "tau": ("models.TAU", "TAU_Config"),
    "stpanet": ("models.STPANet", "STPANet_Config"),
}


class MigrationContractTests(unittest.TestCase):
    def test_packages_export_only_model_and_config(self):
        for _, (module_name, config_name) in MIGRATED_MODELS.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, "Model"))
                self.assertTrue(hasattr(module, config_name))
                self.assertEqual(set(module.__all__), {"Model", config_name})
                self.assertFalse(hasattr(module, "Main"))
                self.assertFalse(hasattr(module, "loss_fn"))

    def test_models_subclass_system_and_define_required_methods(self):
        system_module = importlib.import_module("models.Model_system")
        required_methods = {
            "get_model",
            "forward",
            "configure_optimizers",
            "training_step",
            "validation_step",
        }
        for _, (module_name, _) in MIGRATED_MODELS.items():
            with self.subTest(module=module_name):
                model_cls = importlib.import_module(module_name).Model
                self.assertTrue(issubclass(model_cls, system_module.System))
                for method_name in required_methods:
                    self.assertIn(method_name, model_cls.__dict__)

    def test_instrument_registry_resolves_each_alias(self):
        from Instrument.models import Model_Instrument

        for alias in MIGRATED_MODELS:
            with self.subTest(alias=alias):
                instrument = Model_Instrument(Namespace(load_model=alias), mode="test")
                self.assertIs(instrument.model, importlib.import_module(MIGRATED_MODELS[alias][0]).Model)


if __name__ == "__main__":
    unittest.main()
