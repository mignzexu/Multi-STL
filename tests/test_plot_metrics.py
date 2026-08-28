# pyright: reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false
import json
import os
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLOT_METRICS_PATH = REPO_ROOT / "utils" / "plot_metrics.py"
spec = importlib.util.spec_from_file_location("plot_metrics", PLOT_METRICS_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"无法加载 {PLOT_METRICS_PATH}")
plot_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_metrics)


def sample_metrics() -> dict[str, Any]:
    return {
        "10": {
            "train": {"loss": 3.0, "lr": 0.0005},
            "valid": {
                "loss": 4.0,
                "w10": {
                    "mae": 0.4,
                    "mse": 0.16,
                    "rmse": 0.5,
                    "csi": [0.7, 0.6, 0.5],
                    "pod": [0.8, 0.7, 0.6],
                    "far": [0.1, 0.2, 0.3],
                    "hss": [0.9, 0.8, 0.7],
                },
            },
        },
        "2": {
            "train": {"loss": 5.0, "lr": 0.001},
            "valid": {
                "loss": 6.0,
                "w10": {
                    "mae": 0.6,
                    "mse": 0.36,
                    "rmse": 0.7,
                    "csi": [0.5, 0.4, 0.3],
                    "pod": [0.6, 0.5, 0.4],
                    "far": [0.3, 0.4, 0.5],
                    "hss": [0.7, 0.6, 0.5],
                },
            },
        },
        "1": {
            "train": {"loss": 7.0, "lr": 0.001},
            "valid": {
                "loss": 8.0,
                "w10": {
                    "mae": 0.8,
                    "mse": 0.64,
                    "rmse": 0.9,
                    "csi": [0.3, 0.2, 0.1],
                    "pod": [0.4, 0.3, 0.2],
                    "far": [0.5, 0.6, 0.7],
                    "hss": [0.5, 0.4, 0.3],
                },
            },
        },
    }


class PlotMetricsTests(unittest.TestCase):
    def write_metrics(self, work_dir: Path, ex_name: str, metrics: dict[str, Any]):
        run_dir = Path(work_dir) / ex_name
        run_dir.mkdir(parents=True)
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f)
        return run_dir, metrics_path

    def assert_png(self, path: Path):
        path = Path(path)
        self.assertTrue(path.exists(), f"missing png: {path}")
        self.assertGreater(path.stat().st_size, 0, f"empty png: {path}")
        self.assertEqual(path.suffix, ".png")

    def test_run_paths_match_visualization_style_arguments(self):
        run_dir, metrics_path, save_dir = plot_metrics.run_paths("work_dirs", "drift_w3s")
        self.assertEqual(run_dir, os.path.join("work_dirs", "drift_w3s"))
        self.assertEqual(metrics_path, os.path.join("work_dirs", "drift_w3s", "metrics.json"))
        self.assertEqual(save_dir, os.path.join("work_dirs", "drift_w3s", "TPro"))

    def test_run_paths_rejects_path_traversal_project_name(self):
        with self.assertRaises(ValueError):
            plot_metrics.run_paths("work_dirs", "../outside")
        with self.assertRaises(ValueError):
            plot_metrics.run_paths("work_dirs", "/tmp/outside")

    def test_safe_child_path_rejects_unsafe_output_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                plot_metrics.safe_child_path(tmp, "../outside")
            with self.assertRaises(ValueError):
                plot_metrics.safe_child_path(tmp, "/tmp/outside")

    def test_warning_helper_writes_stderr(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            plot_metrics.warn("bad metrics")
        self.assertIn("WARNING", stderr.getvalue())
        self.assertIn("bad metrics", stderr.getvalue())

    def test_epoch_keys_are_sorted_as_numbers(self):
        items = plot_metrics.sorted_epoch_items(sample_metrics())
        self.assertEqual([epoch for epoch, _ in items], [1, 2, 10])

    def test_invalid_json_is_skipped_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            metrics_path.write_text("{bad json", encoding="utf-8")
            stderr = StringIO()
            with redirect_stderr(stderr):
                metrics = plot_metrics.load_metrics(str(metrics_path))
            self.assertIsNone(metrics)
            self.assertIn("WARNING", stderr.getvalue())
            self.assertIn("格式错误", stderr.getvalue())

    def test_scalar_extraction_skips_invalid_points(self):
        metrics = sample_metrics()
        metrics["2"]["train"]["loss"] = "bad"
        stderr = StringIO()
        with redirect_stderr(stderr):
            xs, ys = plot_metrics.scalar_series(
                plot_metrics.sorted_epoch_items(metrics), ("train", "loss"), "train.loss"
            )
        self.assertEqual(xs, [1, 10])
        self.assertEqual(ys, [7.0, 3.0])
        self.assertIn("WARNING", stderr.getvalue())

    def test_plot_metrics_writes_loss_lr_and_validation_pngs(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp) / "TPro"
            wrote = plot_metrics.plot_metrics(sample_metrics(), str(save_dir))
            self.assertTrue(wrote)

            self.assert_png(save_dir / "Loss.png")
            self.assert_png(save_dir / "LR.png")
            self.assert_png(save_dir / "w10" / "MAE_RMSE.png")
            self.assert_png(save_dir / "w10" / "CSI.png")
            self.assert_png(save_dir / "w10" / "POD.png")
            self.assert_png(save_dir / "w10" / "FAR.png")
            self.assert_png(save_dir / "w10" / "HSS.png")
            self.assertFalse((save_dir / "w10" / "MSE.png").exists())

    def test_new_scalar_list_metrics_and_modal_groups_plot_automatically(self):
        metrics = sample_metrics()
        for epoch in metrics.values():
            epoch["valid"]["w10"]["bias"] = 0.2
            epoch["valid"]["radar"] = {
                "accuracy": 0.9,
                "mse": 0.01,
                "fss": [0.1, 0.2],
            }

        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp) / "TPro"
            wrote = plot_metrics.plot_metrics(metrics, str(save_dir))
            self.assertTrue(wrote)

            self.assert_png(save_dir / "w10" / "BIAS.png")
            self.assert_png(save_dir / "radar" / "ACCURACY.png")
            self.assert_png(save_dir / "radar" / "FSS.png")
            self.assertFalse((save_dir / "radar" / "MSE.png").exists())
            self.assertFalse((save_dir / "radar" / "MAE_RMSE.png").exists())

    def test_empty_validation_group_does_not_write_png(self):
        metrics = sample_metrics()
        for epoch in metrics.values():
            epoch["valid"] = {"loss": 1.0, "w10": {"mse": 1.0}}

        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp) / "TPro"
            stderr = StringIO()
            with redirect_stderr(stderr):
                plot_metrics.plot_metrics(metrics, str(save_dir))
            self.assertFalse((save_dir / "w10" / "MSE.png").exists())
            self.assertFalse((save_dir / "w10" / "MAE_RMSE.png").exists())
            self.assertIn("WARNING", stderr.getvalue())

    def test_malicious_validation_group_name_cannot_write_outside_tpro(self):
        metrics = sample_metrics()
        for epoch in metrics.values():
            epoch["valid"] = {"loss": 1.0, "../outside": {"csi": [0.5]}}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_dir = tmp_path / "TPro"
            stderr = StringIO()
            with redirect_stderr(stderr):
                plot_metrics.plot_metrics(metrics, str(save_dir))
            self.assertIn("WARNING", stderr.getvalue())
            self.assertFalse((tmp_path / "outside").exists())

    def test_malicious_list_metric_name_cannot_write_outside_group_dir(self):
        metrics = sample_metrics()
        for epoch in metrics.values():
            epoch["valid"] = {"loss": 1.0, "w10": {"../outside": [0.5]}}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_dir = tmp_path / "TPro"
            stderr = StringIO()
            with redirect_stderr(stderr):
                plot_metrics.plot_metrics(metrics, str(save_dir))
            self.assertIn("WARNING", stderr.getvalue())
            self.assertFalse((save_dir / "outside").exists())
            self.assertFalse((tmp_path / "outside").exists())

    def test_malicious_scalar_metric_name_cannot_write_outside_group_dir(self):
        metrics = sample_metrics()
        for epoch in metrics.values():
            epoch["valid"] = {"loss": 1.0, "w10": {"../outside": 0.5}}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_dir = tmp_path / "TPro"
            stderr = StringIO()
            with redirect_stderr(stderr):
                plot_metrics.plot_metrics(metrics, str(save_dir))
            self.assertIn("WARNING", stderr.getvalue())
            self.assertFalse((save_dir / "outside").exists())
            self.assertFalse((tmp_path / "outside").exists())

    def test_cli_generates_expected_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work_dirs"
            self.write_metrics(work_dir, "drift_w3s", sample_metrics())

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "utils" / "plot_metrics.py"),
                    "-wd",
                    str(work_dir),
                    "-ex",
                    "drift_w3s",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            save_dir = work_dir / "drift_w3s" / "TPro"
            self.assert_png(save_dir / "Loss.png")
            self.assert_png(save_dir / "LR.png")
            self.assert_png(save_dir / "w10" / "MAE_RMSE.png")
            self.assert_png(save_dir / "w10" / "CSI.png")
            non_png = [path for path in save_dir.rglob("*") if path.is_file() and path.suffix != ".png"]
            self.assertEqual(non_png, [])

    def test_cli_missing_project_warns_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "utils" / "plot_metrics.py"),
                    "-wd",
                    str(Path(tmp) / "work_dirs"),
                    "-ex",
                    "missing_run",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("WARNING", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_cli_rejects_path_traversal_without_writing_outside_work_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            work_dir = tmp_path / "work_dirs"
            outside_dir = tmp_path / "outside"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "utils" / "plot_metrics.py"),
                    "-wd",
                    str(work_dir),
                    "-ex",
                    "../outside",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("WARNING", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((outside_dir / "TPro").exists())


if __name__ == "__main__":
    unittest.main()
