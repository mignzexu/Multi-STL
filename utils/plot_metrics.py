# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import argparse
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


Metrics = dict[str, Any]
EpochItems = list[tuple[int, Metrics]]
Series = tuple[list[int], list[float]]
PlotSeries = list[tuple[str, list[int], list[float]]]


def create_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-STL: 训练指标绘图工具")
    _ = parser.add_argument("--ex_name", "-ex", type=str, required=True, help="实验项目名称")
    _ = parser.add_argument("--work_dir", "-wd", type=str, default="work_dirs", help="工作目录")
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def load_pyplot() -> Any:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    return importlib.import_module("matplotlib.pyplot")


def run_paths(work_dir: str, ex_name: str) -> tuple[str, str, str]:
    if os.path.isabs(ex_name) or ".." in Path(ex_name).parts:
        raise ValueError(f"实验项目名称不能是绝对路径或包含 '..': {ex_name}")

    work_root = Path(work_dir).resolve()
    run_path = (work_root / ex_name).resolve()
    if os.path.commonpath([str(work_root), str(run_path)]) != str(work_root):
        raise ValueError(f"实验项目路径超出工作目录: {ex_name}")

    run_dir = os.path.join(work_dir, ex_name)
    return run_dir, os.path.join(run_dir, "metrics.json"), os.path.join(run_dir, "TPro")


def safe_child_path(root: str, *parts: str) -> str:
    for part in parts:
        if os.path.isabs(part) or ".." in Path(part).parts:
            raise ValueError(f"输出路径片段不能是绝对路径或包含 '..': {part}")

    root_path = Path(root).resolve()
    child_path = root_path.joinpath(*parts).resolve()
    if os.path.commonpath([str(root_path), str(child_path)]) != str(root_path):
        raise ValueError(f"输出路径超出目标目录: {child_path}")
    return str(child_path)


def load_metrics(metrics_path: str) -> Metrics | None:
    if not os.path.exists(metrics_path):
        warn(f"metrics.json 不存在: {metrics_path}")
        return None

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except json.JSONDecodeError as exc:
        warn(f"metrics.json 格式错误: {metrics_path} ({exc})")
        return None
    except OSError as exc:
        warn(f"无法读取 metrics.json: {metrics_path} ({exc})")
        return None

    if not isinstance(metrics, dict) or not metrics:
        warn(f"metrics.json 为空或不是字典: {metrics_path}")
        return None

    return metrics


def sorted_epoch_items(metrics: Metrics) -> EpochItems:
    epoch_items: EpochItems = []
    for epoch_key, epoch_value in metrics.items():
        try:
            epoch = int(epoch_key)
        except (TypeError, ValueError):
            warn(f"跳过非数字 epoch: {epoch_key}")
            continue

        if not isinstance(epoch_value, dict):
            warn(f"跳过格式错误的 epoch {epoch_key}: 记录不是字典")
            continue

        epoch_items.append((epoch, epoch_value))

    epoch_items.sort(key=lambda item: item[0])
    if not epoch_items:
        warn("没有可用的 epoch 记录")
    return epoch_items


def _value_at_path(epoch_value: Metrics, path: tuple[str, ...]) -> Any | None:
    current: Any = epoch_value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def scalar_series(epoch_items: EpochItems, path: tuple[str, ...], label: str) -> Series:
    xs: list[int] = []
    ys: list[float] = []
    for epoch, epoch_value in epoch_items:
        value = _value_at_path(epoch_value, path)
        if value is None:
            warn(f"epoch {epoch} 缺少指标 {label}")
            continue
        if not is_number(value):
            warn(f"epoch {epoch} 指标 {label} 不是数字，已跳过")
            continue
        xs.append(epoch)
        ys.append(float(value))
    return xs, ys


def validation_group_names(epoch_items: EpochItems) -> list[str]:
    names: set[str] = set()
    for _, epoch_value in epoch_items:
        valid = epoch_value.get("valid")
        if not isinstance(valid, dict):
            continue
        for name, value in valid.items():
            if name != "loss" and isinstance(value, dict):
                names.add(name)
    return sorted(names)


def group_scalar_series(epoch_items: EpochItems, group_name: str, metric_name: str) -> Series:
    return scalar_series(epoch_items, ("valid", group_name, metric_name), f"valid.{group_name}.{metric_name}")


def group_list_series(epoch_items: EpochItems, group_name: str, metric_name: str) -> dict[int, Series]:
    series: defaultdict[int, Series] = defaultdict(lambda: ([], []))
    for epoch, epoch_value in epoch_items:
        value = _value_at_path(epoch_value, ("valid", group_name, metric_name))
        if value is None:
            warn(f"epoch {epoch} 缺少指标 valid.{group_name}.{metric_name}")
            continue
        if not isinstance(value, list):
            warn(f"epoch {epoch} 指标 valid.{group_name}.{metric_name} 不是列表，已跳过")
            continue
        for index, item in enumerate(value):
            if not is_number(item):
                warn(f"epoch {epoch} 指标 valid.{group_name}.{metric_name}[{index}] 不是数字，已跳过")
                continue
            xs, ys = series[index]
            xs.append(epoch)
            ys.append(float(item))
    return dict(series)


def list_metric_names(epoch_items: EpochItems, group_name: str) -> list[str]:
    names: set[str] = set()
    for _, epoch_value in epoch_items:
        group = _value_at_path(epoch_value, ("valid", group_name))
        if not isinstance(group, dict):
            continue
        for name, value in group.items():
            if isinstance(value, list):
                names.add(name)
    return sorted(names)


def scalar_metric_names(epoch_items: EpochItems, group_name: str) -> list[str]:
    names: set[str] = set()
    for _, epoch_value in epoch_items:
        group = _value_at_path(epoch_value, ("valid", group_name))
        if not isinstance(group, dict):
            continue
        for name, value in group.items():
            if name == "mse":
                continue
            if is_number(value):
                names.add(name)
    return sorted(names)


def save_line_plot(series: PlotSeries, title: str, ylabel: str, save_path: str) -> bool:
    available = [(label, xs, ys) for label, xs, ys in series if xs and ys]
    if not available:
        warn(f"没有可绘制数据: {title}")
        return False

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt = load_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, xs, ys in available:
        ax.plot(xs, ys, marker="o", markersize=1, linewidth=0.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    if len(available) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_loss(epoch_items: EpochItems, save_dir: str) -> bool:
    train_loss = scalar_series(epoch_items, ("train", "loss"), "train.loss")
    valid_loss = scalar_series(epoch_items, ("valid", "loss"), "valid.loss")
    return save_line_plot(
        [("train loss", train_loss[0], train_loss[1]), ("valid loss", valid_loss[0], valid_loss[1])],
        "Loss",
        "Loss",
        os.path.join(save_dir, "Loss.png"),
    )


def plot_lr(epoch_items: EpochItems, save_dir: str) -> bool:
    lr = scalar_series(epoch_items, ("train", "lr"), "train.lr")
    return save_line_plot([("lr", lr[0], lr[1])], "LR", "LR", os.path.join(save_dir, "LR.png"))


def plot_validation_group(epoch_items: EpochItems, save_dir: str, group_name: str) -> bool:
    try:
        group_dir = safe_child_path(save_dir, group_name)
    except ValueError as exc:
        warn(str(exc))
        return False
    wrote_any = False
    scalar_names = scalar_metric_names(epoch_items, group_name)

    if "mae" in scalar_names or "rmse" in scalar_names:
        mae = group_scalar_series(epoch_items, group_name, "mae")
        rmse = group_scalar_series(epoch_items, group_name, "rmse")
        wrote_any = save_line_plot(
            [("mae", mae[0], mae[1]), ("rmse", rmse[0], rmse[1])],
            "MAE_RMSE",
            "Value",
            safe_child_path(group_dir, "MAE_RMSE.png"),
        ) or wrote_any

    for metric_name in scalar_names:
        if metric_name in {"mae", "rmse"}:
            continue
        try:
            metric_path = safe_child_path(group_dir, f"{metric_name.upper()}.png")
        except ValueError as exc:
            warn(str(exc))
            continue
        xs, ys = group_scalar_series(epoch_items, group_name, metric_name)
        wrote_any = save_line_plot(
            [(metric_name, xs, ys)],
            metric_name.upper(),
            metric_name.upper(),
            metric_path,
        ) or wrote_any

    for metric_name in list_metric_names(epoch_items, group_name):
        try:
            metric_path = safe_child_path(group_dir, f"{metric_name.upper()}.png")
        except ValueError as exc:
            warn(str(exc))
            continue
        series = group_list_series(epoch_items, group_name, metric_name)
        plot_series: PlotSeries = []
        for index in sorted(series):
            xs, ys = series[index]
            plot_series.append((f"{metric_name.upper()}[{index}]", xs, ys))
        wrote_any = save_line_plot(
            plot_series,
            metric_name.upper(),
            metric_name.upper(),
            metric_path,
        ) or wrote_any

    if not wrote_any:
        warn(f"验证指标组 {group_name} 没有可绘制数据")
    return wrote_any


def plot_metrics(metrics: Metrics, save_dir: str) -> bool:
    epoch_items = sorted_epoch_items(metrics)
    if not epoch_items:
        return False

    os.makedirs(save_dir, exist_ok=True)
    wrote_any = plot_loss(epoch_items, save_dir)
    wrote_any = plot_lr(epoch_items, save_dir) or wrote_any
    for group_name in validation_group_names(epoch_items):
        wrote_any = plot_validation_group(epoch_items, save_dir, group_name) or wrote_any
    return wrote_any


def main() -> int:
    args = create_args()
    try:
        _, metrics_path, save_dir = run_paths(args.work_dir, args.ex_name)
    except ValueError as exc:
        warn(str(exc))
        return 0
    metrics = load_metrics(metrics_path)
    if metrics is None:
        return 0
    _ = plot_metrics(metrics, save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
