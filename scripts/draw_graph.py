import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _validate_data(data: Dict[str, Dict[str, float]], chart_name: str) -> None:
    missing = []
    for dataset, model_map in data.items():
        for model, value in model_map.items():
            if value is None:
                missing.append(f"{chart_name} -> {dataset} / {model}")
    if missing:
        msg = "\n".join(missing)
        raise ValueError(f"Please fill all data values. Missing:\n{msg}")


def _plot_grouped_bar(
    data: Dict[str, Dict[str, float]],
    models: List[str],
    metric_name: str,
    title: str,
    output_path: str,
) -> None:
    datasets = list(data.keys())
    x = np.arange(len(datasets))
    group_width = 0.8
    colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
    color_map = {
        model: colors[i % len(colors)] if colors else None
        for i, model in enumerate(models)
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    handles = {}
    for di, dataset in enumerate(datasets):
        present_models = [m for m in models if m in data[dataset]]
        if not present_models:
            continue
        bar_width = group_width / len(present_models)
        left = x[di] - group_width / 2 + bar_width / 2
        for mi, model in enumerate(present_models):
            value = data[dataset][model]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            bar = ax.bar(
                left + mi * bar_width,
                value,
                bar_width,
                label=model,
                color=color_map.get(model),
            )
            if model not in handles:
                handles[model] = bar[0]

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    if handles:
        ordered_models = [m for m in models if m in handles]
        ordered_handles = [handles[m] for m in ordered_models]
        ax.legend(
            ordered_handles,
            ordered_models,
            frameon=False,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _collect_models(data: Dict[str, Dict[str, float]], preferred: List[str]) -> List[str]:
    model_set = set()
    for model_map in data.values():
        model_set.update(model_map.keys())
    ordered = [m for m in preferred if m in model_set]
    remaining = sorted(model_set - set(ordered))
    return ordered + remaining


def _plot_time_with_improvement(
    data: Dict[str, Dict[str, float]],
    baseline_model: str,
    target_model: str,
    extra_models: Optional[List[str]],
    metric_name: str,
    title: str,
    output_path: str,
) -> None:
    models = [baseline_model]
    if extra_models:
        models.extend(extra_models)
    if target_model not in models:
        models.append(target_model)

    datasets = list(data.keys())
    values = np.array([[data[d][m] for m in models] for d in datasets], dtype=float)

    x = np.arange(len(datasets))
    bar_width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, model in enumerate(models):
        ax.bar(x + i * bar_width, values[:, i], bar_width, label=model)

    ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    # Fill your experiment data below (replace None with numeric values).
    acc_data = {
        "RGB": {
            "MS_GraphRAG": 75.67,
            "HippoRAG": 80.67,
            "HippoRAG 2": 75.00,
            "EraRAG": 73.33,
            "CacheGraphRAG": 94.00,
        },
        "2WikiMultiHopQA": {
            "MS_GraphRAG": 36.50,
            "HippoRAG": 55.50,
            "HippoRAG 2": 45.50,
            "EraRAG": 49.00,
            "CacheGraphRAG": 73.17,
        },
        "SpecificQA":{
            "MS_GraphRAG": 57.83,
            "HippoRAG 2": 50.00,
            "EraRAG": 55.00,
            "CacheGraphRAG": 72.50,
        }
    }

    time_data = {
        "RGB": {
            "MS_GraphRAG": 25693.87 ,
            "HippoRAG": 24052.18,
            "CacheGraphRAG": 4811.89,
        },
        "2WikiMultiHopQA": {
            "MS_GraphRAG": 7182.71,
            "HippoRAG": 9413.42,
            "CacheGraphRAG": 2908.57,
        },
    }

    _validate_data(acc_data, "ACC")
    _validate_data(time_data, "Time")

    _plot_grouped_bar(
        data=acc_data,
        models=_collect_models(
            acc_data,
            preferred=["MS_GraphRAG", "HippoRAG", "HippoRAG 2", "EraRAG", "CacheGraphRAG"],
        ),
        metric_name="ACC",
        title="ACC Comparison",
        output_path="figures/acc_comparison.png",
    )

    _plot_time_with_improvement(
        data=time_data,
        baseline_model="MS_GraphRAG",
        target_model="CacheGraphRAG",
        extra_models=["HippoRAG"],
        metric_name="Time",
        title="Time Comparison",
        output_path="figures/time_comparison.png",
    )


if __name__ == "__main__":
    main()
