from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
GITHUB_REPO = SCRIPT_DIR.parent
EXTERNAL_DATA = Path(
    os.environ.get(
        "PEROVSKITE_EXTERNAL_DATA",
        GITHUB_REPO / "data",
    )
)
RESULTS_DIR = Path(
    os.environ.get(
        "PEROVSKITE_RESULTS_DIR",
        EXTERNAL_DATA / "results" / "optimization",
    )
)
FIGURE_DATA_DIR = EXTERNAL_DATA / "source_data" / "figure_sources"
PROPERTY_DIR = EXTERNAL_DATA / "source_data" / "property_models"
MODEL_DATA_DIR = PROPERTY_DIR
OUTPUT_DIR = SCRIPT_DIR / "output_png"


def configure_style(font_size: float = 14) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": font_size,
            "axes.linewidth": 1.8,
            "axes.edgecolor": "black",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": True,
            "savefig.facecolor": "white",
        }
    )


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def save_png(fig: plt.Figure, name: str, dpi: int = 300) -> Path:
    path = output_path(name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def load_run(name: str) -> np.lib.npyio.NpzFile:
    path = RESULTS_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing optimization results: {path}. Install the public result/source-data release."
        )
    return np.load(path, allow_pickle=True)


def validate_png(path: Path, min_width: int = 700, min_height: int = 450) -> None:
    with Image.open(path) as image:
        if image.width < min_width or image.height < min_height:
            raise RuntimeError(f"Unexpectedly small output: {path} {image.size}")
        print(f"Validated {path.name}: {image.size}")


def calculate_structural_diversity(smiles_by_round: np.ndarray) -> list[float]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    values: list[float] = []
    for smiles_group in smiles_by_round:
        fingerprints = []
        for smiles in smiles_group:
            molecule = Chem.MolFromSmiles(str(smiles))
            if molecule is not None:
                fingerprints.append(generator.GetFingerprint(molecule))
        total_distance = 0.0
        pair_count = 0
        for index in range(1, len(fingerprints)):
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprints[index], fingerprints[:index]
            )
            total_distance += sum(1.0 - value for value in similarities)
            pair_count += len(similarities)
        values.append(total_distance / pair_count)
    return values


def plot_property_evolution(output_name: str) -> Path:
    configure_style(18)
    data = load_run("1e-5_5")["sample_properties"]
    colors = mpl.colormaps["viridis"].resampled(len(data))
    pooled = np.vstack(data)
    fig, axes = plt.subplots(5, 2, figsize=(11, 19.2), sharex=True, sharey=True)
    for round_index, ax in enumerate(axes.flat):
        ax.scatter(
            pooled[:, 0],
            pooled[:, 1],
            color="lightgray",
            alpha=0.52,
            s=11,
            edgecolors="none",
        )
        current = data[round_index]
        ax.scatter(
            current[:, 0],
            current[:, 1],
            color=colors(round_index),
            alpha=0.85,
            s=12,
            edgecolors="none",
        )
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 25)
        ax.set_xlabel("DN (kcal/mol)")
        ax.set_ylabel("DM (Debye)")
        ax.text(0.02, 0.95, f"Round {round_index + 1}", transform=ax.transAxes, va="top")
    fig.tight_layout(h_pad=2.1, w_pad=2.2)
    return save_png(fig, output_name)


def _marginal_panel(
    figure: plt.Figure,
    spec,
    values: np.ndarray,
    k_label: str,
    round_number: int,
) -> None:
    nested = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=spec,
        width_ratios=[1.0, 0.17],
        height_ratios=[0.17, 1.0],
        wspace=0.04,
        hspace=0.04,
    )
    hist_x = figure.add_subplot(nested[0, 0])
    main_ax = figure.add_subplot(nested[1, 0])
    hist_y = figure.add_subplot(nested[1, 1])
    x_values = values[:, 0]
    y_values = values[:, 1]
    main_ax.scatter(
        x_values,
        y_values,
        color="green",
        alpha=0.58,
        edgecolors="black",
        linewidths=0.28,
        s=13,
    )
    main_ax.set_xlim(0, 60)
    main_ax.set_ylim(0, 25)
    main_ax.set_xticks(np.arange(0, 61, 10))
    main_ax.set_yticks(np.arange(0, 26, 5))
    main_ax.set_xlabel("DN (kcal/mol)")
    main_ax.set_ylabel("DM (Debye)")
    main_ax.grid(True, linestyle="--", alpha=0.48)
    main_ax.text(
        0.12,
        0.86,
        f"k={k_label}, round={round_number}",
        transform=main_ax.transAxes,
        bbox={"facecolor": "white", "edgecolor": "0.4", "alpha": 0.9},
    )
    hist_x.hist(x_values, bins=30, color="0.65")
    hist_x.set_xlim(0, 60)
    hist_x.axis("off")
    hist_y.hist(y_values, bins=30, orientation="horizontal", color="0.30")
    hist_y.set_ylim(0, 25)
    hist_y.axis("off")


def plot_pareto_grid(output_name: str) -> Path:
    configure_style(13)
    runs = [
        ("1e-1", load_run("1e-1_5")["sample_properties"]),
        ("1e-3", load_run("1e-3_5")["sample_properties"]),
        ("1e-5", load_run("1e-5_5")["sample_properties"]),
    ]
    selected_rounds = [1, 5, 10]
    fig = plt.figure(figsize=(12.7, 12.9))
    outer = fig.add_gridspec(
        3,
        3,
        left=0.06,
        right=0.985,
        bottom=0.05,
        top=0.98,
        wspace=0.26,
        hspace=0.27,
    )
    for row_index, (k_label, data) in enumerate(runs):
        fig.text(
            0.012,
            0.955 - row_index * 0.331,
            chr(ord("a") + row_index),
            fontsize=27,
            fontweight="bold",
        )
        for column_index, round_number in enumerate(selected_rounds):
            _marginal_panel(
                fig,
                outer[row_index, column_index],
                data[round_number - 1],
                k_label,
                round_number,
            )
    return save_png(fig, output_name)
