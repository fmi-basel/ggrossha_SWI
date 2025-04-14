import argparse

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from faim_ipa.utils import get_git_root, create_logger
from matplotlib import pyplot as plt
from rich.pretty import pretty_repr

sys.path.append(str(get_git_root()))
from source.s02_segment.config import SegmentationConfig


def main(config: SegmentationConfig, measurement_files: list[Path]):
    logger = create_logger("qc")
    logger.info("Config:")
    logger.info(pretty_repr(config))
    logger.info("Measurement files:")
    logger.info(pretty_repr(measurement_files))

    output_dir = config.output_dir.parent / "s03_measurements" / "quality-control"
    output_dir.mkdir(parents=True, exist_ok=True)
    create_qc_plots(output_dir, measurement_file)
    logger.info("Done.")


def create_qc_plots(output_dir: Path, csv_files: list[Path]):
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "sans-serif",
        }
    )
    dfs = []
    for f in csv_files:
        m = pd.read_csv(
            f,
            index_col=None,
            dtype={
                "channel": np.int16,
                "time": np.int16,
                "length": np.float32,
                "area": np.float32,
                "mean_intensity": np.float32,
                "percentile.05_intensity": np.float32,
                "focus_slice_start": np.int16,
                "focus_slice_end": np.int16,
                "id": str,
            },
        )
        dfs.append((m, f.name))

    all_results = pd.concat([df for df, _ in dfs])

    create_segmentation_quality_overview(output_dir, dfs)

    vmin = all_results["percentile.05_intensity"].min() - 50
    vmax = (
        np.quantile(
            np.nan_to_num(all_results["mean_intensity"], nan=vmin),
            0.99,
        )
        + 50
    )
    xlim = [int(all_results["time"].min()), int(all_results["time"].max())]
    for df, file_name in dfs:
        create_measurements_overview(output_dir, file_name, df, vmin, vmax, xlim)


def create_segmentation_quality_overview(
    output_dir: Path,
    dfs: list[tuple[pd.DataFrame, str]],
):
    missing_segmentations = []
    for df, _ in dfs:
        missing_segmentations.append(
            (df.iloc[0]["id"], df.isnull().sum()["focus_plane"], len(df))
        )

    missing_segmentations = pd.DataFrame(
        missing_segmentations, columns=["id", "n_missing", "n_time_points"]
    ).sort_values("id", ascending=True)
    bad_frames_perc = (
        missing_segmentations["n_missing"]
        * 100
        / missing_segmentations["n_time_points"]
    )
    color = ["blue" if v < 10 else "red" for v in bad_frames_perc]

    plt.figure(figsize=(12, 4))
    plt.bar(missing_segmentations["id"], bad_frames_perc, color=color)
    x_start, x_end = (
        int(missing_segmentations["id"].min()),
        int(missing_segmentations["id"].max() + 1),
    )
    plt.plot([x_start - 0.5, x_end + 0.5], [10, 10], "--", color="red")
    plt.xticks(range(x_start, x_end, 5))
    plt.yticks(range(0, 101, 10))
    plt.ylim([0, 100])
    plt.xlim([x_start - 0.5, x_end + 0.5])
    plt.xlabel("Worm ID")
    plt.ylabel("Missing Segmentation [%]")
    plt.suptitle("Segmentation Quality Overview")
    plt.tight_layout()
    plt.savefig(output_dir / "segmentation-quality-overview.svg")
    plt.close()


def create_measurements_overview(
    output_dir: Path,
    file_name: str,
    df: pd.DataFrame,
    vmin: float,
    vmax: float,
    xlim: list[int],
):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 12), nrows=3, ncols=1)

    # Length & Area
    miss_perc = np.round(df["length"].isnull().sum() / (df["time"].max() + 1) * 100, 1)
    ax1_1 = ax1.twinx()
    ax1.plot(df["time"], df["length"], ".", color="darkgoldenrod")
    ax1.plot(df["time"], df["length"], "-", alpha=0.2, color="darkgoldenrod")
    ax1.set_ylim([0, 1500])
    ax1.set_ylabel("Length", color="darkgoldenrod")
    ax1.tick_params(axis="y", colors="darkgoldenrod")
    ax1.spines["left"].set_color("darkgoldenrod")
    ax1.set_xlim(xlim)

    ax1.bar(
        df["time"],
        df["length"].isnull() * 1500,
        color="dimgray",
        label=f"Missing Segmentation ({miss_perc}%)",
    )
    ax1.legend(loc="upper left")

    ax1_1.plot(df["time"], df["area"], ".", color="steelblue")
    ax1_1.plot(df["time"], df["area"], "-", alpha=0.2, color="steelblue")
    ax1_1.set_ylim([1000, 100000])
    ax1_1.set_ylabel("Area", color="steelblue")
    ax1_1.tick_params(axis="y", colors="steelblue")
    ax1_1.spines["right"].set_color("steelblue")
    ax1_1.spines["left"].set_color("darkgoldenrod")
    ax1_1.set_xlim(xlim)

    # Intensity
    ax2_1 = ax2.twinx()
    ax2.plot(df["time"], df["percentile.05_intensity"], ".", color="black")
    ax2.plot(df["time"], df["percentile.05_intensity"], "-", alpha=0.2, color="black")
    ax2.set_ylim([vmin, vmax])
    ax2.set_ylabel("5th Percentile Intensity", color="black")
    ax2.bar(df["time"], df["focus_plane"].isnull() * vmax, color="dimgray")
    ax2.set_xlim(xlim)

    ax2_1.plot(df["time"], df["mean_intensity"], ".", color="darkgreen")
    ax2_1.plot(df["time"], df["mean_intensity"], "-", alpha=0.2, color="darkgreen")
    ax2_1.set_ylim([vmin, vmax])
    ax2_1.set_ylabel("Mean Intensity", color="darkgreen")
    ax2_1.tick_params(axis="y", colors="darkgreen")
    ax2_1.spines["right"].set_color("darkgreen")
    ax2_1.set_xlim(xlim)

    # Focus plane
    ax3.plot(df["time"], df["focus_slice_start"], color="black")
    ax3.plot(df["time"], df["focus_slice_end"], color="black")
    ax3.fill_between(
        df["time"],
        df["focus_slice_start"],
        df["focus_slice_end"],
        color="gray",
        alpha=0.3,
        label="Focus Slice Range",
    )
    ax3.legend(loc="lower left")
    ax3.set_ylim([0, 25])
    ax3.set_ylabel("Planes used for MIP", color="black")

    ax3.bar(df["time"], df["length"].isnull() * 25, color="dimgray")

    ax3.set_xlabel("Timepoint")
    ax3.set_xlim(xlim)

    fig.suptitle(f"Quality Control\nworm_{df['id'].iloc[0]}")

    fig.tight_layout()
    name = file_name.replace(".csv", "-QC.svg")
    fig.savefig(output_dir / name)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
    )
    parser.add_argument("--measurement_files", nargs="+")

    args = parser.parse_args()

    config = SegmentationConfig.load(Path(args.config))
    measurement_files = []
    for measurement_file in args.measurement_files:
        with open(measurement_file, "r") as f:
            msf = [Path(m) for m in yaml.safe_load(f)]
            measurement_files.extend(msf)

    main(config, measurement_files)
