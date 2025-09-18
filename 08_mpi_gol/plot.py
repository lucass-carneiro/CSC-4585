import matplotlib.pyplot as plt
from scipy.interpolate import griddata  # Correct import
import pandas as pd
import numpy as np
import os
import sys
import tomllib


def parse_header(file):
    with open(file, "r") as f:
        for line in f:
            if line.startswith("#"):
                header = line
            else:
                break

    return header[1:].strip().split()


def read_tsv(file, read_header=True):
    basename = os.path.basename(file)
    name, _ = os.path.splitext(basename)

    if read_header:
        header = parse_header(file)
    else:
        header = None

    if read_header:
        df = pd.read_csv(
            file,
            sep="\\s+",
            comment="#",
            names=header
        )
    else:
        df = pd.read_csv(
            file,
            sep="\\s+",
            header=None
        )

    return (name, header, df)


def plot_grid(ax, df, step):
    x = df["1:row"].to_numpy()
    y = df["2:col"].to_numpy()
    z = df["3:state"].to_numpy()

    width = x.max() + 1
    height = y.max() + 1

    image = np.zeros((width, height), dtype=int)
    image[x, y] = z

    ax.imshow(image, cmap="viridis", origin="upper")

    ax.set_xticks(np.arange(-0.5, width, 1), minor=False, labels=[])
    ax.set_yticks(np.arange(-0.5, height, 1), minor=False, labels=[])

    ax.grid(
        True,
        which="major",
        color="lightgray",
        linestyle="-",
        linewidth=0.5
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <config-file.toml> <num-mpi-procs>")
        exit()

    with open(sys.argv[1], "rb") as f:
        toml_file = tomllib.load(f)
        iterations = int(toml_file["general"]["generations"])
        ranks = int(sys.argv[2])

    for it in range(0, iterations):
        plt.close("all")
        plt.title(f"Conway's Game of Life - Iteration {it}")

        fig, ax = plt.subplots(1)

        dfs = []

        for rank in range(0, ranks):
            file_in = f"gol_it_{it:08}_rank_{rank:08}.dat"

            name, header, df = read_tsv(file_in)
            dfs.append(df)

        merged_df = pd.concat(dfs)
        sorted_df = merged_df.sort_values("1:row", ascending=True)

        plot_grid(ax, sorted_df, it)

        file_out = f"gol_it_{it:08}.png"
        fig.tight_layout()
        fig.savefig(file_out, dpi=300)
