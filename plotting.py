from pathlib import Path
from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt

from geometry import Line, Point


def plot_lines(image, lines: Iterable[Line], path: Path):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.get_cmap("Greys"))
    ax.autoscale(False)
    for line in lines:
        (x0, y0) = line.distance * np.array(
            [np.cos(line.angle.radians), np.sin(line.angle.radians)]
        )
        ax.axline((x0, y0), slope=np.tan(line.angle.radians + np.pi / 2))
    fig.savefig(path)
    plt.close(fig)


def plot_points(image, points: Iterable[Point], path: Path):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.get_cmap("Greys"), origin="lower")
    ax.autoscale(False)
    ax.invert_yaxis()
    for point in points:
        ax.scatter(point.x, point.y, marker="x", s=15)
    fig.savefig(path)
    plt.close(fig)


def plot_polygons(image, polygons, path: Path, plot_text: bool):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap=plt.get_cmap("Greys"))  # , origin='lower')
    ax.autoscale(False)
    # ax.invert_yaxis()
    for polygon in polygons:
        ax.plot(polygon[:, 0], polygon[:, 1], "-r", linewidth=2)
        n_points = len(polygon)
        for i, point in enumerate(polygon):
            ax.scatter(point[0], point[1], marker="+", color="cyan", s=10)
            if plot_text and i < n_points - 1:
                ax.text(point[0], point[1], f"{i}: {point}", color="cyan")
    fig.savefig(path)
    plt.close(fig)
