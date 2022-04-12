from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import total_ordering
from typing import Optional

import numpy as np


@total_ordering
class Angle:
    def __init__(
        self, *, radians: Optional[float] = None, degrees: Optional[float] = None
    ):
        if radians is None and degrees is None:
            raise ValueError
        elif radians is not None and degrees is not None:
            raise ValueError
        else:
            self._radians = radians
            self._degrees = degrees
        self.normalize()

    @property
    def radians(self) -> float:
        if self._radians is not None:
            return self._radians
        else:
            return np.deg2rad(self._degrees)

    @radians.setter
    def radians(self, radians: float):
        self._radians = radians
        self._degrees = None

    @property
    def degrees(self) -> float:
        if self._degrees is not None:
            return self._degrees
        else:
            return np.rad2deg(self._radians)

    @degrees.setter
    def degrees(self, degrees: float):
        self._degrees = degrees
        self._radians = None

    def __repr__(self):
        return f"{self.degrees}°"

    def __eq__(self, other) -> bool:
        if self._radians is not None:
            return self.radians == other.radians
        else:
            return self.degrees == other.degrees

    def __lt__(self, other) -> bool:
        if self._radians is not None:
            return self.radians < other.radians
        else:
            return self.degrees < other.degrees

    def __add__(self, other) -> Angle:
        if self._radians is not None:
            return Angle(radians=self.radians + other.radians)
        elif self._degrees is not None:
            return Angle(degrees=self.degrees + other.degrees)
        else:
            raise NotImplementedError

    def normalize(self):
        if self._radians is not None:
            self._radians = np.mod(self._radians, 2 * np.pi)
        elif self._degrees is not None:
            self._degrees = np.mod(self._degrees, 360)
        else:
            raise NotImplementedError


@dataclass
class Point:
    x: int
    y: int


@dataclass(order=True)
class Line:
    strength: int = dataclasses.field(compare=True)
    angle: Angle = dataclasses.field(compare=True)
    distance: float = dataclasses.field(compare=True)


def intersect(first: Line, second: Line, /) -> Point:
    matrix = np.array(
        [
            [np.cos(first.angle.radians), np.sin(first.angle.radians)],
            [np.cos(second.angle.radians), np.sin(second.angle.radians)],
        ]
    )
    vector = np.array([[first.distance], [second.distance]])
    return Point(*np.linalg.solve(matrix, vector))
