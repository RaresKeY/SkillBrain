"""
Map and location models for the delivery system.

This module defines the spatial representation of the delivery area.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


@dataclass
class Location:
    """
    Named location on the map.

    Attributes:
        id: Unique location identifier
        name: Human-readable location name
        x: X coordinate in kilometers
        y: Y coordinate in kilometers
    """
    id: str
    name: str
    x: float
    y: float

    def to_coordinates(self) -> Tuple[float, float]:
        """Convert location to coordinate tuple."""
        return (self.x, self.y)

    def __str__(self) -> str:
        return f"{self.name} ({self.x}, {self.y})"


class DeliveryMap:
    """
    Represents the 2D delivery area with distance calculations.

    The map uses a Cartesian coordinate system with the depot at origin (0, 0).
    All distances are in kilometers.
    """

    def __init__(self, width: float, height: float):
        """
        Initialize delivery map.

        Args:
            width: Map width in kilometers
            height: Map height in kilometers
        """
        self.width = width
        self.height = height
        self.depot: Tuple[float, float] = (0.0, 0.0)
        self.locations: Dict[str, Location] = {}

    def add_location(self, location: Location) -> None:
        """
        Add a named location to the map.

        Args:
            location: Location object to add
        """
        self.locations[location.id] = location

    def get_location(self, location_id: str) -> Optional[Location]:
        """
        Retrieve a location by its ID.

        Args:
            location_id: Unique identifier of the location

        Returns:
            Location object if found, None otherwise
        """
        return self.locations.get(location_id)

    def distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points.

        This is the straight-line distance, appropriate for direct travel.

        Args:
            p1: First point as (x, y) tuple
            p2: Second point as (x, y) tuple

        Returns:
            Distance in kilometers

        Example:
            >>> map = DeliveryMap(100, 100)
            >>> map.distance((0, 0), (3, 4))
            5.0
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)

    def manhattan_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Calculate Manhattan (taxicab) distance between two points.

        This represents grid-based travel where only horizontal and vertical
        movement is allowed (like city blocks).

        Args:
            p1: First point as (x, y) tuple
            p2: Second point as (x, y) tuple

        Returns:
            Manhattan distance in kilometers

        Example:
            >>> map = DeliveryMap(100, 100)
            >>> map.manhattan_distance((0, 0), (3, 4))
            7.0
        """
        return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])

    def distance_from_depot(self, point: Tuple[float, float]) -> float:
        """
        Calculate distance from depot to given point.

        Args:
            point: Destination point as (x, y) tuple

        Returns:
            Distance in kilometers
        """
        return self.distance(self.depot, point)

    def is_within_bounds(self, point: Tuple[float, float]) -> bool:
        """
        Check if point is within map boundaries.

        Args:
            point: Point to check as (x, y) tuple

        Returns:
            True if point is within map bounds
        """
        x, y = point
        return 0 <= x <= self.width and 0 <= y <= self.height

    def __str__(self) -> str:
        return f"DeliveryMap({self.width}km × {self.height}km, {len(self.locations)} locations)"

    def __repr__(self) -> str:
        return f"DeliveryMap(width={self.width}, height={self.height})"
