"""
Vehicle and VehicleType models for the delivery fleet system.

This module defines the core vehicle entities used in route planning and delivery operations.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VehicleType:
    """
    Defines a category of vehicle with its operational characteristics.

    Attributes:
        name: Human-readable vehicle type name
        capacity_m3: Maximum cargo volume in cubic meters
        cost_per_km: Operating cost per kilometer traveled
        purchase_price: Initial acquisition cost
        max_range_km: Maximum distance the vehicle can travel in one day
    """
    name: str
    capacity_m3: float
    cost_per_km: float
    purchase_price: float
    max_range_km: float

    def __str__(self) -> str:
        return (f"{self.name} (Capacity: {self.capacity_m3}m³, "
                f"Cost: ${self.cost_per_km}/km, Price: ${self.purchase_price})")


@dataclass
class Vehicle:
    """
    Represents an individual vehicle in the fleet.

    Attributes:
        id: Unique vehicle identifier
        vehicle_type: Reference to VehicleType defining this vehicle's specs
        current_location: Current position as (x, y) coordinates
        purchase_day: Game day when vehicle was acquired
    """
    id: str
    vehicle_type: VehicleType
    current_location: Tuple[float, float] = (0.0, 0.0)  # Default to depot
    purchase_day: int = 0

    def can_carry(self, volume: float) -> bool:
        """
        Check if vehicle has capacity for given volume.

        Args:
            volume: Volume in cubic meters to check

        Returns:
            True if volume fits within vehicle capacity
        """
        return volume <= self.vehicle_type.capacity_m3

    def calculate_trip_cost(self, distance_km: float) -> float:
        """
        Calculate operating cost for a given distance.

        Args:
            distance_km: Distance to travel in kilometers

        Returns:
            Total cost for the trip
        """
        return distance_km * self.vehicle_type.cost_per_km

    def can_travel_distance(self, distance_km: float) -> bool:
        """
        Check if vehicle can travel the given distance within its range.

        Args:
            distance_km: Distance to check

        Returns:
            True if distance is within vehicle's max range
        """
        return distance_km <= self.vehicle_type.max_range_km

    def __str__(self) -> str:
        return f"Vehicle {self.id} ({self.vehicle_type.name}) at {self.current_location}"

    def __repr__(self) -> str:
        return f"Vehicle(id='{self.id}', type='{self.vehicle_type.name}')"
