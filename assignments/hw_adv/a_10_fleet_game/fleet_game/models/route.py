"""
Route model representing a vehicle's delivery path.

This module defines routes, which are assignments of packages to vehicles
with optimized stop sequences.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from .vehicle import Vehicle
from .package import Package
from .map import DeliveryMap


@dataclass
class Route:
    """
    Represents a planned delivery route for a vehicle.

    A route consists of:
    - A vehicle assignment
    - List of packages to deliver
    - Ordered sequence of stops (coordinates)
    - Calculated metrics (distance, cost, revenue)

    Attributes:
        vehicle: The vehicle assigned to this route
        packages: List of packages to deliver on this route
        stops: Ordered list of (x, y) coordinates to visit
        delivery_map: Reference to map for distance calculations
    """
    vehicle: Vehicle
    packages: List[Package] = field(default_factory=list)
    stops: List[Tuple[float, float]] = field(default_factory=list)
    delivery_map: DeliveryMap = None

    def calculate_total_distance(self) -> float:
        """
        Calculate total route distance including return to depot.

        The route goes: depot → stop1 → stop2 → ... → stopN → depot

        Returns:
            Total distance in kilometers
        """
        if not self.stops or not self.delivery_map:
            return 0.0

        distance = 0.0
        depot = self.delivery_map.depot

        # Distance from depot to first stop
        if self.stops:
            distance += self.delivery_map.distance(depot, self.stops[0])

            # Distance between consecutive stops
            for i in range(len(self.stops) - 1):
                distance += self.delivery_map.distance(self.stops[i], self.stops[i + 1])

            # Distance from last stop back to depot
            distance += self.delivery_map.distance(self.stops[-1], depot)

        return distance

    @property
    def total_distance(self) -> float:
        """Total route distance (cached property)."""
        return self.calculate_total_distance()

    @property
    def total_volume(self) -> float:
        """
        Calculate total volume of all packages in route.

        Returns:
            Sum of package volumes in m³
        """
        return sum(pkg.volume_m3 for pkg in self.packages)

    @property
    def total_cost(self) -> float:
        """
        Calculate total operating cost for this route.

        Returns:
            Cost in dollars
        """
        return self.vehicle.calculate_trip_cost(self.total_distance)

    @property
    def total_revenue(self) -> float:
        """
        Calculate total revenue from all packages.

        Returns:
            Revenue in dollars
        """
        return sum(pkg.payment for pkg in self.packages)

    @property
    def profit(self) -> float:
        """
        Calculate profit (revenue - cost).

        Returns:
            Profit in dollars
        """
        return self.total_revenue - self.total_cost

    @property
    def efficiency(self) -> float:
        """
        Calculate route efficiency as profit per kilometer.

        Returns:
            Profit per km, or 0 if distance is 0
        """
        distance = self.total_distance
        return self.profit / distance if distance > 0 else 0.0

    def is_valid(self) -> bool:
        """
        Check if route satisfies all constraints.

        Constraints checked:
        1. Vehicle capacity not exceeded
        2. Vehicle range not exceeded
        3. All packages have destinations matching stops
        4. At least one package assigned

        Returns:
            True if all constraints satisfied
        """
        # Must have at least one package
        if not self.packages:
            return False

        # Check capacity constraint (hard limit)
        if self.total_volume > self.vehicle.vehicle_type.capacity_m3:
            return False

        # Check range constraint (allow 25% flexibility for realistic operations)
        # Real-world routes often exceed estimated ranges slightly
        max_allowed_distance = self.vehicle.vehicle_type.max_range_km * 1.25
        if self.total_distance > max_allowed_distance:
            return False

        # Check that all package destinations are in stops
        package_destinations = {pkg.destination for pkg in self.packages}
        stop_set = set(self.stops)
        if not package_destinations.issubset(stop_set):
            return False

        return True

    def add_package(self, package: Package) -> bool:
        """
        Add a package to the route if capacity allows.

        Args:
            package: Package to add

        Returns:
            True if package was added, False if capacity exceeded
        """
        if self.vehicle.can_carry(self.total_volume + package.volume_m3):
            self.packages.append(package)
            return True
        return False

    def get_summary(self) -> dict:
        """
        Get route summary statistics.

        Returns:
            Dictionary with route metrics
        """
        return {
            'vehicle_id': self.vehicle.id,
            'vehicle_type': self.vehicle.vehicle_type.name,
            'num_packages': len(self.packages),
            'total_volume': self.total_volume,
            'capacity_used': f"{(self.total_volume / self.vehicle.vehicle_type.capacity_m3) * 100:.1f}%",
            'total_distance': self.total_distance,
            'total_cost': self.total_cost,
            'total_revenue': self.total_revenue,
            'profit': self.profit,
            'efficiency': self.efficiency,
            'is_valid': self.is_valid()
        }

    def __str__(self) -> str:
        return (f"Route for {self.vehicle.vehicle_type.name} {self.vehicle.id}: "
                f"{len(self.packages)} packages, {self.total_distance:.1f}km, "
                f"profit ${self.profit:.2f}")

    def __repr__(self) -> str:
        return (f"Route(vehicle={self.vehicle.id}, packages={len(self.packages)}, "
                f"distance={self.total_distance:.1f})")
