"""
Student routing agent - Sweep Algorithm implementation.

The Sweep Algorithm uses polar angle sorting to create geographically
efficient routes. It sweeps around the depot like a radar, grouping
nearby packages together.

Time Complexity: O(n log n) for sorting + O(n²) for TSP = O(n²)
Space Complexity: O(n)

Trade-offs:
+ Good for radially distributed packages
+ Simple to understand and implement
+ Fast execution
+ Creates geographically coherent routes
- Not optimal for clustered distributions
- Doesn't prioritize package value
- Sensitive to depot location
"""

import math
from typing import List, Tuple
from ..base import RouteAgent
from ...models import Package, Vehicle, Route, DeliveryMap
from ...core import Router


class StudentAgent(RouteAgent):
    """
    Sweep Algorithm routing agent.

    Algorithm:
    1. Calculate polar angle for each package from depot
    2. Sort packages by angle (sweep clockwise)
    3. Assign packages to vehicles in sweep order (respecting capacity)
    4. Optimize each route using Nearest Neighbor TSP
    5. Optionally apply 2-opt improvement

    This algorithm works well when packages are distributed around the depot
    in a radial pattern, as it naturally groups nearby destinations together.
    """

    def __init__(self, delivery_map: DeliveryMap, use_2opt: bool = False):
        """
        Initialize student agent.

        Args:
            delivery_map: Map for distance calculations
            use_2opt: Whether to apply 2-opt local search improvement
        """
        super().__init__(delivery_map, "Student Agent (Sweep)")
        self.description = "Sweep algorithm with polar angle sorting"
        self.use_2opt = use_2opt
        self.router = Router()

    def plan_routes(self, packages: List[Package], fleet: List[Vehicle]) -> List[Route]:
        """
        Create routes using sweep algorithm.

        Args:
            packages: List of packages to deliver
            fleet: Available vehicles

        Returns:
            List of routes
        """
        if not self.validate_inputs(packages, fleet):
            return []

        print(f"[{self.name}] Planning routes for {len(packages)} packages with {len(fleet)} vehicles...")

        # Step 1: Calculate polar angles and sort
        print(f"[{self.name}] Step 1: Calculating polar angles...")
        packages_with_angles = self._calculate_polar_angles(packages)

        # Sort by angle (sweep clockwise starting from east)
        packages_with_angles.sort(key=lambda x: x[0])
        sorted_packages = [pkg for angle, pkg in packages_with_angles]

        # Step 2: Assign packages in sweep order
        print(f"[{self.name}] Step 2: Assigning packages to vehicles...")
        routes = self._assign_packages_by_sweep(sorted_packages, fleet)

        # Step 3: Optimize stop order for each route
        print(f"[{self.name}] Step 3: Optimizing route stop orders...")
        for i, route in enumerate(routes):
            route.stops = self._optimize_route_stops(route.packages)
            print(f"[{self.name}]   Route {i+1}: {len(route.packages)} packages, "
                  f"{route.total_distance:.1f}km, "
                  f"{route.total_volume:.1f}/{route.vehicle.vehicle_type.capacity_m3:.1f}m³")

        # Summary
        total_assigned = sum(len(r.packages) for r in routes)
        total_distance = sum(r.total_distance for r in routes)
        print(f"[{self.name}] ✓ Created {len(routes)} routes with {total_assigned}/{len(packages)} packages")
        print(f"[{self.name}] ✓ Total distance: {total_distance:.1f}km")

        return routes

    def _calculate_polar_angles(self, packages: List[Package]) -> List[Tuple[float, Package]]:
        """
        Calculate polar angle for each package relative to depot.

        The angle is measured counterclockwise from the positive x-axis (east).
        Returns angles in range [-π, π].

        Args:
            packages: List of packages

        Returns:
            List of (angle, package) tuples
        """
        packages_with_angles = []

        for pkg in packages:
            dx = pkg.destination[0] - self.delivery_map.depot[0]
            dy = pkg.destination[1] - self.delivery_map.depot[1]

            # Calculate angle using atan2 (handles all quadrants correctly)
            angle = math.atan2(dy, dx)

            packages_with_angles.append((angle, pkg))

        return packages_with_angles

    def _assign_packages_by_sweep(self, sorted_packages: List[Package],
                                   fleet: List[Vehicle]) -> List[Route]:
        """
        Assign packages to vehicles in sweep order, respecting capacity.

        This method fills vehicles one at a time, moving to the next vehicle
        when current vehicle reaches capacity.

        Args:
            sorted_packages: Packages sorted by polar angle
            fleet: Available vehicles

        Returns:
            List of routes
        """
        routes = []
        available_vehicles = fleet.copy()
        current_route = None
        unassigned_packages = []

        for pkg in sorted_packages:
            # Try to add to current route
            if current_route and current_route.add_package(pkg):
                continue

            # Current route is full or doesn't exist, need new vehicle
            if available_vehicles:
                vehicle = available_vehicles.pop(0)
                current_route = Route(
                    vehicle=vehicle,
                    packages=[pkg],
                    stops=[],  # Will be filled in optimization step
                    delivery_map=self.delivery_map
                )
                routes.append(current_route)
            else:
                # No more vehicles available
                unassigned_packages.append(pkg)

        # Report unassigned packages if any
        if unassigned_packages:
            total_unassigned_volume = sum(p.volume_m3 for p in unassigned_packages)
            total_unassigned_value = sum(p.payment for p in unassigned_packages)
            print(f"[{self.name}] ⚠ Warning: {len(unassigned_packages)} packages "
                  f"({total_unassigned_volume:.1f}m³, ${total_unassigned_value:.0f}) "
                  f"could not be assigned - insufficient fleet capacity")

        return routes

    def _optimize_route_stops(self, packages: List[Package]) -> List[Tuple]:
        """
        Optimize the order of stops for a set of packages.

        Uses Nearest Neighbor TSP heuristic starting from depot,
        optionally followed by 2-opt local search improvement.

        Args:
            packages: Packages to deliver

        Returns:
            Ordered list of stop coordinates
        """
        if not packages:
            return []

        # Extract destination coordinates
        destinations = [pkg.destination for pkg in packages]

        # Apply nearest neighbor TSP
        optimized_stops = self.router.nearest_neighbor_tsp(
            destinations,
            self.delivery_map.depot,
            self.delivery_map
        )

        # Optionally apply 2-opt improvement
        if self.use_2opt and len(optimized_stops) > 2:
            optimized_stops = self.router.two_opt_improvement(
                optimized_stops,
                self.delivery_map,
                max_iterations=50
            )

        return optimized_stops

    def __str__(self) -> str:
        """String representation of agent."""
        opt_str = " (with 2-opt)" if self.use_2opt else ""
        return f"{self.name}{opt_str}: {self.description}"

    def __repr__(self) -> str:
        """Developer representation of agent."""
        return f"StudentAgent(name='{self.name}', use_2opt={self.use_2opt})"
