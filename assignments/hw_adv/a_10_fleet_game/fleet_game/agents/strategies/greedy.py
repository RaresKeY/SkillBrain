"""
Greedy routing agent implementation.

This agent uses greedy heuristics for fast route planning:
- First-Fit Decreasing bin packing for package-to-vehicle assignment
- Nearest Neighbor for route optimization

Time Complexity: O(n²) where n is number of packages
Space Complexity: O(n)

Trade-offs:
+ Very fast execution
+ Simple to understand
+ Reasonable solutions for most cases
- May not find optimal solution
- No backtracking or look-ahead
"""

from typing import List
from ..base import RouteAgent
from ...models import Package, Vehicle, Route, DeliveryMap
from ...core import Router


class GreedyAgent(RouteAgent):
    """
    Greedy routing agent using nearest neighbor and first-fit heuristics.

    Algorithm:
    1. Sort packages by value density (payment per m³) - greedy choice
    2. Assign packages to vehicles using First-Fit Decreasing:
       - Try to fit package in first vehicle with capacity
       - If no vehicle fits, use a new vehicle
    3. For each route, optimize stop order using Nearest Neighbor TSP
    4. Optionally apply 2-opt improvement
    """

    def __init__(self, delivery_map: DeliveryMap, use_2opt: bool = False):
        """
        Initialize greedy agent.

        Args:
            delivery_map: Map for distance calculations
            use_2opt: Whether to apply 2-opt local search improvement
        """
        super().__init__(delivery_map, "Greedy Agent")
        self.description = "Fast greedy algorithm using nearest neighbor and first-fit"
        self.use_2opt = use_2opt
        self.router = Router()

    def plan_routes(self, packages: List[Package], fleet: List[Vehicle]) -> List[Route]:
        """
        Create routes using greedy heuristics.

        Args:
            packages: List of packages to deliver
            fleet: Available vehicles

        Returns:
            List of routes
        """
        if not self.validate_inputs(packages, fleet):
            return []

        print(f"[{self.name}] Planning routes for {len(packages)} packages with {len(fleet)} vehicles...")

        # Step 1: Sort packages by value density (greedy heuristic)
        # This prioritizes high-value, low-volume packages
        sorted_packages = sorted(packages, key=lambda p: p.value_density, reverse=True)

        # Step 2: Assign packages to vehicles (First-Fit Decreasing)
        routes = []
        available_vehicles = fleet.copy()
        unassigned_packages = []

        for pkg in sorted_packages:
            # Try to fit in existing route
            placed = False
            for route in routes:
                if route.add_package(pkg):
                    placed = True
                    break

            # Need new vehicle
            if not placed:
                if available_vehicles:
                    # Create new route with new vehicle
                    vehicle = available_vehicles.pop(0)
                    new_route = Route(
                        vehicle=vehicle,
                        packages=[pkg],
                        stops=[],
                        delivery_map=self.delivery_map
                    )
                    routes.append(new_route)
                else:
                    # No more vehicles - track unassigned
                    unassigned_packages.append(pkg)

        # Report unassigned packages if any
        if unassigned_packages:
            total_unassigned_volume = sum(p.volume_m3 for p in unassigned_packages)
            print(f"[{self.name}] Warning: {len(unassigned_packages)} packages ({total_unassigned_volume:.1f}m³) could not be assigned - insufficient capacity")

        # Step 3: Optimize stop order for each route
        for route in routes:
            route.stops = self._optimize_route_stops(route.packages)

        # Summary
        total_packages_assigned = sum(len(r.packages) for r in routes)
        print(f"[{self.name}] Created {len(routes)} routes with {total_packages_assigned}/{len(packages)} packages")

        return routes

    def _optimize_route_stops(self, packages: List[Package]) -> List[tuple]:
        """
        Optimize the order of stops for a set of packages.

        Uses Nearest Neighbor TSP heuristic, optionally followed by 2-opt.

        Args:
            packages: Packages to deliver

        Returns:
            Ordered list of stop coordinates
        """
        if not packages:
            return []

        # Extract destinations
        destinations = [pkg.destination for pkg in packages]

        # Apply nearest neighbor
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
        opt_str = " (with 2-opt)" if self.use_2opt else ""
        return f"{self.name}{opt_str}: {self.description}"
