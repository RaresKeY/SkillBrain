"""
Routing utilities and algorithms.

This module provides helper functions for route calculations and optimization,
including TSP (Traveling Salesman Problem) heuristics.
"""

import math
from typing import List, Tuple
from ..models import DeliveryMap, Package


class Router:
    """
    Utility class for route calculations and optimization.

    This class provides static methods for common routing operations
    that can be used by various routing agents.
    """

    @staticmethod
    def calculate_route_distance(stops: List[Tuple[float, float]],
                                 delivery_map: DeliveryMap,
                                 return_to_depot: bool = True) -> float:
        """
        Calculate total distance for a sequence of stops.

        Args:
            stops: Ordered list of (x, y) coordinates
            delivery_map: Map for distance calculations
            return_to_depot: Whether to include return trip to depot

        Returns:
            Total distance in kilometers
        """
        if not stops:
            return 0.0

        distance = 0.0
        depot = delivery_map.depot

        # Depot to first stop
        distance += delivery_map.distance(depot, stops[0])

        # Between consecutive stops
        for i in range(len(stops) - 1):
            distance += delivery_map.distance(stops[i], stops[i + 1])

        # Return to depot
        if return_to_depot:
            distance += delivery_map.distance(stops[-1], depot)

        return distance

    @staticmethod
    def nearest_neighbor_tsp(points: List[Tuple[float, float]],
                            start: Tuple[float, float],
                            delivery_map: DeliveryMap) -> List[Tuple[float, float]]:
        """
        Solve TSP using nearest neighbor greedy heuristic.

        Starting from a point, repeatedly visit the nearest unvisited point.

        Args:
            points: List of points to visit
            start: Starting point (usually depot)
            delivery_map: Map for distance calculations

        Returns:
            Ordered list of points in visit order

        Time Complexity: O(n²) where n is number of points
        Space Complexity: O(n)

        Example:
            >>> router = Router()
            >>> points = [(10, 20), (30, 40), (5, 10)]
            >>> route = router.nearest_neighbor_tsp(points, (0, 0), map)
        """
        if not points:
            return []

        unvisited = points.copy()
        route = []
        current = start

        while unvisited:
            # Find nearest unvisited point
            nearest = min(unvisited,
                         key=lambda p: delivery_map.distance(current, p))
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return route

    @staticmethod
    def two_opt_improvement(route: List[Tuple[float, float]],
                           delivery_map: DeliveryMap,
                           max_iterations: int = 100) -> List[Tuple[float, float]]:
        """
        Improve route using 2-opt local search.

        2-opt iteratively removes crossing edges in the route to reduce distance.
        This is a local optimization that improves an existing route.

        Args:
            route: Initial route to improve
            delivery_map: Map for distance calculations
            max_iterations: Maximum number of improvement iterations

        Returns:
            Improved route

        Time Complexity: O(n² * iterations) where n is route length
        """
        if len(route) < 3:
            return route

        improved_route = route.copy()
        improved = True
        iteration = 0

        def route_distance(r: List[Tuple[float, float]]) -> float:
            return Router.calculate_route_distance(r, delivery_map, return_to_depot=True)

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            best_distance = route_distance(improved_route)

            for i in range(1, len(improved_route) - 1):
                for j in range(i + 1, len(improved_route)):
                    # Try reversing segment [i:j+1]
                    new_route = improved_route[:i] + improved_route[i:j+1][::-1] + improved_route[j+1:]
                    new_distance = route_distance(new_route)

                    if new_distance < best_distance:
                        improved_route = new_route
                        best_distance = new_distance
                        improved = True
                        break

                if improved:
                    break

        return improved_route

    @staticmethod
    def optimize_package_sequence(packages: List[Package],
                                  delivery_map: DeliveryMap,
                                  use_2opt: bool = False) -> List[Package]:
        """
        Optimize the delivery sequence for a list of packages.

        Uses nearest neighbor TSP, optionally followed by 2-opt improvement.

        Args:
            packages: List of packages to sequence
            delivery_map: Map for distance calculations
            use_2opt: Whether to apply 2-opt improvement (slower but better)

        Returns:
            Packages in optimized delivery order
        """
        if len(packages) <= 1:
            return packages

        # Extract destinations
        destinations = [pkg.destination for pkg in packages]

        # Optimize route
        optimized_destinations = Router.nearest_neighbor_tsp(
            destinations,
            delivery_map.depot,
            delivery_map
        )

        if use_2opt:
            optimized_destinations = Router.two_opt_improvement(
                optimized_destinations,
                delivery_map
            )

        # Reorder packages to match optimized destinations
        destination_to_package = {pkg.destination: pkg for pkg in packages}
        optimized_packages = [destination_to_package[dest]
                             for dest in optimized_destinations]

        return optimized_packages

    @staticmethod
    def estimate_delivery_time(distance_km: float,
                              avg_speed_kmh: float = 50.0,
                              stop_time_minutes: int = 10) -> float:
        """
        Estimate total time for route including stops.

        Args:
            distance_km: Total route distance
            avg_speed_kmh: Average travel speed
            stop_time_minutes: Time per delivery stop

        Returns:
            Estimated time in hours
        """
        travel_time = distance_km / avg_speed_kmh
        # This is simplified - would need number of stops for accurate calculation
        return travel_time

    @staticmethod
    def cluster_packages_by_proximity(packages: List[Package],
                                     delivery_map: DeliveryMap,
                                     num_clusters: int) -> List[List[Package]]:
        """
        Group packages into clusters based on proximity.

        Simple k-means-like clustering to divide packages among vehicles.
        This is a basic implementation for educational purposes.

        Args:
            packages: Packages to cluster
            delivery_map: Map for distance calculations
            num_clusters: Number of clusters to create

        Returns:
            List of package clusters
        """
        if not packages or num_clusters <= 0:
            return []

        if len(packages) <= num_clusters:
            return [[pkg] for pkg in packages]

        # Simple approach: sort by angle from depot and divide
        def angle_from_depot(pkg: Package) -> float:
            dx = pkg.destination[0] - delivery_map.depot[0]
            dy = pkg.destination[1] - delivery_map.depot[1]
            return math.atan2(dy, dx)

        sorted_packages = sorted(packages, key=angle_from_depot)

        # Divide into roughly equal clusters
        cluster_size = len(sorted_packages) // num_clusters
        clusters = []

        for i in range(num_clusters):
            start_idx = i * cluster_size
            if i == num_clusters - 1:
                # Last cluster gets remaining packages
                clusters.append(sorted_packages[start_idx:])
            else:
                end_idx = start_idx + cluster_size
                clusters.append(sorted_packages[start_idx:end_idx])

        return clusters
