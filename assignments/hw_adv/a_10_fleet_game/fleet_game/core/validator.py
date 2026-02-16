"""
Validation utilities for routes and constraints.

This module provides functions to validate routes and check constraint satisfaction.
"""

from typing import List, Tuple
from ..models import Route, Package, Vehicle


class RouteValidator:
    """Validates routes against game constraints."""

    @staticmethod
    def validate_capacity(route: Route) -> Tuple[bool, str]:
        """
        Check if route respects vehicle capacity.

        Args:
            route: Route to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        total_volume = route.total_volume
        capacity = route.vehicle.vehicle_type.capacity_m3

        if total_volume > capacity:
            return False, f"Capacity exceeded: {total_volume:.1f}m³ > {capacity}m³"

        return True, ""

    @staticmethod
    def validate_range(route: Route) -> Tuple[bool, str]:
        """
        Check if route respects vehicle range.

        Args:
            route: Route to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        distance = route.total_distance
        max_range = route.vehicle.vehicle_type.max_range_km

        if distance > max_range:
            return False, f"Range exceeded: {distance:.1f}km > {max_range}km"

        return True, ""

    @staticmethod
    def validate_stops_match_packages(route: Route) -> Tuple[bool, str]:
        """
        Check that all package destinations are in route stops.

        Args:
            route: Route to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not route.packages:
            return False, "No packages in route"

        package_destinations = {pkg.destination for pkg in route.packages}
        stop_set = set(route.stops)

        if not package_destinations.issubset(stop_set):
            missing = package_destinations - stop_set
            return False, f"Missing destinations in stops: {missing}"

        return True, ""

    @staticmethod
    def validate_all(route: Route) -> Tuple[bool, List[str]]:
        """
        Run all validation checks on a route.

        Args:
            route: Route to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        valid, error = RouteValidator.validate_capacity(route)
        if not valid:
            errors.append(error)

        valid, error = RouteValidator.validate_range(route)
        if not valid:
            errors.append(error)

        if route.stops:  # Only check if stops are defined
            valid, error = RouteValidator.validate_stops_match_packages(route)
            if not valid:
                errors.append(error)

        return len(errors) == 0, errors

    @staticmethod
    def validate_solution(routes: List[Route], packages: List[Package]) -> Tuple[bool, List[str]]:
        """
        Validate a complete solution (set of routes).

        Checks:
        - All routes are individually valid
        - All packages are assigned exactly once
        - No duplicate package assignments

        Args:
            routes: List of routes to validate
            packages: Original list of packages to deliver

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Validate each route
        for i, route in enumerate(routes):
            valid, route_errors = RouteValidator.validate_all(route)
            if not valid:
                errors.append(f"Route {i} ({route.vehicle.id}): {', '.join(route_errors)}")

        # Check package coverage
        all_assigned_packages = []
        for route in routes:
            all_assigned_packages.extend(route.packages)

        assigned_ids = {pkg.id for pkg in all_assigned_packages}
        original_ids = {pkg.id for pkg in packages}

        # Check for missing packages
        missing = original_ids - assigned_ids
        if missing:
            errors.append(f"Unassigned packages: {missing}")

        # Check for duplicate assignments
        if len(all_assigned_packages) != len(assigned_ids):
            duplicates = [pkg.id for pkg in all_assigned_packages
                         if sum(1 for p in all_assigned_packages if p.id == pkg.id) > 1]
            errors.append(f"Duplicate package assignments: {set(duplicates)}")

        return len(errors) == 0, errors
