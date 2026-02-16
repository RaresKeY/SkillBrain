"""
Metrics and analytics utilities for route and game performance.

This module provides functions for calculating and displaying performance metrics.
"""

from typing import List, Dict
from ..models import Route, DayHistory, GameState


def calculate_route_metrics(routes: List[Route]) -> Dict:
    """
    Calculate aggregate metrics for a set of routes.

    Args:
        routes: List of routes to analyze

    Returns:
        Dictionary containing performance metrics
    """
    if not routes:
        return {
            'total_distance': 0.0,
            'total_cost': 0.0,
            'total_revenue': 0.0,
            'total_profit': 0.0,
            'vehicles_used': 0,
            'packages_delivered': 0,
            'avg_efficiency': 0.0,
            'avg_capacity_utilization': 0.0
        }

    total_distance = sum(r.total_distance for r in routes)
    total_cost = sum(r.total_cost for r in routes)
    total_revenue = sum(r.total_revenue for r in routes)
    total_profit = total_revenue - total_cost

    # Calculate average efficiency and capacity utilization
    efficiencies = [r.efficiency for r in routes if r.total_distance > 0]
    avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0.0

    utilizations = [
        (r.total_volume / r.vehicle.vehicle_type.capacity_m3)
        for r in routes
    ]
    avg_capacity_utilization = sum(utilizations) / len(utilizations) if utilizations else 0.0

    return {
        'total_distance': total_distance,
        'total_cost': total_cost,
        'total_revenue': total_revenue,
        'total_profit': total_profit,
        'vehicles_used': len(routes),
        'packages_delivered': sum(len(r.packages) for r in routes),
        'avg_efficiency': avg_efficiency,
        'avg_capacity_utilization': avg_capacity_utilization * 100  # As percentage
    }


def format_route_summary(route: Route) -> str:
    """
    Format a route as a human-readable summary string.

    Args:
        route: Route to format

    Returns:
        Formatted string with route details
    """
    summary = route.get_summary()
    return f"""
Route Summary for {summary['vehicle_type']} ({summary['vehicle_id']}):
  Packages: {summary['num_packages']}
  Volume: {summary['total_volume']:.1f}m³ ({summary['capacity_used']})
  Distance: {summary['total_distance']:.1f} km
  Cost: ${summary['total_cost']:.2f}
  Revenue: ${summary['total_revenue']:.2f}
  Profit: ${summary['profit']:.2f}
  Efficiency: ${summary['efficiency']:.2f}/km
  Valid: {summary['is_valid']}
"""


def format_metrics_table(metrics: Dict) -> str:
    """
    Format metrics dictionary as a table.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Formatted string table
    """
    return f"""
Performance Metrics:
{'─' * 50}
  Total Distance:     {metrics['total_distance']:>10.1f} km
  Total Cost:         ${metrics['total_cost']:>10.2f}
  Total Revenue:      ${metrics['total_revenue']:>10.2f}
  Total Profit:       ${metrics['total_profit']:>10.2f}
  Vehicles Used:      {metrics['vehicles_used']:>10}
  Packages Delivered: {metrics['packages_delivered']:>10}
  Avg Efficiency:     ${metrics['avg_efficiency']:>10.2f}/km
  Avg Capacity Use:   {metrics['avg_capacity_utilization']:>10.1f}%
{'─' * 50}
"""


def compare_agent_results(agent_results: Dict[str, Dict]) -> str:
    """
    Create a comparison table for multiple agent results.

    Args:
        agent_results: Dictionary mapping agent names to their metrics

    Returns:
        Formatted comparison table
    """
    if not agent_results:
        return "No agent results to compare."

    header = f"{'Agent':<20} {'Profit':>12} {'Distance':>12} {'Vehicles':>10} {'Packages':>10}"
    separator = '─' * 70

    lines = ["\nAgent Comparison:", separator, header, separator]

    for agent_name, metrics in agent_results.items():
        line = (f"{agent_name:<20} "
                f"${metrics['total_profit']:>11.2f} "
                f"{metrics['total_distance']:>11.1f} km "
                f"{metrics['vehicles_used']:>10} "
                f"{metrics['packages_delivered']:>10}")
        lines.append(line)

    lines.append(separator)

    # Identify best performers
    best_profit_agent = max(agent_results.items(),
                           key=lambda x: x[1]['total_profit'])
    best_distance_agent = min(agent_results.items(),
                             key=lambda x: x[1]['total_distance'])

    lines.append(f"\nBest Profit: {best_profit_agent[0]} (${best_profit_agent[1]['total_profit']:.2f})")
    lines.append(f"Best Distance: {best_distance_agent[0]} ({best_distance_agent[1]['total_distance']:.1f} km)")

    return '\n'.join(lines)


def format_game_statistics(game_state: GameState) -> str:
    """
    Format overall game statistics.

    Args:
        game_state: Current game state

    Returns:
        Formatted statistics string
    """
    stats = game_state.get_statistics()

    return f"""
Game Statistics:
{'═' * 50}
  Current Day:        {stats['current_day']}
  Current Balance:    ${stats['current_balance']:,.2f}
  Total Days Played:  {stats['total_days']}
  Fleet Size:         {stats['fleet_size']} vehicles

  Total Profit:       ${stats['total_profit']:,.2f}
  Avg Daily Profit:   ${stats['avg_daily_profit']:,.2f}
  Total Packages:     {stats['total_packages']}
  Delivery Rate:      {stats['delivery_rate']:.1f}%
{'═' * 50}
"""


def format_day_history(history: DayHistory) -> str:
    """
    Format a single day's history.

    Args:
        history: DayHistory record

    Returns:
        Formatted string
    """
    return f"""Day {history.day} Results:
  Packages: {history.packages_delivered}/{history.packages_attempted} ({history.delivery_rate:.1f}%)
  Revenue: ${history.revenue:.2f}
  Costs: ${history.costs:.2f}
  Profit: ${history.profit:+.2f}
  Agent: {history.agent_used}
  Routes: {history.routes_count}
  Distance: {history.total_distance:.1f} km
  Balance: ${history.balance_end:,.2f}
"""
