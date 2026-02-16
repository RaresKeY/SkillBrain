#!/usr/bin/env python3
"""
Delivery Fleet Management System - Main Entry Point

A simulation game for learning algorithmic optimization through route planning.
Students implement and compare different routing algorithms (greedy, backtracking, DP, etc.)
to manage a delivery fleet and maximize profits.

Usage:
    python main.py
"""

from pathlib import Path

from fleet_game.core import GameEngine
from fleet_game.agents import (
    BacktrackingAgent,
    BlankAgent,
    CheatingRAgent,
    GreedyAgent,
    PruningBacktrackingAgent,
    RAgent,
    StudentAgent,
)
from fleet_game.utils import compare_agent_results, format_game_statistics, format_metrics_table


def print_banner():
    """Display welcome banner."""
    print("=" * 70)
    print(" " * 15 + "DELIVERY FLEET MANAGEMENT SYSTEM")
    print(" " * 20 + "Art of Programming - Phase 1")
    print("=" * 70)
    print()


def print_menu():
    """Display main menu."""
    print("\n" + "─" * 70)
    print("MAIN MENU:")
    print("─" * 70)
    print("  1. Start New Game")
    print("  2. Load Saved Game")
    print("  3. View Game Status")
    print("  4. Start Day (Load Packages)")
    print("  5. Test Agent (Compare Algorithms)")
    print("  6. Apply Agent Solution")
    print("  7. Execute Day")
    print("  8. Advance to Next Day")
    print("  9. Purchase Vehicle")
    print(" 10. View Statistics")
    print(" 11. Save Game")
    print("  0. Exit")
    print("─" * 70)


def test_agents_menu(engine: GameEngine):
    """
    Interactive agent testing and comparison.

    Args:
        engine: Game engine instance
    """
    print("\n" + "=" * 70)
    print("AGENT TESTING & COMPARISON")
    print("=" * 70)

    available_agents = list(engine.agents.keys())
    if not available_agents:
        print("No agents registered!")
        return

    print("\nAvailable Agents:")
    for i, agent_name in enumerate(available_agents, 1):
        agent = engine.agents[agent_name]
        print(f"  {i}. {agent}")

    print("\nOptions:")
    print("  a. Test all agents")
    print("  c. Compare all agents")
    print("  Or enter agent number to test individually")

    choice = input("\nYour choice: ").strip().lower()

    if choice == 'a' or choice == 'c':
        # Test all agents
        results = {}
        for agent_name in available_agents:
            metrics = engine.test_agent(agent_name)
            if metrics:
                results[agent_name] = metrics
                if choice == 'a':  # Show individual results
                    print(format_metrics_table(metrics))

        if choice == 'c' and results:  # Show comparison
            comparison = compare_agent_results(results)
            print(comparison)

    elif choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(available_agents):
            agent_name = available_agents[idx]
            metrics = engine.test_agent(agent_name)
            if metrics:
                print(format_metrics_table(metrics))

                # Ask if user wants to apply this solution
                apply = input(f"\nApply {agent_name}'s solution? (y/n): ").strip().lower()
                if apply == 'y':
                    engine.apply_agent_solution(agent_name)
        else:
            print("Invalid agent number!")


def purchase_vehicle_menu(engine: GameEngine):
    """
    Interactive vehicle purchase menu.

    Args:
        engine: Game engine instance
    """
    print("\n" + "=" * 70)
    print("VEHICLE PURCHASE")
    print("=" * 70)

    print(f"Current Balance: ${engine.game_state.balance:,.2f}")
    print(f"Current Fleet Size: {len(engine.game_state.fleet)} vehicles\n")

    print("Available Vehicle Types:")
    vehicle_types = list(engine.vehicle_types.items())
    for i, (key, vtype) in enumerate(vehicle_types, 1):
        print(f"  {i}. {vtype}")

    choice = input("\nEnter vehicle number to purchase (0 to cancel): ").strip()

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(vehicle_types):
            vehicle_key = vehicle_types[idx][0]
            engine.purchase_vehicle(vehicle_key)
        elif idx == -1:
            print("Cancelled.")
        else:
            print("Invalid choice!")


def main(project_root: Path | None = None):
    """Main game loop."""
    print_banner()

    # Initialize game engine
    root = project_root or Path(__file__).parent
    data_dir = root / "data"
    engine = GameEngine(data_dir)

    # Register agents
    print("Registering routing agents...")
    engine.register_agent("greedy", GreedyAgent(engine.delivery_map))
    engine.register_agent("greedy_2opt", GreedyAgent(engine.delivery_map, use_2opt=True))
    engine.register_agent("student", StudentAgent(engine.delivery_map))
    engine.register_agent("student_2opt", StudentAgent(engine.delivery_map, use_2opt=True))
    engine.register_agent("agent_r", RAgent(engine.delivery_map))
    engine.register_agent("agent_r_cheating", CheatingRAgent(engine.delivery_map))
    engine.register_agent("blank_agent", BlankAgent(engine.delivery_map))
    try:
        from fleet_game.agents import NeuralRAgent

        engine.register_agent("agent_r_neural", NeuralRAgent(engine.delivery_map))
    except Exception as exc:
        print(f"Warning: failed to register Agent R Neural: {exc}")
    engine.register_agent("backtracking", BacktrackingAgent(engine.delivery_map, max_packages=12))
    engine.register_agent("pruning_backtracking", PruningBacktrackingAgent(engine.delivery_map, max_packages=15))

    print("\nWelcome to the Delivery Fleet Management System!")
    print("Manage your fleet, optimize routes, and maximize profits!")

    # Main game loop
    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip()

        try:
            if choice == '1':
                # New Game
                engine.new_game()

            elif choice == '2':
                # Load Game
                save_file = input("Enter save file name (default: savegame.json): ").strip()
                if not save_file:
                    save_file = "savegame.json"
                engine.load_game(save_file)

            elif choice == '3':
                # View Status
                if engine.game_state:
                    print(f"\n{engine.get_status()}")
                    print(f"Fleet: {len(engine.game_state.fleet)} vehicles")
                    print(f"Pending packages: {len(engine.game_state.packages_pending)}")
                else:
                    print("\nNo active game! Start a new game first.")

            elif choice == '4':
                # Start Day
                engine.start_day()

            elif choice == '5':
                # Test Agent
                if not engine.game_state:
                    print("\nNo active game! Start a new game first.")
                elif not engine.game_state.packages_pending:
                    print("\nNo packages to deliver! Start a day first.")
                else:
                    test_agents_menu(engine)

            elif choice == '6':
                # Apply Agent Solution
                if not engine.game_state:
                    print("\nNo active game! Start a new game first.")
                elif not engine.game_state.packages_pending:
                    print("\nNo packages to deliver! Start a day first.")
                else:
                    print("\nAvailable agents:")
                    for i, (name, agent) in enumerate(engine.agents.items(), 1):
                        print(f"  {i}. {name}")

                    agent_choice = input("\nEnter agent number: ").strip()
                    if agent_choice.isdigit():
                        idx = int(agent_choice) - 1
                        agent_names = list(engine.agents.keys())
                        if 0 <= idx < len(agent_names):
                            engine.apply_agent_solution(agent_names[idx])

            elif choice == '7':
                # Execute Day
                if not engine.game_state:
                    print("\nNo active game! Start a new game first.")
                elif not engine.game_state.current_routes:
                    print("\nNo routes planned! Test and apply an agent solution first.")
                else:
                    # Determine which agent was used (from last applied)
                    agent_name = "Manual"  # Default
                    engine.execute_day(agent_name)

            elif choice == '8':
                # Advance Day
                if engine.game_state:
                    engine.advance_to_next_day()
                else:
                    print("\nNo active game!")

            elif choice == '9':
                # Purchase Vehicle
                if engine.game_state:
                    purchase_vehicle_menu(engine)
                else:
                    print("\nNo active game! Start a new game first.")

            elif choice == '10':
                # View Statistics
                if engine.game_state:
                    stats = format_game_statistics(engine.game_state)
                    print(stats)
                else:
                    print("\nNo active game!")

            elif choice == '11':
                # Save Game
                if engine.game_state:
                    save_file = input("Enter save file name (default: savegame.json): ").strip()
                    if not save_file:
                        save_file = "savegame.json"
                    engine.save_game(save_file)
                else:
                    print("\nNo active game to save!")

            elif choice == '0':
                # Exit
                print("\nThank you for playing! Goodbye!")
                break

            else:
                print("\nInvalid choice! Please try again.")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Game session ended.")
    print("=" * 70)


if __name__ == "__main__":
    main()
