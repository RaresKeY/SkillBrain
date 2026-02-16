"""Module entrypoint for `python -m fleet_game` from the assignment root."""


if __name__ == "__main__":
    from pathlib import Path

    from main import main

    assignment_root = Path(__file__).resolve().parents[1]
    main(assignment_root)
