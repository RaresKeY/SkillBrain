# Fleet Game (Extracted and Cleaned)

This folder contains a cleaned extraction of the delivery fleet game from `old_code`.

## Structure

- `fleet_game/`: core package
  - `core/`: engine, router, validation, difficulty
  - `models/`: game entities and state
  - `utils/`: data loading, metrics, package generation
  - `agents/`: agent interfaces and all bundled agents
    - `strategies/`: `greedy`, `backtracking`, `student`, `agent_r`, `agent_r_neural`, `agent_r_cheating`, `blank_agent`
  - `ui/`: Pygame rendering and manual mode components
- `data/`: bundled JSON game data
- `tests/`: pytest suite for engine/models/router/data loader
- `main.py`: CLI entrypoint
- `main_pygame.py`: Pygame UI entrypoint

## Notes

- Legacy `src` path hacks were removed in favor of package-relative imports.
- Legacy drafts/research branches remain under `old_code/` for reference only.

## Run

```bash
python3 main.py
```

## Run Pygame UI

```bash
python3 main_pygame.py
```

## Test

```bash
pytest -q
```

If `pytest` is not installed, install requirements first.
