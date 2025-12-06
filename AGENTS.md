# Repository Guidelines

## Project Structure & Module Organization
- Production bot lives in `lighter_edgex_hedge.py`; manual operations run through `examples/hedge_cli.py`; failsafe liquidation is `emergency_close.py`.
- Strategy inputs stay in `bot_config.json` (rotation settings) and `.env` (API keys); keep runtime state in `bot_state.json` and logs under `logs/`.
- Exchange helpers are in `lighter_client.py` and `edgex_client.py`; ad-hoc diagnostics (e.g., `check_all_spreads.py`, `check_volume.py`, `debug_positions.py`) sit at repo root.
- Reference docs are under `doc/`; experiments and sample scripts live in `examples/`. Container flows rely on `Dockerfile` and `docker-compose.yml`—ensure volumes include state files for clean restarts.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`.
- Run automated rotation: `python lighter_edgex_hedge.py --state-file bot_state.json`.
- Dockerized bot: `docker-compose up -d lighter_edgex_hedge` (logs via `docker-compose logs -f lighter_edgex_hedge`).
- Funding scan before changes: `python examples/hedge_cli.py funding_all --symbols BTC ETH SOL`.
- Smoke full cycle: `python examples/hedge_cli.py test_auto --notional 20`.
- Safety drill: `python emergency_close.py --dry-run` (then rerun without `--dry-run` to flatten).

## Coding Style & Naming Conventions
- Python 3.10+; follow PEP 8 with 4-space indents and module-level docstrings on command entry points.
- Prefer dataclasses (e.g., a `BotConfig`) and type hints for configs/services; route logging through `logging.getLogger(__name__)`.
- Modules, files, and CLI subcommands stay snake_case and read like user intents (`open`, `funding_all`, `test_auto`).

## Testing Guidelines
- Use `pytest`; tests follow `test_*.py` naming (see root tests). Run with `pytest` from repo root.
- Manual smoke for open/hold/close is `python examples/hedge_cli.py test_auto --notional 20`; capture `logs/auto_rotation_bot.log` or CLI output when reporting issues.
- Add new tests under `tests/` if expanding coverage; keep them fast and deterministic.

## Commit & Pull Request Guidelines
- Commit subjects in present tense (e.g., `Add funding health checks`); wrap body lines ≤72 chars when needed. Keep scope tight (separate strategy tuning, SDK bumps, and state persistence tweaks).
- PRs should summarize behavior changes, list manual test commands, link issues, and add screenshots or terminal captures for dashboard/CLI updates. Call out any default config changes (e.g., `bot_config.json`) and required operator follow-up.

## Security & Configuration Tips
- Never commit secrets (`.env`, API keys, state exports); extend `.env.example` for new settings.
- Validate creds before runs with `python examples/hedge_cli.py status`; rotate keys per exchange policy and document rotations privately.
- Ensure Docker volumes mount state/log files so restarts reconcile positions instead of re-opening stale legs.
