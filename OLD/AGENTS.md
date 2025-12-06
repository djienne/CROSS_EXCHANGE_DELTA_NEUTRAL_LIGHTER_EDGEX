# Repository Guidelines

## Project Structure & Module Organization
Core hedging logic lives in `hedge_cli.py`, which loads `hedge_config.json` and drives EdgeX/Lighter order flow, logging to `hedge_cli.log` under `logs/`. Safeguards like `liquidation_monitor.py` sit alongside automation helpers (`auto_rotation_bot.py`, `auto_rotation_bot_backup.py`). Sample integrations in `examples/` demonstrate data ingestion and market-making; treat them as blueprints. Reference notes live in `doc/`, while `.env` and presets such as `rotation_bot_config.json` capture environment-sensitive settings.

## Build, Test, and Development Commands
Create an isolated environment with `python -m venv .venv && source .venv/bin/activate`, then install dependencies via `pip install -r requirements.txt`. Run analysis utilities like `python hedge_cli.py funding` or `python hedge_cli.py funding_all --symbols BTC ETH SOL` to validate exchange connectivity. Execute simulated hedges with `python hedge_cli.py test --config hedge_config.json --notional 20` and promote to live orders using `open`/`close` subcommands. Keep liquidation monitoring active in another session: `python liquidation_monitor.py --interval 60 --margin-threshold 20.0`.

## Coding Style & Naming Conventions
Python files target 3.8+ and follow four-space indentation with concise, sentence-case docstrings. Favor dataclasses and type hints when modelling configs, mirroring `AppConfig` in `hedge_cli.py`. Functions should be descriptive verbs or verb phrases (`load_env`, `cross_price`), while module-level constants remain uppercase. Surface operational events through the `logging` package—prefer structured messages over print statements.

## Testing Guidelines
There is no standalone pytest suite; rely on CLI drills to validate behaviour. For new features, add idempotent `test_*` or `verify_*` subcommands so contributors can reproduce scenarios with `python hedge_cli.py <command>`. Capture and inspect `logs/` output after each run, and document any manual steps in the PR. If you introduce pure functions, add lightweight self-checks under a `__main__` guard to keep regression checks scriptable.

## Commit & Pull Request Guidelines
History is minimal (e.g., `v1`), so please adopt clear, imperative summaries such as `feat: align lighter tick rounding`. Group related edits into focused commits and note any config or API contract changes in the body. Pull requests should outline motivation, list the commands executed for validation, and call out required credentials or environment tweaks. Link issues when available and provide screenshots or log excerpts for monitoring-facing work.

## Security & Configuration Tips
Never commit populated `.env`, API keys, or exchange secrets; prefer template updates and document new variables in `README.md`. Review `hedge_config.json` diffs carefully—misconfigured exchanges can invert hedges. When sharing logs, redact account identifiers and signature material. Before deploying, sanity-check funding direction and leverage limits using the read-only commands above.
