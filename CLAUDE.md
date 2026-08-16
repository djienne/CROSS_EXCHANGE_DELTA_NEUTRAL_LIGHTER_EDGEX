# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Cross-exchange delta-neutral hedging system for cryptocurrency perpetual futures. Opens simultaneous long/short positions on EdgeX and Lighter exchanges to capture funding rate arbitrage while maintaining market-neutral exposure.

## Quick Reference Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Analysis (safe, no trading)
python examples/hedge_cli.py funding_all     # Compare funding rates
python examples/hedge_cli.py capacity        # Check available capital
python examples/hedge_cli.py status          # Check current positions
python check_all_spreads.py                  # Check cross-exchange spreads
python check_volume.py                       # Check 24h trading volume

# Test trading (minimal capital)
python examples/hedge_cli.py test --notional 20      # Full open+close cycle
python examples/hedge_cli.py test_leverage           # Verify leverage setup

# Production bot
python lighter_edgex_hedge.py                        # Run locally
docker-compose up -d lighter_edgex_hedge             # Run with Docker (recommended)
docker-compose logs -f lighter_edgex_hedge           # Monitor logs

# Emergency close (WINDOWS: must use Docker)
python emergency_close.py --dry-run                  # Check positions (Linux/macOS)
docker-compose run emergency_close --dry-run         # Check positions (Windows)
```

## Architecture

```
utils.py (shared utilities - ~200 lines)
    ├── Colors class (ANSI terminal colors)
    ├── Rounding: _round_to_tick, _ceil_to_tick, _floor_to_tick
    ├── Environment: load_env()
    ├── DateTime: utc_now, utc_now_iso, to_iso_z, from_iso_z
    └── Math: compute_base_size_from_quote, get_avg_mid, _calculate_apr

bot_state.py (state management - ~500 lines)
    ├── BotState dataclass (position tracking)
    ├── BotConfig dataclass (configuration)
    ├── StateManager class (JSON persistence)
    ├── DEFAULT_SYMBOLS list
    └── Display functions: display_funding_table, display_cycle_summary, display_status

lighter_client.py (Lighter exchange operations - ~680 lines)
    ├── imports utils.py (rounding functions)
    ├── Balance, position, order functions
    └── WebSocket price fetching

edgex_client.py (EdgeX exchange operations - ~460 lines)
    ├── imports utils.py (rounding functions)
    └── Balance, position, order functions

lighter_edgex_hedge.py (production bot - ~2900 lines)
    ├── imports utils.py, bot_state.py
    ├── imports lighter_client.py, edgex_client.py
    └── State machine, funding analysis, position management

examples/hedge_cli.py (manual CLI - ~2300 lines)
    ├── imports utils.py
    ├── imports lighter_client.py, edgex_client.py (via aliases)
    └── Interactive trading commands

emergency_close.py (emergency tool - ~400 lines)
    ├── imports utils.py
    ├── imports lighter_client.py, edgex_client.py
    └── Position closing (bypasses config files)
```

### Module Dependencies
```
utils.py ← lighter_client.py, edgex_client.py, bot_state.py
                    ↑                    ↑
            lighter_edgex_hedge.py, examples/hedge_cli.py, emergency_close.py
```

### Bot State Machine

1. **IDLE** → Waiting to start
2. **ANALYZING** → Fetching funding rates, volumes, spreads
3. **OPENING** → Executing delta-neutral entry
4. **HOLDING** → Monitoring position, collecting funding
5. **CLOSING** → Exiting both positions
6. **WAITING** → Cooldown before next cycle
7. **ERROR** → Manual intervention required

State persists in `logs/bot_state.json` for crash recovery.

## Configuration

### bot_config.json (production bot)
- `symbols_to_monitor`: Symbols to analyze
- `leverage`: 1-5x leverage on both exchanges
- `notional_per_position`: Max USD position size
- `hold_duration_hours`: Position hold time
- `min_net_apr_threshold`: Minimum APR to open (%)
- `min_volume_usd`: Minimum 24h volume filter (default: $250M)
- `max_spread_pct`: Maximum cross-exchange spread (default: 0.15%)

### .env credentials

```env
# EdgeX (CRITICAL: EDGEX_ACCOUNT_ID must be integer)
EDGEX_ACCOUNT_ID=123456
EDGEX_STARK_PRIVATE_KEY=0x...

# Lighter
LIGHTER_PRIVATE_KEY=0x...
LIGHTER_ACCOUNT_INDEX=0
LIGHTER_API_KEY_INDEX=0
```

## Critical Implementation Details

### EdgeX SDK Gotchas

```python
# CRITICAL: account_id MUST be int, not string (SDK uses bitwise operations)
edgex_client = EdgeXClient(account_id=int(env["EDGEX_ACCOUNT_ID"]), ...)

# CRITICAL: contract_id MUST be str in CreateOrderParams
params = CreateOrderParams(contract_id=str(contract_id), ...)
```

- Contract naming: `{SYMBOL}{QUOTE}` (e.g., "BTCUSD", "PAXGUSD")
- Package imports as `edgex_sdk` (not `edgex_python_sdk`)

### Lighter SDK Notes

- Market identification: Symbol only (e.g., "BTC", "PAXG")
- Position attributes: Use `pos.position` (unsigned) with `pos.sign` (1=long, -1=short)
- Entry price: `pos.avg_entry_price` (NOT `pos.entry_price`)
- Capital query: WebSocket `user_stats/{account_index}` channel
- Position close: Dual reduce-only orders (buy + sell), only offsetting side executes

### Position Sizing (Delta-Neutral)

Sizes must be IDENTICAL on both exchanges:
1. Get tick sizes from both exchanges
2. Use coarser tick size (max of both)
3. Floor to that tick size
4. Both exchanges get the same size

### Order Execution

Aggressive limit orders crossing the spread:
```python
# BUY: price = mid × 1.03 (3% above)
# SELL: price = mid × 0.97 (3% below)
```

### Rate Limiting (Lighter)

- Global semaphore limits max 2 concurrent Lighter API calls
- 1.0s staggered delay between symbol fetches
- Exponential backoff on 429 errors
- Funding rate cache (5-minute TTL) prevents redundant calls

## Helper Functions

### utils.py (shared utilities)
- `load_env()`: Load and parse .env credentials
- `_round_to_tick()`, `_ceil_to_tick()`, `_floor_to_tick()`: Decimal-precise rounding
- `utc_now()`, `to_iso_z()`, `from_iso_z()`: Timezone-aware datetime helpers
- `compute_base_size_from_quote()`, `get_avg_mid()`: Position sizing math

### bot_state.py (state management)
- `BotState`: Dataclass for tracking position details
- `BotConfig`: Dataclass for bot configuration
- `StateManager`: JSON persistence with crash recovery
- `display_funding_table()`: Colored terminal output for funding rates
- `display_cycle_summary()`: Position entry/exit summaries

### lighter_client.py (Lighter operations)
- `get_lighter_balance()`: Fetch balance via WebSocket
- `get_lighter_market_details()`: Get market_id and tick sizes
- `get_lighter_best_bid_ask()`: Fetch prices via WebSocket
- `get_lighter_position_details()`: Full position info with PnL
- `lighter_close_position()`: Close with reduce-only order
- `lighter_place_aggressive_order()`: Place market-crossing order

### edgex_client.py (EdgeX operations)
- `get_edgex_balance()`: Fetch USD balance
- `get_edgex_contract_details()`: Get contract_id and tick sizes
- `get_edgex_best_bid_ask()`: Fetch prices from quote API
- `get_edgex_position_details()`: Position info with calculated PnL
- `close_position()`: Close with offsetting aggressive order
- `place_aggressive_order()`: Place market-crossing order

## Logs & State Files

- `logs/bot_state.json`: Bot state (persists across restarts)
- `logs/lighter_edgex_hedge.log`: Full DEBUG output
- `hedge_cli.log`: CLI operations

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `TypeError: unsupported operand type(s) for +: 'int' and 'str'` | Use `int(env["EDGEX_ACCOUNT_ID"])` |
| Position size mismatch | Bot uses coarser tick size + floor |
| Unhedged position detected | Run `emergency_close.py`, fix state file |
| Rate limit errors (429) | Built-in retry with backoff; reduce `symbols_to_monitor` if persistent |
| Windows compatibility | Use Docker for all commands |

## Windows Limitation

Lighter SDK only supports Linux/macOS. On Windows, ALL commands must run via Docker:
```bash
docker-compose run emergency_close --dry-run
docker-compose up lighter_edgex_hedge
```
