"""
bot_state.py
------------
State management for the delta-neutral rotation bot.

Contains:
- BotState: State machine states
- BotConfig: Bot configuration dataclass
- StateManager: Persistent state management with file-based storage
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

from utils import utc_now_iso, utc_now, from_iso_z, Colors

logger = logging.getLogger(__name__)


# ==================== Default Configuration ====================

DEFAULT_SYMBOLS: List[str] = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "ASTER",
    "PAXG",
    "DOGE",
    "XRP",
    "LINK",
    "HYPE",
    "XPL",
    "TRUMP",
    "LTC",
    "PUMP",
    "FARTCOIN",
]


# ==================== State Machine ====================

class BotState:
    """State machine states for the rotation bot."""
    IDLE = "IDLE"
    ANALYZING = "ANALYZING"
    OPENING = "OPENING"
    HOLDING = "HOLDING"
    CLOSING = "CLOSING"
    WAITING = "WAITING"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


# ==================== Configuration ====================

@dataclass
class BotConfig:
    """Bot configuration parameters."""
    symbols_to_monitor: List[str]
    quote: str = "USD"
    leverage: int = 3
    notional_per_position: float = 100.0
    hold_duration_hours: float = 8.0
    wait_between_cycles_minutes: float = 5.0
    check_interval_seconds: int = 60
    min_net_apr_threshold: float = 5.0
    min_volume_usd: float = 250_000_000  # Minimum 24h combined volume in USD (default: $250M)
    max_spread_pct: float = 0.15  # Maximum cross-exchange spread percentage (default: 0.15%)
    enable_stop_loss: bool = True
    enable_pnl_tracking: bool = True
    enable_health_monitoring: bool = True

    @staticmethod
    def load_from_file(config_file: str) -> 'BotConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)

            # Remove comment fields (any key starting with 'comment')
            data = {k: v for k, v in data.items() if not k.startswith('comment')}

            # Provide defaults for any missing fields (backward compatibility)
            defaults = {
                'symbols_to_monitor': DEFAULT_SYMBOLS,
                'quote': 'USD',
                'leverage': 3,
                'notional_per_position': 100.0,
                'hold_duration_hours': 8.0,
                'wait_between_cycles_minutes': 5.0,
                'check_interval_seconds': 60,
                'min_net_apr_threshold': 5.0,
                'min_volume_usd': 250_000_000,
                'max_spread_pct': 0.15,
                'enable_stop_loss': True,
                'enable_pnl_tracking': True,
                'enable_health_monitoring': True
            }

            for key, default_value in defaults.items():
                if key not in data:
                    data[key] = default_value
                    logger.info(f"Using default value for {key}: {default_value}")

            return BotConfig(**data)
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found, using defaults")
            return BotConfig(symbols_to_monitor=DEFAULT_SYMBOLS)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return BotConfig(symbols_to_monitor=DEFAULT_SYMBOLS)


# ==================== State Manager ====================

class StateManager:
    """Manages bot state persistence and recovery."""

    # Default MUST match what docker-compose sets via BOT_STATE_FILE
    # (`BOT_STATE_FILE=/app/logs/bot_state.json`, with ./ mounted at /app).
    #
    # This used to default to a bare "bot_state.json" at the repo root, so the container
    # wrote logs/bot_state.json while anything run on the host read a different, stale
    # file at the root. Two files, one of them silently wrong, is how a bot ends up
    # "recovering" a position that closed months ago - or forgetting a live one.
    DEFAULT_STATE_FILE = "logs/bot_state.json"
    LEGACY_STATE_FILE = "bot_state.json"

    def __init__(self, state_file: str = DEFAULT_STATE_FILE):
        self.state_file = state_file
        if (
            os.path.abspath(state_file) != os.path.abspath(self.LEGACY_STATE_FILE)
            and os.path.exists(self.LEGACY_STATE_FILE)
        ):
            logger.warning(
                "Stale legacy state file '%s' exists but '%s' is in use. The legacy file "
                "is NOT read - delete it once you have confirmed it holds nothing you "
                "need, so it cannot be mistaken for live state.",
                self.LEGACY_STATE_FILE,
                state_file,
            )
        self.state = {
            "version": "1.0",
            "state": BotState.IDLE,
            "current_cycle": 0,  # Current cycle number (increments when position opens)
            "current_position": None,
            "capital_status": {
                "edgex_total": 0.0,
                "edgex_available": 0.0,
                "lighter_total": 0.0,
                "lighter_available": 0.0,
                "total_capital": 0.0,
                "total_available": 0.0,
                "max_position_notional": 0.0,
                "limiting_exchange": None,
                "last_updated": None,
                "initial_total_capital": None  # Set once on first capital refresh, never changes
            },
            "completed_cycles": [],
            "cumulative_stats": {
                "total_cycles": 0,
                "successful_cycles": 0,
                "failed_cycles": 0,
                "total_realized_pnl": 0.0,
                "total_trading_pnl": 0.0,
                "total_funding_pnl": 0.0,
                "total_fees_paid": 0.0,
                "best_cycle_pnl": 0.0,
                "worst_cycle_pnl": 0.0,
                "total_volume_traded": 0.0,
                "total_hold_time_hours": 0.0,
                "by_symbol": {},
                "last_error": None,
                "last_error_at": None
            },
            "config": None,
            "last_updated": utc_now_iso()
        }

    def load(self) -> bool:
        """Load state from file. Returns True if loaded successfully."""
        if not os.path.exists(self.state_file):
            logger.info(f"No state file found at {self.state_file}, starting fresh")
            return False

        try:
            with open(self.state_file, 'r') as f:
                content = f.read().strip()

            # Handle empty file
            if not content:
                logger.info(f"State file {self.state_file} is empty, starting fresh")
                return False

            loaded_state = json.loads(content)

            # Merge with default state to handle new fields (backward compatibility)
            self.state.update(loaded_state)

            # Ensure capital_status exists (for older state files)
            if "capital_status" not in self.state:
                self.state["capital_status"] = {
                    "edgex_total": 0.0,
                    "edgex_available": 0.0,
                    "lighter_total": 0.0,
                    "lighter_available": 0.0,
                    "total_capital": 0.0,
                    "total_available": 0.0,
                    "max_position_notional": 0.0,
                    "limiting_exchange": None,
                    "last_updated": None,
                    "initial_total_capital": None
                }

            # Ensure initial_total_capital field exists (for older state files)
            if "initial_total_capital" not in self.state["capital_status"]:
                self.state["capital_status"]["initial_total_capital"] = None

            logger.info(f"Loaded state from {self.state_file}")
            logger.info(f"Current state: {self.state['state']}")
            return True
        except json.JSONDecodeError as e:
            logger.warning(f"State file {self.state_file} is corrupted or invalid JSON: {e}")
            logger.info("Starting fresh with new state")
            return False
        except Exception as e:
            logger.warning(f"Could not load state file: {e}")
            logger.info("Starting fresh with new state")
            return False

    def save(self):
        """Save current state to file with retry logic for Windows Docker volumes."""
        import time
        self.state["last_updated"] = utc_now_iso()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Write to temp file first, then atomic rename
                temp_file = self.state_file + ".tmp"
                with open(temp_file, 'w') as f:
                    json.dump(self.state, f, indent=2)
                os.replace(temp_file, self.state_file)
                logger.debug(f"Saved state to {self.state_file}")
                return  # Success
            except OSError as e:
                if e.errno == 16 and attempt < max_retries - 1:  # Device or resource busy
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                elif attempt == max_retries - 1:
                    logger.debug(f"Failed to save state after {max_retries} attempts: {e}")
                else:
                    logger.error(f"Failed to save state: {e}")
                    break
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                break

    def set_state(self, new_state: str):
        """Update bot state."""
        logger.info(f"State transition: {self.state['state']} -> {new_state}")
        self.state["state"] = new_state
        self.save()

    def get_state(self) -> str:
        """Get current bot state."""
        return self.state["state"]

    def set_config(self, config: BotConfig):
        """Set bot configuration."""
        self.state["config"] = asdict(config)
        self.save()

    def get_config(self) -> Optional[BotConfig]:
        """Get bot configuration."""
        if self.state["config"]:
            return BotConfig(**self.state["config"])
        return None


# ==================== Display Functions ====================

def display_funding_table(funding_data: list, excluded_data: list = None, current_symbol: Optional[str] = None, limit: int = 10):
    """Display funding rates comparison table with volume information and excluded symbols."""
    if not funding_data and not excluded_data:
        logger.info(f"{Colors.GRAY}No funding data available{Colors.RESET}")
        return

    logger.info(f"\n{Colors.BOLD}Funding Rates Overview{Colors.RESET}")
    logger.info(f"{Colors.CYAN}{'-' * 120}{Colors.RESET}")
    logger.info(f"{'Symbol':<10} {'EdgeX APR':>10} {'Lighter APR':>12} {'Net APR':>10} {'24h Volume':>14} {'Spread':>9} {'Long':>8} {'Status':<25}")
    logger.info(f"{Colors.GRAY}{'-' * 120}{Colors.RESET}")

    best_symbol = funding_data[0].get('symbol') if funding_data else None

    # Display available symbols (sorted by net APR)
    for idx, data in enumerate(funding_data[:limit]):
        symbol = data.get('symbol', 'N/A')
        edgex_apr = data.get('edgex_apr', 0)
        lighter_apr = data.get('lighter_apr', 0)
        net_apr = data.get('net_apr', 0)
        long_exch = data.get('long_exch', 'N/A')
        total_volume = data.get('total_volume')
        spread_pct = data.get('spread_pct')

        # Format volume
        if total_volume is not None:
            if total_volume >= 1_000_000_000:
                volume_str = f"${total_volume/1e9:.2f}B"
            elif total_volume >= 1_000_000:
                volume_str = f"${total_volume/1e6:.0f}M"
            else:
                volume_str = f"${total_volume/1e3:.0f}K"
        else:
            volume_str = "N/A"

        # Format spread
        if spread_pct is not None:
            spread_str = f"{spread_pct:.3f}%"
        else:
            spread_str = "N/A"

        # Color code based on status
        status = ""
        color = Colors.RESET
        if symbol == current_symbol:
            status = "< CURRENT"
            color = Colors.CYAN
        elif idx == 0:
            status = "* BEST"
            color = Colors.GREEN

        logger.info(f"{color}{symbol:<10} {edgex_apr:>9.2f}% {lighter_apr:>11.2f}% {net_apr:>9.2f}% {volume_str:>14} {spread_str:>9} {long_exch:>8} {status:<25}{Colors.RESET}")

    # Display excluded symbols (grayed out)
    if excluded_data:
        logger.info(f"{Colors.GRAY}{'-' * 120}{Colors.RESET}")
        for data in excluded_data:
            symbol = data.get('symbol', 'N/A')
            total_volume = data.get('total_volume')
            spread_pct = data.get('spread_pct')
            missing_on = data.get('missing_on', [])

            # Format volume
            if total_volume is not None:
                if total_volume >= 1_000_000_000:
                    volume_str = f"${total_volume/1e9:.2f}B"
                elif total_volume >= 1_000_000:
                    volume_str = f"${total_volume/1e6:.0f}M"
                else:
                    volume_str = f"${total_volume/1e3:.0f}K"
            else:
                volume_str = "N/A"

            # Format spread
            if spread_pct is not None:
                spread_str = f"{spread_pct:.3f}%"
            else:
                spread_str = "N/A"

            # Determine exclusion reason
            if missing_on and isinstance(missing_on, list) and len(missing_on) > 0:
                reason = missing_on[0]
                if "Volume data unavailable" in reason:
                    status = f"x EXCLUDED: Volume N/A"
                elif "Volume too low" in reason or "volume" in reason.lower():
                    status = f"x EXCLUDED: {reason}"
                elif "Spread too wide" in reason or "spread" in reason.lower():
                    status = f"x EXCLUDED: {reason}"
                else:
                    status = f"x UNAVAILABLE: {', '.join(missing_on)}"
            else:
                status = "x UNAVAILABLE"

            # Show excluded symbols in gray with N/A for rates
            logger.info(f"{Colors.GRAY}{symbol:<10} {'N/A':>10} {'N/A':>12} {'N/A':>10} {volume_str:>14} {spread_str:>9} {'N/A':>8} {status:<25}{Colors.RESET}")

    logger.info(f"{Colors.GRAY}{'-' * 120}{Colors.RESET}")

    # Show comparison if current is not best
    if current_symbol and best_symbol and current_symbol != best_symbol:
        current_data = next((d for d in funding_data if d.get('symbol') == current_symbol), None)
        if current_data:
            current_apr = current_data.get('net_apr', 0)
            best_apr = funding_data[0].get('net_apr', 0)
            diff = best_apr - current_apr
            logger.info(f"{Colors.YELLOW}Note: {best_symbol} currently has +{diff:.2f}% better APR{Colors.RESET}")

    logger.info("")


def display_cycle_summary(cycle: dict, cumulative_stats: dict):
    """Display summary of completed cycle."""
    logger.info(f"\n{Colors.CYAN}{'=' * 70}")
    logger.info(f"CYCLE #{cycle['cycle_number']} COMPLETE - {cycle['symbol']}")
    logger.info(f"{'=' * 70}{Colors.RESET}")

    pnl = cycle['pnl_breakdown']['total_realized_pnl']
    color = Colors.GREEN if pnl >= 0 else Colors.RED

    logger.info(f"Duration: {cycle['duration_hours']:.2f} hours")
    logger.info(f"Realized PnL: {color}${pnl:+.4f} ({cycle['pnl_breakdown']['realized_pnl_percent']:+.3f}%){Colors.RESET}")
    logger.info(f"  EdgeX: ${cycle['pnl_breakdown']['edgex_balance_change']:+.4f}")
    logger.info(f"  Lighter: ${cycle['pnl_breakdown']['lighter_balance_change']:+.4f}")

    logger.info(f"\n{Colors.CYAN}Cumulative Stats:{Colors.RESET}")
    logger.info(f"  Total Cycles: {cumulative_stats['total_cycles']}")
    logger.info(f"  Success Rate: {(cumulative_stats['successful_cycles'] / cumulative_stats['total_cycles'] * 100):.1f}%")

    total_pnl_color = Colors.GREEN if cumulative_stats['total_realized_pnl'] >= 0 else Colors.RED
    logger.info(f"  Total PnL: {total_pnl_color}${cumulative_stats['total_realized_pnl']:+.2f}{Colors.RESET}")
    logger.info(f"  Best Cycle: ${cumulative_stats['best_cycle_pnl']:+.4f}")
    logger.info(f"  Worst Cycle: ${cumulative_stats['worst_cycle_pnl']:+.4f}")
    logger.info(f"  Avg PnL/Cycle: ${(cumulative_stats['total_realized_pnl'] / cumulative_stats['total_cycles']):+.4f}")
    logger.info(f"{'=' * 70}\n")


def display_status(state_mgr: 'StateManager'):
    """Display current bot status."""
    state = state_mgr.get_state()
    pos = state_mgr.state.get("current_position")
    stats = state_mgr.state["cumulative_stats"]
    capital = state_mgr.state.get("capital_status", {})

    logger.info(f"\n{Colors.BOLD}{'=' * 70}")
    logger.info(f"BOT STATUS")
    logger.info(f"{'=' * 70}{Colors.RESET}")
    logger.info(f"State: {Colors.CYAN}{state}{Colors.RESET}")

    # Display capital status
    if capital and capital.get("last_updated"):
        logger.info(f"\nCapital Status:")
        logger.info(f"  EdgeX:   ${capital['edgex_total']:.2f} total, ${capital['edgex_available']:.2f} available")
        logger.info(f"  Lighter: ${capital['lighter_total']:.2f} total, ${capital['lighter_available']:.2f} available")
        logger.info(f"  {Colors.BOLD}Total:   ${capital['total_capital']:.2f} total, ${capital['total_available']:.2f} available{Colors.RESET}")

        # Display long-term PnL if initial capital is available
        initial_capital = capital.get('initial_total_capital')
        current_capital = capital.get('total_capital', 0)
        if initial_capital is not None and initial_capital > 0:
            long_term_pnl_dollars = current_capital - initial_capital
            long_term_pnl_percent = (long_term_pnl_dollars / initial_capital) * 100
            pnl_color = Colors.GREEN if long_term_pnl_dollars >= 0 else Colors.RED
            logger.info(f"  {Colors.BOLD}Long-term PnL: {pnl_color}{long_term_pnl_percent:+.2f}%{Colors.RESET} ({pnl_color}${long_term_pnl_dollars:+.2f}{Colors.RESET} from ${initial_capital:.2f})")

        max_pos_color = Colors.GREEN if capital['max_position_notional'] > 100 else Colors.YELLOW
        logger.info(f"  {max_pos_color}Max Position: ${capital['max_position_notional']:.2f} (limited by {capital['limiting_exchange']}){Colors.RESET}")

    if pos:
        logger.info(f"\nCurrent Position:")
        logger.info(f"  Symbol: {pos['symbol']}")
        logger.info(f"  Setup: Long {pos['long_exchange']}, Short {pos['short_exchange']}")
        logger.info(f"  Opened: {pos['opened_at']}")
        logger.info(f"  Target Close: {pos['target_close_at']}")

        # Get timing info from state or calculate fresh
        if 'time_elapsed_hours' in pos and 'time_remaining_hours' in pos:
            elapsed = pos['time_elapsed_hours']
            remaining = pos['time_remaining_hours']
            progress = pos.get('progress_percent', 0)
        else:
            opened = from_iso_z(pos['opened_at'])
            target = from_iso_z(pos['target_close_at'])
            now = utc_now()
            elapsed = (now - opened).total_seconds() / 3600
            remaining = (target - now).total_seconds() / 3600
            total_duration = elapsed + remaining
            progress = (elapsed / total_duration * 100) if total_duration > 0 else 0

        logger.info(f"  Time Elapsed: {elapsed:.2f}h ({progress:.1f}%)")
        logger.info(f"  Time Remaining: {remaining:.2f}h")

        # Show position sizing info
        sizing = pos.get('position_sizing', {})
        if sizing:
            configured = sizing.get('configured_notional', 0)
            actual = sizing.get('actual_notional', 0)
            was_limited = sizing.get('was_capital_limited', False)

            if was_limited:
                logger.info(f"  Position Size: {Colors.YELLOW}${actual:.2f}{Colors.RESET} (limited from ${configured:.2f})")
                logger.info(f"  Limited by: {Colors.YELLOW}{sizing.get('limiting_exchange', 'N/A')}{Colors.RESET}")
            else:
                logger.info(f"  Position Size: ${actual:.2f}")

        if pos.get('current_pnl'):
            pnl = pos['current_pnl']['total_unrealized_pnl']
            color = Colors.GREEN if pnl >= 0 else Colors.RED
            logger.info(f"  Unrealized PnL: {color}${pnl:+.4f}{Colors.RESET}")

    logger.info(f"\nCumulative Stats:")
    logger.info(f"  Total Cycles: {stats['total_cycles']}")
    if stats['total_cycles'] > 0:
        logger.info(f"  Success Rate: {(stats['successful_cycles'] / stats['total_cycles'] * 100):.1f}%")

        total_pnl_color = Colors.GREEN if stats['total_realized_pnl'] >= 0 else Colors.RED
        logger.info(f"  Total PnL: {total_pnl_color}${stats['total_realized_pnl']:+.2f}{Colors.RESET}")
        logger.info(f"  Avg PnL/Cycle: ${(stats['total_realized_pnl'] / stats['total_cycles']):+.4f}")

    logger.info(f"{'=' * 70}\n")
