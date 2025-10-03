#!/usr/bin/env python3
"""
auto_rotation_bot.py
--------------------
Automated delta-neutral position rotation bot.

This bot continuously:
1. Analyzes funding rates across multiple symbols
2. Opens the best delta-neutral position
3. Holds for 8 hours collecting funding
4. Closes the position
5. Waits briefly and repeats

Features:
- Persistent state across restarts
- Automatic recovery from crashes
- Comprehensive PnL tracking (trading, funding, fees)
- Health monitoring during hold period
- Graceful shutdown handling

Usage:
    python auto_rotation_bot.py
    python auto_rotation_bot.py --state-file custom_state.json
    python auto_rotation_bot.py --dry-run  # Test mode without trading
"""

import asyncio
import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
import websockets

# EdgeX and Lighter SDKs
from edgex_sdk import Client as EdgeXClient, OrderSide as EdgeXSide, OrderType as EdgeXType, TimeInForce as EdgeXTIF
import lighter

# Import helper functions from hedge_cli.py
# We'll reuse many functions rather than duplicating code
import hedge_cli

# ANSI color codes for console output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'

# ==================== Logging Setup ====================

os.makedirs('logs', exist_ok=True)

# File handler - DEBUG level
file_handler = logging.FileHandler('logs/auto_rotation_bot.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))

# Console handler - INFO level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

# Root logger
logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('lighter').setLevel(logging.WARNING)
logging.getLogger('hedge_cli').setLevel(logging.WARNING)

# ==================== State Management ====================

class BotState:
    """State machine for the rotation bot."""
    IDLE = "IDLE"
    ANALYZING = "ANALYZING"
    OPENING = "OPENING"
    HOLDING = "HOLDING"
    CLOSING = "CLOSING"
    WAITING = "WAITING"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


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
    stop_loss_percent: float = 2.0  # Stop-loss threshold as % of position notional
    enable_stop_loss: bool = True
    max_position_count: int = 1
    enable_pnl_tracking: bool = True
    enable_health_monitoring: bool = True

    @staticmethod
    def load_from_file(config_file: str) -> 'BotConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)

            # Provide defaults for any missing fields (backward compatibility)
            defaults = {
                'symbols_to_monitor': hedge_cli.CRYPTO_LIST,
                'quote': 'USD',
                'leverage': 3,
                'notional_per_position': 100.0,
                'hold_duration_hours': 8.0,
                'wait_between_cycles_minutes': 5.0,
                'check_interval_seconds': 60,
                'min_net_apr_threshold': 5.0,
                'stop_loss_percent': 2.0,
                'enable_stop_loss': True,
                'max_position_count': 1,
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
            return BotConfig(symbols_to_monitor=hedge_cli.CRYPTO_LIST)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return BotConfig(symbols_to_monitor=hedge_cli.CRYPTO_LIST)


class StateManager:
    """Manages bot state persistence and recovery."""

    def __init__(self, state_file: str = "bot_state.json"):
        self.state_file = state_file
        self.state = {
            "version": "1.0",
            "state": BotState.IDLE,
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
                "last_updated": None
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
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }

    def load(self) -> bool:
        """Load state from file. Returns True if loaded successfully."""
        if not os.path.exists(self.state_file):
            logger.info(f"No state file found at {self.state_file}, starting fresh")
            return False

        try:
            with open(self.state_file, 'r') as f:
                loaded_state = json.load(f)

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
                    "last_updated": None
                }

            logger.info(f"Loaded state from {self.state_file}")
            logger.info(f"Current state: {self.state['state']}")
            return True
        except Exception as e:
            logger.error(f"Failed to load state file: {e}")
            return False

    def save(self):
        """Save current state to file."""
        self.state["last_updated"] = datetime.utcnow().isoformat() + "Z"
        try:
            # Write to temp file first, then atomic rename
            temp_file = self.state_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            os.replace(temp_file, self.state_file)
            logger.debug(f"Saved state to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def set_state(self, new_state: str):
        """Update bot state."""
        logger.info(f"State transition: {self.state['state']} → {new_state}")
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


# ==================== Balance & Position Helpers ====================

async def get_edgex_balance(env: dict) -> Tuple[float, float]:
    """Get EdgeX total and available USD balance."""
    edgex = None
    try:
        edgex = EdgeXClient(
            base_url=env["EDGEX_BASE_URL"],
            account_id=int(env["EDGEX_ACCOUNT_ID"]),
            stark_private_key=env["EDGEX_STARK_PRIVATE_KEY"]
        )

        resp = await edgex.get_account_asset()
        data = resp.get("data", {})
        asset_list = data.get("collateralAssetModelList", [])

        total = 0.0
        available = 0.0
        for asset in asset_list:
            if str(asset.get("assetId", "")) == "1":  # USD
                total = float(asset.get("total", "0") or "0")
                available = float(asset.get("available", "0") or "0")
                break

        return total, available
    except Exception as e:
        logger.error(f"Error fetching EdgeX balance: {e}")
        return 0.0, 0.0
    finally:
        if edgex:
            await edgex.close()


async def get_lighter_balance(env: dict) -> Tuple[float, float]:
    """Get Lighter total and available USD balance via WebSocket."""
    try:
        account_index = int(env.get("ACCOUNT_INDEX", env.get("LIGHTER_ACCOUNT_INDEX", "0")))
        ws_url = env["LIGHTER_WS_URL"]

        async with websockets.connect(ws_url) as ws:
            # Subscribe to user stats
            subscribe_msg = {
                "method": "subscribe",
                "channel": f"user_stats/{account_index}"
            }
            await ws.send(json.dumps(subscribe_msg))

            # Wait for response
            for _ in range(10):  # Max 10 messages
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)

                if data.get("channel") == f"user_stats/{account_index}":
                    stats = data.get("data", {})
                    total = float(stats.get("total_value_usd", "0") or "0")
                    available = float(stats.get("available_balance_usd", "0") or "0")
                    return total, available

        return 0.0, 0.0
    except Exception as e:
        logger.error(f"Error fetching Lighter balance: {e}")
        return 0.0, 0.0


async def get_position_prices(env: dict, symbol: str, quote: str,
                              long_exchange: str, short_exchange: str,
                              e_contract_id: str, l_market_id: int) -> Dict:
    """Get current position prices from both exchanges."""
    prices = {
        "edgex_mark": None,
        "lighter_mark": None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    # EdgeX price
    edgex = None
    try:
        edgex = EdgeXClient(
            base_url=env["EDGEX_BASE_URL"],
            account_id=int(env["EDGEX_ACCOUNT_ID"]),
            stark_private_key=env["EDGEX_STARK_PRIVATE_KEY"]
        )
        quote_resp = await edgex.quote.get_24_hour_quote(e_contract_id)
        if quote_resp.get("code") == "SUCCESS" and quote_resp.get("data"):
            quote_data = quote_resp["data"][0]
            prices["edgex_mark"] = float(quote_data.get("lastPrice", "0") or "0")
    except Exception as e:
        logger.error(f"Error fetching EdgeX price: {e}")
    finally:
        if edgex:
            try:
                await edgex.close()
            except:
                pass

    # Lighter price
    lighter_api_client = None
    try:
        lighter_api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
        order_api = lighter.OrderApi(lighter_api_client)
        orderbook = await order_api.order_book(market_id=int(l_market_id), depth=1)

        if orderbook.bids and orderbook.asks:
            bid = float(orderbook.bids[0].price)
            ask = float(orderbook.asks[0].price)
            prices["lighter_mark"] = (bid + ask) / 2
    except Exception as e:
        logger.error(f"Error fetching Lighter price: {e}")
    finally:
        if lighter_api_client:
            try:
                await lighter_api_client.close()
            except:
                pass

    return prices


async def get_position_pnl(env: dict, symbol: str, quote: str,
                          long_exchange: str, short_exchange: str,
                          e_contract_id: str, l_market_id: int) -> Dict:
    """Get unrealized PnL from both exchanges."""
    pnl_data = {
        "edgex_unrealized_pnl": 0.0,
        "lighter_unrealized_pnl": 0.0,
        "total_unrealized_pnl": 0.0,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    # EdgeX PnL
    edgex = None
    try:
        edgex = EdgeXClient(
            base_url=env["EDGEX_BASE_URL"],
            account_id=int(env["EDGEX_ACCOUNT_ID"]),
            stark_private_key=env["EDGEX_STARK_PRIVATE_KEY"]
        )
        positions_resp = await edgex.get_account_positions()
        positions = positions_resp.get("data", {}).get("positionList", [])

        for pos in positions:
            if pos.get("contractId") == e_contract_id:
                pnl_data["edgex_unrealized_pnl"] = float(pos.get("unrealizedPnl", "0") or "0")
                break
    except Exception as e:
        logger.error(f"Error fetching EdgeX PnL: {e}")
    finally:
        if edgex:
            try:
                await edgex.close()
            except:
                pass

    # Lighter PnL
    lighter_api_client = None
    try:
        account_index = int(env.get("ACCOUNT_INDEX", env.get("LIGHTER_ACCOUNT_INDEX", "0")))
        lighter_api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
        account_api = lighter.AccountApi(lighter_api_client)
        account_details = await account_api.account(by="index", value=str(account_index))

        if account_details and account_details.accounts:
            acc = account_details.accounts[0]
            if acc.positions:
                for pos in acc.positions:
                    if pos.market_id == int(l_market_id):
                        pnl_data["lighter_unrealized_pnl"] = float(pos.unrealized_pnl or "0")
                        break
    except Exception as e:
        logger.error(f"Error fetching Lighter PnL: {e}")
    finally:
        if lighter_api_client:
            try:
                await lighter_api_client.close()
            except:
                pass

    pnl_data["total_unrealized_pnl"] = pnl_data["edgex_unrealized_pnl"] + pnl_data["lighter_unrealized_pnl"]
    return pnl_data


def check_stop_loss(pnl_data: Dict, notional: float, stop_loss_percent: float) -> Tuple[bool, str]:
    """
    Check if stop-loss condition is triggered.

    Stop-loss logic: If EITHER leg has negative PnL exceeding the threshold, trigger stop-loss.
    We check the WORST performing leg (most negative PnL).

    Args:
        pnl_data: Dictionary with edgex_unrealized_pnl and lighter_unrealized_pnl
        notional: Position notional size in USD
        stop_loss_percent: Stop-loss threshold as percentage (e.g., 2.0 = 2%)

    Returns:
        Tuple of (triggered: bool, reason: str)
    """
    edgex_pnl = pnl_data["edgex_unrealized_pnl"]
    lighter_pnl = pnl_data["lighter_unrealized_pnl"]

    # Calculate stop-loss threshold in absolute USD
    stop_loss_threshold = -abs(notional * (stop_loss_percent / 100.0))

    # Find the worst performing leg (most negative)
    worst_pnl = min(edgex_pnl, lighter_pnl)
    worst_exchange = "EdgeX" if edgex_pnl < lighter_pnl else "Lighter"

    # Trigger if worst leg exceeds threshold
    if worst_pnl < stop_loss_threshold:
        worst_pnl_percent = (worst_pnl / notional) * 100 if notional > 0 else 0
        reason = (f"{worst_exchange} leg hit stop-loss: "
                 f"${worst_pnl:.2f} ({worst_pnl_percent:.2f}%) "
                 f"exceeds threshold of ${stop_loss_threshold:.2f} ({-stop_loss_percent:.2f}%)")
        return True, reason

    return False, ""


async def get_available_capital_and_max_position(env: dict, config: BotConfig) -> Dict:
    """
    Get available capital on both exchanges and calculate maximum openable position.

    Returns:
        Dictionary with capital info and max position size
    """
    capital_info = {
        "edgex_total": 0.0,
        "edgex_available": 0.0,
        "lighter_total": 0.0,
        "lighter_available": 0.0,
        "total_capital": 0.0,
        "total_available": 0.0,
        "max_position_notional": 0.0,
        "limiting_exchange": None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    # Get EdgeX balance
    edgex_total, edgex_avail = await get_edgex_balance(env)
    capital_info["edgex_total"] = edgex_total
    capital_info["edgex_available"] = edgex_avail

    # Get Lighter balance
    lighter_total, lighter_avail = await get_lighter_balance(env)
    capital_info["lighter_total"] = lighter_total
    capital_info["lighter_available"] = lighter_avail

    # Calculate totals
    capital_info["total_capital"] = edgex_total + lighter_total
    capital_info["total_available"] = edgex_avail + lighter_avail

    # Calculate max position size (minimum of both exchanges, accounting for leverage)
    # Max position = min(available_on_each_exchange) * leverage
    # We take the minimum because we need to open positions on BOTH exchanges
    max_edgex_position = edgex_avail * config.leverage if edgex_avail > 0 else 0
    max_lighter_position = lighter_avail * config.leverage if lighter_avail > 0 else 0

    capital_info["max_position_notional"] = min(max_edgex_position, max_lighter_position)
    capital_info["limiting_exchange"] = "EdgeX" if max_edgex_position < max_lighter_position else "Lighter"

    return capital_info


# ==================== Position Opening/Closing ====================

async def open_best_position(state_mgr: StateManager, env: dict, config: BotConfig, dry_run: bool = False):
    """Analyze funding rates and open the best delta-neutral position."""

    logger.info(f"Analyzing funding rates for {len(config.symbols_to_monitor)} symbols...")
    state_mgr.set_state(BotState.ANALYZING)

    # Fetch funding rates for all symbols
    results = await asyncio.gather(*[
        hedge_cli.fetch_symbol_funding(symbol, config.quote, env)
        for symbol in config.symbols_to_monitor
    ])

    # Filter available symbols and sort by net APR
    available = [r for r in results if r.get("available", False)]

    if not available:
        logger.error("No symbols available on both exchanges!")
        state_mgr.set_state(BotState.ERROR)
        state_mgr.state["cumulative_stats"]["last_error"] = "No symbols available"
        state_mgr.state["cumulative_stats"]["last_error_at"] = datetime.utcnow().isoformat() + "Z"
        state_mgr.save()
        return False

    # Sort by net APR descending
    available.sort(key=lambda x: x["net_apr"], reverse=True)

    # Filter by minimum APR threshold
    candidates = [r for r in available if r["net_apr"] >= config.min_net_apr_threshold]

    if not candidates:
        best_available = available[0] if available else None
        if best_available:
            logger.warning(f"Best opportunity ({best_available['symbol']}: {best_available['net_apr']:.2f}%) below threshold ({config.min_net_apr_threshold}%)")
        logger.info("Waiting for better opportunities...")
        state_mgr.set_state(BotState.WAITING)
        return False

    # Get balances once before trying candidates
    logger.info("\nCapturing balance snapshots...")
    edgex_total_before, edgex_avail_before = await get_edgex_balance(env)
    lighter_total_before, lighter_avail_before = await get_lighter_balance(env)

    logger.info(f"EdgeX balance: ${edgex_total_before:.2f} (available: ${edgex_avail_before:.2f})")
    logger.info(f"Lighter balance: ${lighter_total_before:.2f} (available: ${lighter_avail_before:.2f})")

    # Try candidates in order of best APR until one succeeds
    for idx, candidate in enumerate(candidates):
        logger.info(f"\n{Colors.CYAN}{'═' * 70}")
        logger.info(f"TRYING OPPORTUNITY #{idx + 1}: {candidate['symbol']}")
        logger.info(f"{'═' * 70}{Colors.RESET}")
        logger.info(f"Net APR: {Colors.GREEN}{candidate['net_apr']:.2f}%{Colors.RESET}")
        logger.info(f"Setup: Long {candidate['long_exch']}, Short {candidate['short_exch']}")
        logger.info(f"EdgeX APR: {candidate['edgex_apr']:.2f}%, Lighter APR: {candidate['lighter_apr']:.2f}%")

        if dry_run:
            logger.info(f"{Colors.YELLOW}DRY RUN MODE: Would open position on {candidate['symbol']}{Colors.RESET}")
            return True

        # Try to open this candidate
        result = await _try_open_position(state_mgr, env, config, candidate, edgex_total_before, edgex_avail_before, lighter_total_before, lighter_avail_before)
        if result:
            return True  # Success!

        # This candidate failed (insufficient capital or minimum size), try next
        if idx < len(candidates) - 1:
            logger.info(f"{Colors.YELLOW}Trying next candidate...{Colors.RESET}\n")

    # All candidates failed
    logger.error(f"{Colors.RED}All {len(candidates)} candidates failed to open. Insufficient capital or minimum size issues.{Colors.RESET}")
    state_mgr.set_state(BotState.WAITING)
    return False


async def _try_open_position(state_mgr: StateManager, env: dict, config: BotConfig, best: dict,
                             edgex_total_before: float, edgex_avail_before: float,
                             lighter_total_before: float, lighter_avail_before: float) -> bool:
    """Try to open a position for a specific symbol. Returns True if successful, False if should try next."""

    # Calculate maximum position size based on available capital with leverage
    # We need to use the MINIMUM of both exchanges to maintain delta-neutral
    max_edgex_notional = edgex_avail_before * config.leverage
    max_lighter_notional = lighter_avail_before * config.leverage
    max_available_notional = min(max_edgex_notional, max_lighter_notional)

    # Use the smaller of configured notional or max available (with 5% safety buffer)
    actual_notional = min(config.notional_per_position, max_available_notional * 0.95)

    # Check if we have enough capital to open a meaningful position
    if actual_notional < 10.0:  # Minimum $10 position
        logger.error(f"{Colors.RED}Insufficient capital to open position!{Colors.RESET}")
        logger.error(f"Max available notional: ${max_available_notional:.2f}")
        logger.error(f"EdgeX max: ${max_edgex_notional:.2f}, Lighter max: ${max_lighter_notional:.2f}")
        state_mgr.set_state(BotState.ERROR)
        state_mgr.state["cumulative_stats"]["last_error"] = "Insufficient capital"
        state_mgr.state["cumulative_stats"]["last_error_at"] = datetime.utcnow().isoformat() + "Z"
        state_mgr.save()
        return False

    # Log position sizing info
    if actual_notional < config.notional_per_position:
        logger.warning(f"{Colors.YELLOW}Reducing position size due to available capital:{Colors.RESET}")
        logger.warning(f"  Configured: ${config.notional_per_position:.2f}")
        logger.warning(f"  Actual:     ${actual_notional:.2f}")
        limiting_exchange = "EdgeX" if max_edgex_notional < max_lighter_notional else "Lighter"
        logger.warning(f"  Limited by: {limiting_exchange}")
    else:
        logger.info(f"Position size: ${actual_notional:.2f}")

    # Create config for opening
    app_config = hedge_cli.AppConfig(
        symbol=best['symbol'],
        quote=config.quote,
        long_exchange=best['long_exch'].lower(),
        short_exchange=best['short_exch'].lower(),
        leverage=config.leverage,
        notional=actual_notional  # Use calculated notional instead of config value
    )

    # Open position
    logger.info(f"\n{Colors.CYAN}Opening delta-neutral position...{Colors.RESET}")
    state_mgr.set_state(BotState.OPENING)

    edgex = None
    lighter_api_client = None
    edgex2 = None
    lighter_api_client2 = None

    try:
        # Get market details first
        edgex = EdgeXClient(
            base_url=env["EDGEX_BASE_URL"],
            account_id=int(env["EDGEX_ACCOUNT_ID"]),
            stark_private_key=env["EDGEX_STARK_PRIVATE_KEY"]
        )
        contract_name = f"{best['symbol'].upper()}{config.quote.upper()}"
        e_contract_id, e_tick_price, e_tick_size = await hedge_cli.edgex_find_contract_id(edgex, contract_name)

        lighter_api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
        order_api = lighter.OrderApi(lighter_api_client)
        l_market_id, l_tick_price, l_tick_size = await hedge_cli.lighter_get_market_details(order_api, best['symbol'])

        # Get entry prices
        edgex_bid, edgex_ask = await hedge_cli.edgex_get_prices(edgex, e_contract_id)
        lighter_orderbook = await order_api.order_book(market_id=int(l_market_id), depth=1)
        lighter_bid = float(lighter_orderbook.bids[0].price) if lighter_orderbook.bids else None
        lighter_ask = float(lighter_orderbook.asks[0].price) if lighter_orderbook.asks else None

        edgex_mid = (edgex_bid + edgex_ask) / 2 if edgex_bid and edgex_ask else None
        lighter_mid = (lighter_bid + lighter_ask) / 2 if lighter_bid and lighter_ask else None

        # Calculate minimum position size based on tick sizes
        # Use coarser tick size (larger of the two) to ensure both exchanges can handle it
        coarser_tick_size = max(e_tick_size, l_tick_size)

        # Estimate minimum notional (typically exchanges require at least 1 unit of the coarser tick)
        # We'll use a conservative estimate: 10 units of the coarser tick size
        min_position_base = coarser_tick_size * 10

        # Calculate minimum notional in quote currency
        price_estimate = edgex_mid if edgex_mid else (lighter_mid if lighter_mid else 1.0)
        min_notional_required = min_position_base * price_estimate

        # Check if our actual_notional meets the minimum requirement
        if actual_notional < min_notional_required:
            logger.error(f"{Colors.RED}Position size too small for {best['symbol']}!{Colors.RESET}")
            logger.error(f"  Required minimum: ${min_notional_required:.2f}")
            logger.error(f"  Available:        ${actual_notional:.2f}")
            logger.error(f"  Coarser tick:     {coarser_tick_size:.8f} (EdgeX: {e_tick_size:.8f}, Lighter: {l_tick_size:.8f})")
            logger.warning(f"{Colors.YELLOW}Skipping {best['symbol']}, will try again next cycle...{Colors.RESET}")

            # Close clients
            if edgex:
                await edgex.close()
                edgex = None
            if lighter_api_client:
                await lighter_api_client.close()
                lighter_api_client = None

            # Return False but don't set ERROR state - just skip this opportunity
            return False

        logger.info(f"✓ Minimum check passed: ${actual_notional:.2f} >= ${min_notional_required:.2f}")

        # Close initial clients before opening position
        if edgex:
            await edgex.close()
            edgex = None
        if lighter_api_client:
            await lighter_api_client.close()
            lighter_api_client = None

        # Open the position using hedge_cli with calculated notional
        await hedge_cli.open_hedge(app_config, env, size_base=None, size_quote=actual_notional, cross_ticks=10)

        # Get position sizes after opening
        await asyncio.sleep(2)  # Brief delay for settlement

        edgex2 = EdgeXClient(
            base_url=env["EDGEX_BASE_URL"],
            account_id=int(env["EDGEX_ACCOUNT_ID"]),
            stark_private_key=env["EDGEX_STARK_PRIVATE_KEY"]
        )
        edgex_size = await hedge_cli.edgex_get_open_size(edgex2, e_contract_id)

        lighter_api_client2 = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
        account_api = lighter.AccountApi(lighter_api_client2)
        account_index = int(env.get("ACCOUNT_INDEX", env.get("LIGHTER_ACCOUNT_INDEX", "0")))
        lighter_size = await hedge_cli.lighter_get_open_size(account_api, account_index, int(l_market_id))

        logger.info(f"{Colors.GREEN}Position opened successfully!{Colors.RESET}")
        logger.info(f"EdgeX size: {edgex_size}")
        logger.info(f"Lighter size: {lighter_size}")

        # Record position details
        opened_at = datetime.utcnow()
        target_close_at = opened_at + timedelta(hours=config.hold_duration_hours)

        state_mgr.state["current_position"] = {
            "symbol": best['symbol'],
            "quote": config.quote,
            "long_exchange": best['long_exch'].lower(),
            "short_exchange": best['short_exch'].lower(),
            "leverage": config.leverage,
            "opened_at": opened_at.isoformat() + "Z",
            "target_close_at": target_close_at.isoformat() + "Z",

            "position_sizing": {
                "configured_notional": config.notional_per_position,
                "actual_notional": actual_notional,
                "max_edgex_notional": max_edgex_notional,
                "max_lighter_notional": max_lighter_notional,
                "limiting_exchange": "EdgeX" if max_edgex_notional < max_lighter_notional else "Lighter",
                "was_capital_limited": actual_notional < config.notional_per_position,
                "edgex_available_at_open": edgex_avail_before,
                "lighter_available_at_open": lighter_avail_before
            },

            "entry": {
                "edgex_contract_id": e_contract_id,
                "lighter_market_id": str(l_market_id),
                "edgex_entry_price": edgex_mid,
                "lighter_entry_price": lighter_mid,
                "size_base": abs(edgex_size),
                "edgex_size": edgex_size,
                "lighter_size": lighter_size,
                "edgex_balance_before": edgex_total_before,
                "lighter_balance_before": lighter_total_before,
                "timestamp": opened_at.isoformat() + "Z"
            },

            "expected_funding": {
                "edgex_rate_per_period": best['edgex_rate'],
                "lighter_rate_per_period": best['lighter_rate'],
                "net_apr": best['net_apr'],
                "expected_duration_hours": config.hold_duration_hours
            },

            "current_pnl": {
                "edgex_unrealized_pnl": 0.0,
                "lighter_unrealized_pnl": 0.0,
                "total_unrealized_pnl": 0.0,
                "last_updated": opened_at.isoformat() + "Z"
            }
        }

        state_mgr.set_state(BotState.HOLDING)
        logger.info(f"Now holding until {target_close_at.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        return True

    except Exception as e:
        logger.error(f"{Colors.RED}Failed to open position: {e}{Colors.RESET}", exc_info=True)
        state_mgr.set_state(BotState.ERROR)
        state_mgr.state["cumulative_stats"]["last_error"] = str(e)
        state_mgr.state["cumulative_stats"]["last_error_at"] = datetime.utcnow().isoformat() + "Z"
        state_mgr.save()
        return False
    finally:
        # Ensure all clients are closed
        if edgex:
            try:
                await edgex.close()
            except:
                pass
        if lighter_api_client:
            try:
                await lighter_api_client.close()
            except:
                pass
        if edgex2:
            try:
                await edgex2.close()
            except:
                pass
        if lighter_api_client2:
            try:
                await lighter_api_client2.close()
            except:
                pass


async def close_current_position(state_mgr: StateManager, env: dict, dry_run: bool = False):
    """Close the current delta-neutral position and record PnL."""

    pos = state_mgr.state["current_position"]
    if not pos:
        logger.warning("No current position to close")
        return False

    logger.info(f"\n{Colors.CYAN}{'═' * 70}")
    logger.info(f"CLOSING POSITION: {pos['symbol']}")
    logger.info(f"{'═' * 70}{Colors.RESET}")

    if dry_run:
        logger.info(f"{Colors.YELLOW}DRY RUN MODE: Would close position{Colors.RESET}")
        return True

    state_mgr.set_state(BotState.CLOSING)

    try:
        # Get balances and prices before closing
        edgex_total_before_close, _ = await get_edgex_balance(env)
        lighter_total_before_close, _ = await get_lighter_balance(env)

        prices_before = await get_position_prices(
            env, pos['symbol'], pos['quote'],
            pos['long_exchange'], pos['short_exchange'],
            pos['entry']['edgex_contract_id'],
            pos['entry']['lighter_market_id']
        )

        pnl_before = await get_position_pnl(
            env, pos['symbol'], pos['quote'],
            pos['long_exchange'], pos['short_exchange'],
            pos['entry']['edgex_contract_id'],
            pos['entry']['lighter_market_id']
        )

        logger.info(f"Unrealized PnL before close:")
        logger.info(f"  EdgeX: ${pnl_before['edgex_unrealized_pnl']:.4f}")
        logger.info(f"  Lighter: ${pnl_before['lighter_unrealized_pnl']:.4f}")
        logger.info(f"  Total: ${pnl_before['total_unrealized_pnl']:.4f}")

        # Create config for closing
        app_config = hedge_cli.AppConfig(
            symbol=pos['symbol'],
            quote=pos['quote'],
            long_exchange=pos['long_exchange'],
            short_exchange=pos['short_exchange'],
            leverage=pos['leverage']
        )

        # Close position
        await hedge_cli.close_both(app_config, env, cross_ticks=10)

        # Wait for settlement
        await asyncio.sleep(2)

        # Get balances after closing
        edgex_total_after, _ = await get_edgex_balance(env)
        lighter_total_after, _ = await get_lighter_balance(env)

        closed_at = datetime.utcnow()

        logger.info(f"{Colors.GREEN}Position closed successfully!{Colors.RESET}")

        # Calculate PnL breakdown
        edgex_balance_change = edgex_total_after - pos['entry']['edgex_balance_before']
        lighter_balance_change = lighter_total_after - pos['entry']['lighter_balance_before']
        total_balance_change = edgex_balance_change + lighter_balance_change

        logger.info(f"\nBalance changes:")
        logger.info(f"  EdgeX: ${edgex_balance_change:+.4f}")
        logger.info(f"  Lighter: ${lighter_balance_change:+.4f}")
        logger.info(f"  Total: {Colors.GREEN if total_balance_change >= 0 else Colors.RED}${total_balance_change:+.4f}{Colors.RESET}")

        # Calculate hold duration
        opened_at = datetime.fromisoformat(pos['opened_at'].replace('Z', '+00:00'))
        hold_duration = (closed_at - opened_at.replace(tzinfo=None)).total_seconds() / 3600

        # Record completed cycle
        cycle_record = {
            "cycle_number": state_mgr.state["cumulative_stats"]["total_cycles"] + 1,
            "symbol": pos['symbol'],
            "opened_at": pos['opened_at'],
            "closed_at": closed_at.isoformat() + "Z",
            "duration_hours": hold_duration,

            "entry": {
                "long_exchange": pos['long_exchange'],
                "short_exchange": pos['short_exchange'],
                "edgex_price": pos['entry']['edgex_entry_price'],
                "lighter_price": pos['entry']['lighter_entry_price'],
                "size_base": pos['entry']['size_base'],
                "edgex_balance_before": pos['entry']['edgex_balance_before'],
                "lighter_balance_before": pos['entry']['lighter_balance_before']
            },

            "exit": {
                "edgex_price": prices_before['edgex_mark'],
                "lighter_price": prices_before['lighter_mark'],
                "edgex_balance_after": edgex_total_after,
                "lighter_balance_after": lighter_total_after,
                "timestamp": closed_at.isoformat() + "Z"
            },

            "pnl_breakdown": {
                "edgex_balance_change": edgex_balance_change,
                "lighter_balance_change": lighter_balance_change,
                "total_realized_pnl": total_balance_change,
                "realized_pnl_percent": (total_balance_change / (pos['entry']['edgex_balance_before'] + pos['entry']['lighter_balance_before'])) * 100 if (pos['entry']['edgex_balance_before'] + pos['entry']['lighter_balance_before']) > 0 else 0
            },

            "funding_stats": {
                "expected_net_apr": pos['expected_funding']['net_apr'],
                "hold_duration_hours": hold_duration
            },

            "position_sizing": pos.get('position_sizing', {
                "configured_notional": 0,
                "actual_notional": 0,
                "was_capital_limited": False
            }),

            "stop_loss_triggered": pos.get('stop_loss_triggered', False),
            "stop_loss_reason": pos.get('stop_loss_reason', None)
        }

        # Update cumulative stats
        stats = state_mgr.state["cumulative_stats"]
        stats["total_cycles"] += 1
        stats["successful_cycles"] += 1
        stats["total_realized_pnl"] += total_balance_change
        stats["total_volume_traded"] += pos['entry']['size_base'] * 2  # Both legs
        stats["total_hold_time_hours"] += hold_duration

        # Update best/worst cycle PnL (check if first cycle or if value is better/worse)
        if stats["total_cycles"] == 1 or total_balance_change > stats["best_cycle_pnl"]:
            stats["best_cycle_pnl"] = total_balance_change
        if stats["total_cycles"] == 1 or total_balance_change < stats["worst_cycle_pnl"]:
            stats["worst_cycle_pnl"] = total_balance_change

        # Update by-symbol stats
        if pos['symbol'] not in stats["by_symbol"]:
            stats["by_symbol"][pos['symbol']] = {
                "cycles": 0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0
            }

        symbol_stats = stats["by_symbol"][pos['symbol']]
        symbol_stats["cycles"] += 1
        symbol_stats["total_pnl"] += total_balance_change
        symbol_stats["avg_pnl"] = symbol_stats["total_pnl"] / symbol_stats["cycles"]

        # Add to completed cycles
        state_mgr.state["completed_cycles"].append(cycle_record)

        # Keep only last 100 cycles to prevent file from growing too large
        if len(state_mgr.state["completed_cycles"]) > 100:
            state_mgr.state["completed_cycles"] = state_mgr.state["completed_cycles"][-100:]

        # Clear current position
        state_mgr.state["current_position"] = None
        state_mgr.save()

        # Display summary
        display_cycle_summary(cycle_record, stats)

        return True

    except Exception as e:
        logger.error(f"{Colors.RED}Failed to close position: {e}{Colors.RESET}", exc_info=True)
        state_mgr.set_state(BotState.ERROR)
        state_mgr.state["cumulative_stats"]["last_error"] = str(e)
        state_mgr.state["cumulative_stats"]["last_error_at"] = datetime.utcnow().isoformat() + "Z"
        state_mgr.state["cumulative_stats"]["failed_cycles"] += 1
        state_mgr.save()
        return False


# ==================== Display Functions ====================

def display_cycle_summary(cycle: dict, cumulative_stats: dict):
    """Display summary of completed cycle."""
    logger.info(f"\n{Colors.CYAN}{'═' * 70}")
    logger.info(f"CYCLE #{cycle['cycle_number']} COMPLETE - {cycle['symbol']}")
    logger.info(f"{'═' * 70}{Colors.RESET}")

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
    logger.info(f"{'═' * 70}\n")


def display_status(state_mgr: StateManager):
    """Display current bot status."""
    state = state_mgr.get_state()
    pos = state_mgr.state.get("current_position")
    stats = state_mgr.state["cumulative_stats"]
    capital = state_mgr.state.get("capital_status", {})

    logger.info(f"\n{Colors.BOLD}{'═' * 70}")
    logger.info(f"BOT STATUS")
    logger.info(f"{'═' * 70}{Colors.RESET}")
    logger.info(f"State: {Colors.CYAN}{state}{Colors.RESET}")

    # Display capital status
    if capital and capital.get("last_updated"):
        logger.info(f"\nCapital Status:")
        logger.info(f"  EdgeX:   ${capital['edgex_total']:.2f} total, ${capital['edgex_available']:.2f} available")
        logger.info(f"  Lighter: ${capital['lighter_total']:.2f} total, ${capital['lighter_available']:.2f} available")
        logger.info(f"  {Colors.BOLD}Total:   ${capital['total_capital']:.2f} total, ${capital['total_available']:.2f} available{Colors.RESET}")

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
            opened = datetime.fromisoformat(pos['opened_at'].replace('Z', '+00:00'))
            target = datetime.fromisoformat(pos['target_close_at'].replace('Z', '+00:00'))
            now = datetime.utcnow()
            elapsed = (now - opened.replace(tzinfo=None)).total_seconds() / 3600
            remaining = (target.replace(tzinfo=None) - now).total_seconds() / 3600
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

    logger.info(f"{'═' * 70}\n")


# ==================== State Recovery ====================

async def recover_state(state_mgr: StateManager, env: dict) -> bool:
    """Recover bot state by checking actual positions on exchanges."""

    logger.info(f"{Colors.YELLOW}Performing state recovery...{Colors.RESET}")

    state = state_mgr.get_state()
    pos = state_mgr.state.get("current_position")

    # If state is IDLE, verify no positions exist
    if state == BotState.IDLE:
        logger.info("State is IDLE, verifying no positions exist...")
        # We could check all monitored symbols here, but for now trust the state
        logger.info(f"{Colors.GREEN}Recovery complete: IDLE state verified{Colors.RESET}")
        return True

    # If state is HOLDING, verify positions exist
    if state == BotState.HOLDING and pos:
        logger.info(f"State is HOLDING for {pos['symbol']}, verifying positions...")

        edgex = None
        lighter_api_client = None
        try:
            # Check EdgeX
            edgex = EdgeXClient(
                base_url=env["EDGEX_BASE_URL"],
                account_id=int(env["EDGEX_ACCOUNT_ID"]),
                stark_private_key=env["EDGEX_STARK_PRIVATE_KEY"]
            )
            edgex_size = await hedge_cli.edgex_get_open_size(edgex, pos['entry']['edgex_contract_id'])

            # Check Lighter
            lighter_api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
            account_api = lighter.AccountApi(lighter_api_client)
            account_index = int(env.get("ACCOUNT_INDEX", env.get("LIGHTER_ACCOUNT_INDEX", "0")))
            lighter_size = await hedge_cli.lighter_get_open_size(account_api, account_index, int(pos['entry']['lighter_market_id']))

            logger.info(f"Found positions: EdgeX={edgex_size}, Lighter={lighter_size}")

            # Verify hedge is still valid
            if abs(abs(edgex_size) - abs(lighter_size)) > 0.01:
                logger.error(f"{Colors.RED}Position size mismatch detected! Hedge may be broken.{Colors.RESET}")
                state_mgr.set_state(BotState.ERROR)
                state_mgr.state["cumulative_stats"]["last_error"] = f"Position size mismatch: EdgeX={edgex_size}, Lighter={lighter_size}"
                state_mgr.state["cumulative_stats"]["last_error_at"] = datetime.utcnow().isoformat() + "Z"
                state_mgr.save()
                return False

            if edgex_size == 0 and lighter_size == 0:
                logger.warning(f"{Colors.YELLOW}State says HOLDING but no positions found. Setting to IDLE.{Colors.RESET}")
                state_mgr.state["current_position"] = None
                state_mgr.set_state(BotState.IDLE)
                return True

            logger.info(f"{Colors.GREEN}Recovery complete: Positions verified{Colors.RESET}")
            return True

        except Exception as e:
            logger.error(f"Error during recovery: {e}")
            state_mgr.set_state(BotState.ERROR)
            state_mgr.state["cumulative_stats"]["last_error"] = f"Recovery failed: {str(e)}"
            state_mgr.state["cumulative_stats"]["last_error_at"] = datetime.utcnow().isoformat() + "Z"
            state_mgr.save()
            return False
        finally:
            if edgex:
                try:
                    await edgex.close()
                except:
                    pass
            if lighter_api_client:
                try:
                    await lighter_api_client.close()
                except:
                    pass

    # If state is OPENING, CLOSING, or ERROR, require manual intervention
    if state in [BotState.OPENING, BotState.CLOSING, BotState.ERROR]:
        logger.error(f"{Colors.RED}Bot is in {state} state. Manual intervention required.{Colors.RESET}")
        logger.error("Please check positions on both exchanges and either:")
        logger.error("  1. Manually close any open positions and delete bot_state.json")
        logger.error("  2. Fix the state file if positions are correct")
        return False

    # Default: return to IDLE
    logger.warning(f"{Colors.YELLOW}Unknown state during recovery. Resetting to IDLE.{Colors.RESET}")
    state_mgr.set_state(BotState.IDLE)
    state_mgr.state["current_position"] = None
    state_mgr.save()
    return True


# ==================== Main Bot Loop ====================

class RotationBot:
    """Main rotation bot class."""

    def __init__(self, state_file: str = "bot_state.json", config_file: str = "rotation_bot_config.json", dry_run: bool = False):
        self.state_mgr = StateManager(state_file)
        self.config_file = config_file
        self.env = hedge_cli.load_env()
        self.dry_run = dry_run
        self.shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"\n{Colors.YELLOW}Shutdown signal received. Stopping after current operation...{Colors.RESET}")
        self.shutdown_requested = True

    async def run(self):
        """Main bot loop."""

        logger.info(f"{Colors.BOLD}{Colors.CYAN}{'═' * 70}")
        logger.info(f"AUTOMATED DELTA-NEUTRAL ROTATION BOT")
        logger.info(f"{'═' * 70}{Colors.RESET}")

        if self.dry_run:
            logger.info(f"{Colors.YELLOW}Running in DRY RUN mode (no actual trading){Colors.RESET}\n")

        # Load or create state
        self.state_mgr.load()

        # Load config from file
        config = BotConfig.load_from_file(self.config_file)
        logger.info(f"Loaded configuration from {self.config_file}")

        logger.info(f"Monitoring symbols: {', '.join(config.symbols_to_monitor)}")
        logger.info(f"Max position size: ${config.notional_per_position} (auto-adjusts to available capital)")
        logger.info(f"Hold duration: {config.hold_duration_hours} hours")
        logger.info(f"Leverage: {config.leverage}x\n")

        # Recover state
        if not await recover_state(self.state_mgr, self.env):
            logger.error(f"{Colors.RED}State recovery failed. Exiting.{Colors.RESET}")
            return

        # Main loop
        while not self.shutdown_requested:
            try:
                state = self.state_mgr.get_state()

                # IDLE -> Start new cycle
                if state == BotState.IDLE:
                    # Update capital status before displaying
                    capital_info = await get_available_capital_and_max_position(self.env, config)
                    self.state_mgr.state["capital_status"] = capital_info
                    self.state_mgr.save()

                    display_status(self.state_mgr)
                    success = await open_best_position(self.state_mgr, self.env, config, self.dry_run)
                    if not success:
                        logger.error("Failed to open position. Waiting 5 minutes before retry...")
                        await asyncio.sleep(300)
                        continue

                # HOLDING -> Monitor position
                elif state == BotState.HOLDING:
                    pos = self.state_mgr.state["current_position"]

                    # Safety check: if no position exists, return to IDLE
                    if not pos:
                        logger.warning(f"{Colors.YELLOW}State is HOLDING but no position found. Returning to IDLE.{Colors.RESET}")
                        self.state_mgr.set_state(BotState.IDLE)
                        continue

                    # Check if it's time to close
                    target_close = datetime.fromisoformat(pos['target_close_at'].replace('Z', '+00:00'))
                    now = datetime.utcnow()

                    if now >= target_close.replace(tzinfo=None):
                        # Time to close
                        success = await close_current_position(self.state_mgr, self.env, self.dry_run)
                        if success:
                            self.state_mgr.set_state(BotState.WAITING)
                        else:
                            logger.error("Failed to close position. Manual intervention required.")
                            break
                    else:
                        # Still holding, update PnL
                        # Calculate elapsed and remaining time first
                        opened = datetime.fromisoformat(pos['opened_at'].replace('Z', '+00:00'))
                        elapsed = (now - opened.replace(tzinfo=None)).total_seconds() / 3600
                        remaining = (target_close.replace(tzinfo=None) - now).total_seconds() / 3600
                        total_duration = config.hold_duration_hours
                        progress_percent = (elapsed / total_duration * 100) if total_duration > 0 else 0

                        # Get current PnL
                        pnl_data = await get_position_pnl(
                            self.env, pos['symbol'], pos['quote'],
                            pos['long_exchange'], pos['short_exchange'],
                            pos['entry']['edgex_contract_id'],
                            pos['entry']['lighter_market_id']
                        )

                        # Update position with timing info and PnL
                        pos['current_pnl'] = pnl_data
                        pos['time_elapsed_hours'] = elapsed
                        pos['time_remaining_hours'] = remaining
                        pos['progress_percent'] = progress_percent
                        pos['last_check'] = datetime.utcnow().isoformat() + "Z"
                        self.state_mgr.save()

                        # Check stop-loss if enabled
                        if config.enable_stop_loss:
                            # Calculate notional from actual position size
                            size_base = pos['entry'].get('size_base', 0)
                            entry_price = pos['entry'].get('edgex_entry_price')

                            if size_base and entry_price:
                                notional = size_base * entry_price
                            else:
                                # Fallback to actual notional used (not configured max)
                                sizing = pos.get('position_sizing', {})
                                notional = sizing.get('actual_notional', config.notional_per_position)

                            stop_loss_triggered, stop_loss_reason = check_stop_loss(
                                pnl_data, notional, config.stop_loss_percent
                            )

                            if stop_loss_triggered:
                                logger.warning(f"\n{Colors.RED}{'!' * 70}")
                                logger.warning(f"STOP-LOSS TRIGGERED!")
                                logger.warning(f"{'!' * 70}{Colors.RESET}")
                                logger.warning(f"Reason: {stop_loss_reason}")
                                logger.warning(f"EdgeX PnL: ${pnl_data['edgex_unrealized_pnl']:+.4f}")
                                logger.warning(f"Lighter PnL: ${pnl_data['lighter_unrealized_pnl']:+.4f}")
                                logger.warning(f"Total PnL: ${pnl_data['total_unrealized_pnl']:+.4f}")
                                logger.warning(f"Closing position immediately...\n")

                                # Record stop-loss event in position data
                                pos['stop_loss_triggered'] = True
                                pos['stop_loss_reason'] = stop_loss_reason
                                pos['stop_loss_time'] = datetime.utcnow().isoformat() + "Z"
                                self.state_mgr.save()

                                # Emergency close
                                success = await close_current_position(self.state_mgr, self.env, self.dry_run)
                                if success:
                                    logger.info(f"{Colors.YELLOW}Position closed due to stop-loss. "
                                              f"Waiting before next cycle...{Colors.RESET}")
                                    self.state_mgr.set_state(BotState.WAITING)
                                else:
                                    logger.error("Failed to close position after stop-loss. Manual intervention required.")
                                    break
                                continue

                        # Update capital status periodically
                        capital_info = await get_available_capital_and_max_position(self.env, config)
                        self.state_mgr.state["capital_status"] = capital_info
                        self.state_mgr.save()

                        # Display PnL and capital status
                        pnl_color = Colors.GREEN if pnl_data['total_unrealized_pnl'] >= 0 else Colors.RED
                        edgex_color = Colors.GREEN if pnl_data['edgex_unrealized_pnl'] >= 0 else Colors.RED
                        lighter_color = Colors.GREEN if pnl_data['lighter_unrealized_pnl'] >= 0 else Colors.RED

                        # Get position sizing info
                        sizing = pos.get('position_sizing', {})

                        logger.info(f"\n{Colors.CYAN}{'─' * 70}{Colors.RESET}")
                        logger.info(f"{Colors.BOLD}HOLDING {pos['symbol']} - Delta Neutral Position{Colors.RESET}")
                        logger.info(f"{Colors.CYAN}{'─' * 70}{Colors.RESET}")
                        logger.info(f"⏱  Time Elapsed:  {Colors.CYAN}{elapsed:.2f}h{Colors.RESET} / {total_duration:.2f}h ({progress_percent:.1f}%)")
                        logger.info(f"⏰  Time Remaining: {Colors.YELLOW}{remaining:.2f}h{Colors.RESET} until close")

                        # Show position sizing info if available
                        if sizing:
                            configured = sizing.get('configured_notional', 0)
                            actual = sizing.get('actual_notional', 0)
                            was_limited = sizing.get('was_capital_limited', False)

                            if was_limited:
                                logger.info(f"💼  Position Size: {Colors.YELLOW}${actual:.2f}{Colors.RESET} (limited from ${configured:.2f})")
                                logger.info(f"    Limited by:   {Colors.YELLOW}{sizing.get('limiting_exchange', 'N/A')}{Colors.RESET}")
                            else:
                                logger.info(f"💼  Position Size: ${actual:.2f}")

                        logger.info(f"📈  EdgeX PnL:     {edgex_color}${pnl_data['edgex_unrealized_pnl']:+.4f}{Colors.RESET}")
                        logger.info(f"📉  Lighter PnL:   {lighter_color}${pnl_data['lighter_unrealized_pnl']:+.4f}{Colors.RESET}")
                        logger.info(f"💰  Total PnL:     {pnl_color}${pnl_data['total_unrealized_pnl']:+.4f}{Colors.RESET}")
                        logger.info(f"💵  Available:     EdgeX ${capital_info['edgex_available']:.2f}, Lighter ${capital_info['lighter_available']:.2f}")
                        logger.info(f"📊  Max Position:  ${capital_info['max_position_notional']:.2f} (limited by {capital_info['limiting_exchange']})")
                        logger.info(f"{Colors.CYAN}{'─' * 70}{Colors.RESET}\n")

                        # Sleep until next check
                        await asyncio.sleep(config.check_interval_seconds)

                # WAITING -> Cooldown between cycles
                elif state == BotState.WAITING:
                    logger.info(f"Waiting {config.wait_between_cycles_minutes} minutes before next cycle...")
                    await asyncio.sleep(config.wait_between_cycles_minutes * 60)
                    self.state_mgr.set_state(BotState.IDLE)

                # ERROR -> Manual intervention required
                elif state == BotState.ERROR:
                    logger.error(f"{Colors.RED}Bot is in ERROR state. Manual intervention required.{Colors.RESET}")
                    logger.error(f"Last error: {self.state_mgr.state['cumulative_stats']['last_error']}")
                    logger.error(f"Error time: {self.state_mgr.state['cumulative_stats']['last_error_at']}")
                    break

                # Unknown state -> Reset to IDLE
                else:
                    logger.warning(f"{Colors.YELLOW}Unknown state '{state}'. Resetting to IDLE.{Colors.RESET}")
                    self.state_mgr.set_state(BotState.IDLE)
                    await asyncio.sleep(1)  # Brief sleep to prevent busy loop

            except Exception as e:
                logger.error(f"{Colors.RED}Unexpected error in main loop: {e}{Colors.RESET}", exc_info=True)
                self.state_mgr.set_state(BotState.ERROR)
                self.state_mgr.state["cumulative_stats"]["last_error"] = str(e)
                self.state_mgr.state["cumulative_stats"]["last_error_at"] = datetime.utcnow().isoformat() + "Z"
                self.state_mgr.save()
                break

        # Shutdown
        logger.info(f"\n{Colors.CYAN}{'═' * 70}")
        logger.info(f"BOT SHUTTING DOWN")
        logger.info(f"{'═' * 70}{Colors.RESET}")
        display_status(self.state_mgr)
        logger.info("Goodbye!\n")


# ==================== Entry Point ====================

def main():
    parser = argparse.ArgumentParser(description="Automated delta-neutral rotation bot")
    parser.add_argument("--state-file", default="bot_state.json", help="Path to state file (default: bot_state.json)")
    parser.add_argument("--config", default="rotation_bot_config.json", help="Path to config file (default: rotation_bot_config.json)")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (no actual trading)")
    args = parser.parse_args()

    bot = RotationBot(state_file=args.state_file, config_file=args.config, dry_run=args.dry_run)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
