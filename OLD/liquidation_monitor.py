#!/usr/bin/env python3
"""
liquidation_monitor.py
----------------------
Continuously monitors delta-neutral positions for liquidation risk.
Checks EdgeX and Lighter positions every N seconds and auto-closes if unhealthy.

Usage:
    python liquidation_monitor.py --interval 60 --margin-threshold 20.0

Configuration:
    Uses hedge_config.json and .env file (same as hedge_cli.py)
"""

import asyncio
import json
import logging
import os
import sys
import argparse
import time
from typing import Optional, Tuple
from dotenv import load_dotenv

# EdgeX SDK
from edgex_sdk import Client as EdgeXClient

# Lighter SDK
import lighter
import websockets

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'

# Logging setup
os.makedirs('logs', exist_ok=True)

# File handler - DEBUG level (captures everything)
file_handler = logging.FileHandler('logs/liquidation_monitor.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))

# Console handler - INFO level only (clean output)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure root logger to DEBUG so file gets everything
logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

# Get our logger
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers in console (but keep in file)
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('lighter').setLevel(logging.WARNING)

# Configuration dataclass
class MonitorConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            data = json.load(f)
        self.symbol = data.get('symbol', 'PAXG')
        self.quote = data.get('quote', 'USD')
        self.long_exchange = data.get('long_exchange', 'lighter')
        self.short_exchange = data.get('short_exchange', 'edgex')
        self.leverage = data.get('leverage', 5)


# ==================== EdgeX Helpers ====================

async def edgex_find_contract_id(client: EdgeXClient, contract_name: str) -> Tuple[str, float, float]:
    """Find EdgeX contract ID and tick sizes."""
    metadata = await client.get_metadata()
    contracts = metadata.get("data", {}).get("contractList", [])
    for c in contracts:
        if c.get("contractName") == contract_name:
            cid = str(c.get("contractId"))
            tick = float(c.get("tickSize", "0.01"))
            step = float(c.get("stepSize", "0.000001"))
            return cid, tick, step
    raise ValueError(f"EdgeX contract '{contract_name}' not found")


async def edgex_get_position(client: EdgeXClient, contract_id: str) -> Optional[dict]:
    """Get EdgeX position details including unrealized PnL and margin info."""
    positions_response = await client.get_account_positions()
    positions = positions_response.get("data", {}).get("positionList", [])
    for p in positions:
        if p.get("contractId") == contract_id:
            # Log all available fields for debugging
            #logger.debug(f"EdgeX position data: {p}")
            return p
    return None


async def edgex_available_usd(client: EdgeXClient) -> Tuple[float, float]:
    """Return (total_usd, available_usd) from EdgeX."""
    try:
        resp = await client.get_account_asset()
        data = resp.get("data", {})
        total = 0.0
        avail = 0.0

        asset_list = data.get("collateralAssetModelList", [])
        if not asset_list:
            return 0.0, 0.0

        for asset in asset_list:
            asset_id = str(asset.get("assetId", ""))
            if asset_id == "1":  # USD
                total = float(asset.get("total", "0") or "0")
                avail = float(asset.get("available", "0") or "0")
                return total, avail

        return 0.0, 0.0
    except Exception as e:
        logger.error(f"EdgeX capital fetch error: {e}")
        return 0.0, 0.0


# ==================== Lighter Helpers ====================

async def lighter_get_market_details(order_api: lighter.OrderApi, symbol: str) -> Tuple[str, float, float]:
    """Get Lighter market ID and tick sizes."""
    try:
        resp = await order_api.order_books()
        for ob in resp.order_books:
            if ob.symbol.upper() == symbol.upper():
                market_id = str(ob.market_id)
                price_tick = 10 ** -ob.supported_price_decimals
                amount_tick = 10 ** -ob.supported_size_decimals
                return market_id, price_tick, amount_tick
        raise ValueError(f"Lighter market '{symbol}' not found")
    except lighter.ApiException as e:
        logger.error(f"Lighter API error getting market details: {e}")
        raise


async def lighter_get_position(account_api: lighter.AccountApi, account_index: int, market_id: str) -> Optional[dict]:
    """Get Lighter position details."""
    try:
        account_details_response = await account_api.account(by="index", value=str(account_index))

        if not (account_details_response and account_details_response.accounts):
            return None

        acc = account_details_response.accounts[0]
        if not acc.positions:
            return None

        for pos in acc.positions:
            if pos.market_id == int(market_id):
                size = float(pos.position)
                sign = int(pos.sign)

                if size == 0:
                    return None

                actual_size = size if sign == 1 else -size

                return {
                    'size': actual_size,
                    'unrealized_pnl': float(pos.unrealized_pnl) if hasattr(pos, 'unrealized_pnl') else 0.0,
                    'entry_price': float(pos.average_entry_price) if hasattr(pos, 'average_entry_price') else 0.0
                }
        return None
    except Exception as e:
        logger.error(f"Lighter API error getting position: {e}")
        return None


async def lighter_available_capital_ws(ws_url: str, account_index: int, timeout: float = 10.0) -> Tuple[Optional[float], Optional[float]]:
    """Get Lighter available balance and portfolio value via WebSocket."""
    sub = {"type": "subscribe", "channel": f"user_stats/{account_index}"}
    start = time.time()
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps(sub))
            while time.time() - start < timeout:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout - (time.time() - start))
                data = json.loads(msg)
                t = data.get("type")
                if t in ("update/user_stats", "subscribed/user_stats"):
                    stats = data.get("stats", {})
                    avail = float(stats.get("available_balance", 0) or 0)
                    portv = float(stats.get("portfolio_value", 0) or 0)
                    return avail, portv
        return None, None
    except Exception as e:
        logger.error(f"Lighter WS capital fetch error: {e}")
        return None, None


async def lighter_best_bid_ask(order_api: lighter.OrderApi, symbol: str, market_id: str, timeout: float = 10.0) -> Tuple[Optional[float], Optional[float]]:
    """Get best bid/ask from Lighter using WebSocket (REST API returns empty order books)."""

    class OrderBookFetcher:
        def __init__(self, market_id: int):
            self.market_id = market_id
            self.best_bid = None
            self.best_ask = None
            self.received_event = asyncio.Event()

        def on_order_book_update(self, mid, order_book):
            if int(mid) == int(self.market_id):
                try:
                    bids = order_book.get('bids', [])
                    asks = order_book.get('asks', [])
                    if bids and asks:
                        self.best_bid = float(bids[0]['price'])
                        self.best_ask = float(asks[0]['price'])
                    self.received_event.set()
                except Exception as e:
                    logger.error(f"Error parsing Lighter order book: {e}")
                    self.received_event.set()

        def on_account_update(self, account_id, update):
            pass

    fetcher = OrderBookFetcher(int(market_id))

    try:
        ws_client = lighter.WsClient(
            order_book_ids=[int(market_id)],
            account_ids=[],
            on_order_book_update=fetcher.on_order_book_update,
            on_account_update=fetcher.on_account_update,
        )

        ws_task = asyncio.create_task(ws_client.run_async())

        try:
            await asyncio.wait_for(fetcher.received_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Lighter: Timeout waiting for order book update")

        ws_task.cancel()
        try:
            await ws_task
        except asyncio.CancelledError:
            pass

        return fetcher.best_bid, fetcher.best_ask

    except Exception as e:
        logger.error(f"Lighter order book error: {e}")
        return None, None


# ==================== EdgeX Order Book ====================

async def edgex_best_bid_ask(client: EdgeXClient, contract_id: str) -> Tuple[Optional[float], Optional[float]]:
    """Get best bid/ask from EdgeX."""
    try:
        q = await client.quote.get_24_hour_quote(contract_id)
        data_list = q.get("data", [])
        if isinstance(data_list, list) and data_list:
            d = data_list[0]
            bid = float(d.get("bestBid")) if d.get("bestBid") else None
            ask = float(d.get("bestAsk")) if d.get("bestAsk") else None

            if bid and ask:
                return bid, ask

            # Fallback to lastPrice with synthetic spread
            last_price_str = d.get("lastPrice")
            if last_price_str:
                last_price = float(last_price_str)
                synthetic_bid = last_price * 0.9995
                synthetic_ask = last_price * 1.0005
                return synthetic_bid, synthetic_ask

        return None, None
    except Exception as e:
        logger.error(f"EdgeX order book error: {e}")
        return None, None


# ==================== Health Check Logic ====================

def calculate_pnl_percentage(position_value: float, unrealized_pnl: float) -> float:
    """
    Calculate unrealized PnL as a percentage of position value.

    Negative values = losing money
    Returns -100% if losing entire position value (near liquidation)
    """
    if position_value == 0:
        return 0.0

    pnl_pct = (unrealized_pnl / position_value) * 100
    return pnl_pct


async def check_positions_health(config: MonitorConfig, env: dict, pnl_threshold: float) -> Tuple[bool, dict]:
    """
    Check both exchange positions for liquidation risk based on PnL%.
    Returns (is_healthy, status_dict)

    pnl_threshold: If PnL% drops below this (e.g., -10 means -10%), trigger close
    """
    contract_name = config.symbol + config.quote

    # Initialize clients
    ex_client = EdgeXClient(
        base_url=env["EDGEX_BASE_URL"],
        account_id=int(env["EDGEX_ACCOUNT_ID"]),
        stark_private_key=env["EDGEX_STARK_PRIVATE_KEY"]
    )

    api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
    order_api = lighter.OrderApi(api_client)
    account_api = lighter.AccountApi(api_client)

    status = {
        'edgex': {},
        'lighter': {},
        'is_healthy': True,
        'warnings': [],
        'leverage': config.leverage
    }

    try:
        # Get contract/market info
        edgex_contract_id, edgex_tick, edgex_step = await edgex_find_contract_id(ex_client, contract_name)
        lighter_market_id, lighter_tick, lighter_step = await lighter_get_market_details(order_api, config.symbol)

        # Get market prices
        edgex_bid, edgex_ask = await edgex_best_bid_ask(ex_client, edgex_contract_id)
        lighter_bid, lighter_ask = await lighter_best_bid_ask(order_api, config.symbol, lighter_market_id)

        edgex_mid = (edgex_bid + edgex_ask) / 2.0 if edgex_bid and edgex_ask else None
        lighter_mid = (lighter_bid + lighter_ask) / 2.0 if lighter_bid and lighter_ask else None

        # EdgeX position and capital
        edgex_position = await edgex_get_position(ex_client, edgex_contract_id)
        edgex_total, edgex_avail = await edgex_available_usd(ex_client)

        if edgex_position and edgex_mid:
            edgex_size = float(edgex_position.get("openSize", "0"))

            # Calculate PnL manually from openValue and current market value
            # openValue is the USD value when position was opened (negative for shorts)
            # Current value = current_price * size
            # PnL = current_value - openValue (for longs)
            # For shorts: openValue is negative, current_value is negative
            open_value = float(edgex_position.get("openValue", "0"))

            if abs(open_value) > 0 and edgex_size != 0:
                # Calculate entry price from open value
                entry_price = abs(open_value) / abs(edgex_size)

                # Current market value
                current_value = edgex_mid * edgex_size

                # PnL = current_value - open_value
                # For short: openValue is negative, current_value is negative
                # If price goes down (good for short), current_value is less negative than openValue
                edgex_upnl = current_value - open_value

                #logger.debug(f"EdgeX PnL: size={edgex_size}, entry={entry_price}, current={edgex_mid}, openValue={open_value}, currentValue={current_value}, pnl={edgex_upnl}")
            else:
                edgex_upnl = 0.0

            edgex_value = abs(edgex_size) * edgex_mid
            edgex_pnl_pct = calculate_pnl_percentage(edgex_value, edgex_upnl)

            status['edgex'] = {
                'size': edgex_size,
                'value': edgex_value,
                'unrealized_pnl': edgex_upnl,
                'pnl_pct': edgex_pnl_pct,
                'price': edgex_mid
            }

            if edgex_pnl_pct < pnl_threshold:
                status['is_healthy'] = False
                status['warnings'].append(f"EdgeX PnL critically low: {edgex_pnl_pct:.2f}%")

        # Lighter position and capital
        lighter_position = await lighter_get_position(account_api, env["ACCOUNT_INDEX"], lighter_market_id)
        lighter_avail, lighter_portv = await lighter_available_capital_ws(env["LIGHTER_WS_URL"], env["ACCOUNT_INDEX"])

        if lighter_position and lighter_mid:
            lighter_size = lighter_position['size']
            lighter_upnl = lighter_position['unrealized_pnl']
            lighter_value = abs(lighter_size) * lighter_mid

            lighter_pnl_pct = calculate_pnl_percentage(lighter_value, lighter_upnl)

            status['lighter'] = {
                'size': lighter_size,
                'value': lighter_value,
                'unrealized_pnl': lighter_upnl,
                'pnl_pct': lighter_pnl_pct,
                'price': lighter_mid
            }

            if lighter_pnl_pct < pnl_threshold:
                status['is_healthy'] = False
                status['warnings'].append(f"Lighter PnL critically low: {lighter_pnl_pct:.2f}%")

        # Check if positions are imbalanced (hedging quality)
        if status['edgex'] and status['lighter']:
            edgex_size = status['edgex']['size']
            lighter_size = status['lighter']['size']

            # Check if properly hedged (opposite directions)
            if (edgex_size > 0 and lighter_size > 0) or (edgex_size < 0 and lighter_size < 0):
                status['warnings'].append("CRITICAL: Positions are in same direction (not hedged)!")
                status['is_healthy'] = False

            # Check size imbalance
            if abs(edgex_size) > 0 and abs(lighter_size) > 0:
                size_ratio = min(abs(edgex_size), abs(lighter_size)) / max(abs(edgex_size), abs(lighter_size))
                if size_ratio < 0.90:
                    status['warnings'].append(f"Position size imbalance detected: {size_ratio*100:.1f}% matched")

    finally:
        await ex_client.close()
        await api_client.close()

    return status['is_healthy'], status


async def close_hedge_positions(config: MonitorConfig, env: dict):
    """Emergency close both positions using hedge_cli logic."""
    logger.warning(f"{Colors.RED}{Colors.BOLD}🚨 EMERGENCY CLOSE TRIGGERED - Closing all positions...{Colors.RESET}")

    # Import hedge_cli close function
    import subprocess
    result = subprocess.run(
        [sys.executable, "hedge_cli.py", "close", "--cross-ticks", "100"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        logger.info(f"{Colors.GREEN}✓ Positions closed successfully{Colors.RESET}")
        return True
    else:
        logger.error(f"{Colors.RED}❌ Failed to close positions: {result.stderr}{Colors.RESET}")
        return False


# ==================== Main Monitor Loop ====================

async def monitor_loop(config_path: str, check_interval: int, pnl_threshold: float, auto_close: bool):
    """Main monitoring loop."""
    load_dotenv()

    # Load config
    config = MonitorConfig(config_path)

    # Load environment
    env = {
        "EDGEX_BASE_URL": os.getenv("EDGEX_BASE_URL", "https://pro.edgex.exchange"),
        "EDGEX_WS_URL": os.getenv("EDGEX_WS_URL", "wss://quote.edgex.exchange"),
        "EDGEX_ACCOUNT_ID": os.getenv("EDGEX_ACCOUNT_ID"),
        "EDGEX_STARK_PRIVATE_KEY": os.getenv("EDGEX_STARK_PRIVATE_KEY"),
        "LIGHTER_BASE_URL": os.getenv("LIGHTER_BASE_URL", os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")),
        "LIGHTER_WS_URL": os.getenv("LIGHTER_WS_URL", os.getenv("WEBSOCKET_URL", "wss://mainnet.zklighter.elliot.ai/stream")),
        "API_KEY_PRIVATE_KEY": os.getenv("API_KEY_PRIVATE_KEY", os.getenv("LIGHTER_PRIVATE_KEY")),
        "ACCOUNT_INDEX": int(os.getenv("ACCOUNT_INDEX", os.getenv("LIGHTER_ACCOUNT_INDEX", "0"))),
        "API_KEY_INDEX": int(os.getenv("API_KEY_INDEX", os.getenv("LIGHTER_API_KEY_INDEX", "0"))),
    }

    logger.info(f"{Colors.CYAN}{'═' * 63}{Colors.RESET}")
    logger.info(f"{Colors.CYAN}{Colors.BOLD}🔍 Liquidation Monitor Started{Colors.RESET}")
    logger.info(f"{Colors.CYAN}{'═' * 63}{Colors.RESET}")
    logger.info(f"{Colors.WHITE}  Market: {Colors.BOLD}{config.symbol}/{config.quote}{Colors.RESET}")
    logger.info(f"{Colors.WHITE}  Check Interval: {Colors.BOLD}{check_interval}s{Colors.RESET}")
    logger.info(f"{Colors.WHITE}  PnL Threshold: {Colors.BOLD}{pnl_threshold}%{Colors.RESET} (closes if PnL% drops below this)")
    logger.info(f"{Colors.WHITE}  Auto-Close: {Colors.BOLD}{Colors.GREEN if auto_close else Colors.RED}{'ENABLED' if auto_close else 'DISABLED'}{Colors.RESET}")
    logger.info(f"{Colors.CYAN}{'═' * 63}{Colors.RESET}\n")

    consecutive_failures = 0
    max_failures = 5

    while True:
        try:
            is_healthy, status = await check_positions_health(config, env, pnl_threshold)
            consecutive_failures = 0  # Reset on success

            # Display status
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            leverage = status.get('leverage', config.leverage)

            logger.info(f"\n{Colors.BLUE}{'─' * 60}{Colors.RESET}")
            logger.info(f"{Colors.BLUE}Check at {Colors.BOLD}{timestamp}{Colors.RESET} | Leverage: {Colors.BOLD}{leverage}x{Colors.RESET}")
            logger.info(f"{Colors.BLUE}{'─' * 60}{Colors.RESET}")

            if status['edgex']:
                ex = status['edgex']
                pnl_color = Colors.GREEN if ex['unrealized_pnl'] >= 0 else Colors.RED
                pnl_pct_color = Colors.GREEN if ex['pnl_pct'] >= 0 else Colors.RED
                logger.info(f"{Colors.MAGENTA}EdgeX ({leverage}x):{Colors.RESET}   Size={ex['size']:+.6f} | PnL={pnl_color}${ex['unrealized_pnl']:+.2f}{Colors.RESET} | PnL%={pnl_pct_color}{ex['pnl_pct']:+.2f}%{Colors.RESET}")
            else:
                logger.info(f"{Colors.MAGENTA}EdgeX ({leverage}x):{Colors.RESET}   {Colors.GRAY}No position{Colors.RESET}")

            if status['lighter']:
                lt = status['lighter']
                pnl_color = Colors.GREEN if lt['unrealized_pnl'] >= 0 else Colors.RED
                pnl_pct_color = Colors.GREEN if lt['pnl_pct'] >= 0 else Colors.RED
                logger.info(f"{Colors.CYAN}Lighter ({leverage}x):{Colors.RESET} Size={lt['size']:+.6f} | PnL={pnl_color}${lt['unrealized_pnl']:+.2f}{Colors.RESET} | PnL%={pnl_pct_color}{lt['pnl_pct']:+.2f}%{Colors.RESET}")
            else:
                logger.info(f"{Colors.CYAN}Lighter ({leverage}x):{Colors.RESET} {Colors.GRAY}No position{Colors.RESET}")

            if not is_healthy:
                logger.warning(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  UNHEALTHY POSITION DETECTED!{Colors.RESET}")
                for warning in status['warnings']:
                    logger.warning(f"{Colors.YELLOW}   {warning}{Colors.RESET}")

                if auto_close:
                    logger.warning(f"\n{Colors.RED}{Colors.BOLD}🚨 AUTO-CLOSE TRIGGERED - Closing positions...{Colors.RESET}")
                    success = await close_hedge_positions(config, env)
                    if success:
                        logger.info(f"{Colors.GREEN}✓ Emergency close completed successfully{Colors.RESET}")
                        break  # Exit monitor after closing
                    else:
                        logger.error(f"{Colors.RED}{Colors.BOLD}❌ Emergency close failed - manual intervention required!{Colors.RESET}")
                else:
                    logger.warning(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Auto-close is DISABLED - Please close positions manually!{Colors.RESET}")
            else:
                logger.info(f"{Colors.GREEN}✓ Positions healthy{Colors.RESET}")

            # Show countdown to next check
            logger.info(f"{Colors.GRAY}⏳ Next check in {check_interval} seconds...{Colors.RESET}")

        except Exception as e:
            consecutive_failures += 1
            logger.error(f"{Colors.RED}❌ Monitor error: {e}{Colors.RESET}")

            if consecutive_failures >= max_failures:
                logger.error(f"{Colors.RED}{Colors.BOLD}❌ Too many consecutive failures ({max_failures}). Exiting.{Colors.RESET}")
                break

            logger.info(f"{Colors.YELLOW}⏳ Retrying in {check_interval} seconds...{Colors.RESET}")

        # Wait for next check
        await asyncio.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor delta-neutral positions for liquidation risk based on PnL%",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default="hedge_config.json", help="Path to config file")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--pnl-threshold", type=float, default=-15.0, help="PnL%% threshold (closes if PnL%% drops below this, e.g., -50 = -50%%)")
    parser.add_argument("--auto-close", action="store_true", default=True, help="Automatically close positions if unhealthy")
    parser.add_argument("--no-auto-close", dest="auto_close", action="store_false", help="Disable auto-close (warning only)")

    args = parser.parse_args()

    try:
        asyncio.run(monitor_loop(args.config, args.interval, args.pnl_threshold, args.auto_close))
    except KeyboardInterrupt:
        logger.info(f"\n\n{Colors.YELLOW}🛑 Monitor stopped by user{Colors.RESET}")
    except Exception as e:
        logger.error(f"{Colors.RED}{Colors.BOLD}Fatal error: {e}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
