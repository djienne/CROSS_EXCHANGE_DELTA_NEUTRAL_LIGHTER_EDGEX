#!/usr/bin/env python3
"""
test_funding_comparison.py
---------------------------
Fetch and compare funding rates (APR) between EdgeX and Lighter exchanges.

Usage:
    python test_funding_comparison.py
    python test_funding_comparison.py --symbols BTC ETH SOL
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

import edgex_client
import lighter_client
from edgex_sdk import Client as EdgeXClient
import lighter

# Load environment variables
load_dotenv()


def load_symbols_from_config(config_file: str = "bot_config.json") -> List[str]:
    """Load symbols from rotation bot configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config.get("symbols_to_monitor", [])
    except FileNotFoundError:
        print(f"Warning: {config_file} not found, using default symbols")
        return ["BTC", "ETH", "SOL"]
    except Exception as e:
        print(f"Warning: Error loading {config_file}: {e}")
        return ["BTC", "ETH", "SOL"]


class FundingData:
    """Container for funding rate data from an exchange."""
    def __init__(self, symbol: str, exchange: str):
        self.symbol = symbol
        self.exchange = exchange
        self.funding_rate: Optional[float] = None  # Rate per period (e.g., 8h or 1h)
        self.apr: Optional[float] = None
        self.error: Optional[str] = None
        self.contract_id: Optional[str] = None
        self.market_id: Optional[int] = None


def calculate_apr(funding_rate: float, periods_per_day: int) -> float:
    """
    Convert funding rate to annualized percentage rate (APR).

    Args:
        funding_rate: Funding rate as decimal per period (e.g., 0.0001 = 0.01%)
        periods_per_day: Number of funding periods per day

    Returns:
        APR as percentage (e.g., 10.0 = 10% per year)
    """
    # funding_rate is in decimal format (e.g., 0.0001 = 0.01%)
    # Convert to percentage APR: rate * periods_per_day * 365 days * 100 (to convert to %)
    daily_rate = funding_rate * periods_per_day
    apr = daily_rate * 365 * 100  # Multiply by 100 to convert decimal to percentage
    return apr


async def fetch_edgex_funding(
    client: EdgeXClient,
    symbol: str,
    quote: str = "USD"
) -> FundingData:
    """Fetch EdgeX funding rate for a symbol."""
    data = FundingData(symbol, "EdgeX")

    try:
        # EdgeX contract name is symbol+quote (e.g., "BTCUSD")
        contract_name = f"{symbol.upper()}{quote.upper()}"

        # Get contract details
        contract_id, _, _ = await edgex_client.get_edgex_contract_details(client, contract_name)
        data.contract_id = contract_id

        # Fetch funding rate
        funding_rate = await edgex_client.get_edgex_funding_rate(client, contract_id)

        if funding_rate is not None:
            data.funding_rate = funding_rate
            # EdgeX funding periods: rate is per 8h but paid every 4h (6 per day)
            data.apr = calculate_apr(funding_rate, 6)
        else:
            data.error = "No funding rate data"

    except Exception as e:
        data.error = str(e)

    return data


async def fetch_lighter_funding(
    api_client: lighter.ApiClient,
    order_api: lighter.OrderApi,
    symbol: str
) -> FundingData:
    """Fetch Lighter funding rate for a symbol."""
    data = FundingData(symbol, "Lighter")

    try:
        # Get market details
        market_id, _, _ = await lighter_client.get_lighter_market_details(order_api, symbol)
        data.market_id = market_id

        # Fetch funding rate
        funding_rate = await lighter_client.get_lighter_funding_rate(api_client, market_id)

        if funding_rate is not None:
            data.funding_rate = funding_rate
            # Lighter rate is 8-hour rate even though payments are hourly (3 periods per day)
            data.apr = calculate_apr(funding_rate, 3)
        else:
            data.error = "No funding rate data"

    except Exception as e:
        data.error = str(e)

    return data


def format_rate(rate: Optional[float]) -> str:
    """Format funding rate for display (convert decimal to percentage)."""
    if rate is None:
        return "N/A"
    # rate is in decimal format (e.g., 0.0001 = 0.01%)
    # Convert to percentage for display
    return f"{rate * 100:+.4f}%"


def format_apr(apr: Optional[float]) -> str:
    """Format APR for display."""
    if apr is None:
        return "N/A"
    return f"{apr:+.2f}%"


def calculate_spread(edgex_data: FundingData, lighter_data: FundingData) -> Tuple[Optional[float], str]:
    """
    Calculate the net APR spread (profit opportunity).

    Returns:
        (net_apr, direction) where direction is "Long EdgeX / Short Lighter" or vice versa
    """
    if edgex_data.apr is None or lighter_data.apr is None:
        return None, "N/A"

    # Evaluate both orientations using the sign of the APR (positive APR = cost to longs, credit to shorts)
    long_edgex_short_lighter = lighter_data.apr - edgex_data.apr
    long_lighter_short_edgex = edgex_data.apr - lighter_data.apr

    if long_edgex_short_lighter >= long_lighter_short_edgex:
        net_apr = long_edgex_short_lighter
        direction = "Long EdgeX / Short Lighter"
    else:
        net_apr = long_lighter_short_edgex
        direction = "Long Lighter / Short EdgeX"

    return net_apr, direction


def print_comparison_table(results: List[Tuple[str, FundingData, FundingData]]):
    """Print a formatted comparison table."""

    print("\n" + "=" * 120)
    print(f"{'FUNDING RATE COMPARISON':^120}")
    print("=" * 120)
    print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 120)

    # Header
    header = (
        f"{'Symbol':<8} | "
        f"{'EdgeX Rate':<12} | "
        f"{'EdgeX APR':<12} | "
        f"{'Lighter Rate':<12} | "
        f"{'Lighter APR':<12} | "
        f"{'Net APR':<12} | "
        f"{'Strategy':<30}"
    )
    print(header)
    print("-" * 120)

    # Sort by net APR (best opportunities first)
    sorted_results = []
    for symbol, edgex, lighter in results:
        net_apr, direction = calculate_spread(edgex, lighter)
        sorted_results.append((symbol, edgex, lighter, net_apr, direction))

    sorted_results.sort(key=lambda x: abs(x[3]) if x[3] is not None else -1, reverse=True)

    # Print rows
    for symbol, edgex, lighter, net_apr, direction in sorted_results:
        if edgex.error and lighter.error:
            print(f"{symbol:<8} | ERROR: {edgex.error[:40]:<92}")
            continue

        edgex_rate = format_rate(edgex.funding_rate)
        edgex_apr = format_apr(edgex.apr)
        lighter_rate = format_rate(lighter.funding_rate)
        lighter_apr = format_apr(lighter.apr)
        net_apr_str = format_apr(net_apr) if net_apr is not None else "N/A"

        # Color coding for net APR
        if net_apr is not None:
            if net_apr > 10:
                net_apr_str = f"[++] {net_apr_str}"
            elif net_apr > 5:
                net_apr_str = f"[+]  {net_apr_str}"
            elif net_apr > 0:
                net_apr_str = f"[=]  {net_apr_str}"
            else:
                net_apr_str = f"[-]  {net_apr_str}"

        row = (
            f"{symbol:<8} | "
            f"{edgex_rate:<12} | "
            f"{edgex_apr:<12} | "
            f"{lighter_rate:<12} | "
            f"{lighter_apr:<12} | "
            f"{net_apr_str:<12} | "
            f"{direction:<30}"
        )
        print(row)

    print("=" * 120)
    print("\nLegend:")
    print("  Rate: Funding rate per 8-hour period (both exchanges)")
    print("  APR:  Annualized percentage rate (365 days)")
    print("  Net APR: Expected profit from delta-neutral hedge (higher is better)")
    print("  Note: EdgeX pays funding every 4h (6x/day), Lighter pays hourly (24x/day)")
    print("  [++] Excellent (>10%)  [+] Good (5-10%)  [=] Fair (0-5%)  [-] Negative (<0%)")
    print("=" * 120)


async def main(symbols: List[str], quote: str = "USD"):
    """Main function to fetch and compare funding rates."""

    print(f"\n[*] Fetching funding rates for {len(symbols)} symbols...")

    # Initialize EdgeX client
    edgex_base_url = os.getenv("EDGEX_BASE_URL", "https://pro.edgex.exchange")
    edgex_account_id = os.getenv("EDGEX_ACCOUNT_ID")
    edgex_private_key = os.getenv("EDGEX_STARK_PRIVATE_KEY")

    if not edgex_account_id or not edgex_private_key:
        print("[X] ERROR: EdgeX credentials not found in .env file")
        sys.exit(1)

    edgex = EdgeXClient(
        base_url=edgex_base_url,
        account_id=edgex_account_id,
        stark_private_key=edgex_private_key
    )

    # Initialize Lighter client
    lighter_base_url = os.getenv("LIGHTER_BASE_URL") or os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")

    api_client = lighter.ApiClient(configuration=lighter.Configuration(host=lighter_base_url))
    order_api = lighter.OrderApi(api_client)

    results = []

    for symbol in symbols:
        print(f"  Fetching {symbol}...", end=" ")

        # Fetch from both exchanges concurrently
        edgex_task = fetch_edgex_funding(edgex, symbol, quote)
        lighter_task = fetch_lighter_funding(api_client, order_api, symbol)

        edgex_data, lighter_data = await asyncio.gather(edgex_task, lighter_task)

        results.append((symbol, edgex_data, lighter_data))

        # Show quick status
        if edgex_data.apr is not None and lighter_data.apr is not None:
            net_apr, _ = calculate_spread(edgex_data, lighter_data)
            print(f"[OK] (Net APR: {net_apr:+.2f}%)")
        else:
            print("[!] (Incomplete data)")

    # Close clients
    await edgex.close()
    await api_client.close()

    # Print comparison table
    print_comparison_table(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare funding rates (APR) between EdgeX and Lighter exchanges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_funding_comparison.py
  python test_funding_comparison.py --symbols BTC ETH SOL
  python test_funding_comparison.py --symbols BTC ETH SOL HYPE PAXG --quote USD

Note:
  - Requires .env file with EdgeX and Lighter credentials
  - Both exchanges quote funding rates per 8-hour period
  - EdgeX pays funding every 4 hours (6x per day)
  - Lighter pays funding hourly (24x per day)
  - Net APR shows the expected profit from delta-neutral hedging
        """
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to compare (default: load from bot_config.json)"
    )

    parser.add_argument(
        "--quote",
        default="USD",
        help="Quote currency (default: USD)"
    )

    args = parser.parse_args()

    # Load symbols from config if not specified
    symbols = args.symbols if args.symbols is not None else load_symbols_from_config()

    if not symbols:
        print("[X] No symbols to compare")
        sys.exit(1)

    try:
        asyncio.run(main(symbols, args.quote))
    except KeyboardInterrupt:
        print("\n\n[X] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
