#!/usr/bin/env python3
"""
Check mid price spreads for all symbols in bot_config.json.
Displays a summary table comparing EdgeX vs Lighter prices.
"""

import asyncio
import json
import os
import sys
from typing import Optional, Dict, List
from dotenv import load_dotenv
from edgex_sdk import Client as EdgeXClient
import lighter

# Import helper functions from client modules
from edgex_client import get_edgex_contract_details, get_edgex_best_bid_ask
from lighter_client import get_lighter_market_details, get_lighter_best_bid_ask


def load_bot_config(config_file="bot_config.json"):
    """Load bot configuration from JSON file."""
    if not os.path.exists(config_file):
        print(f"❌ Error: {config_file} not found")
        sys.exit(1)

    with open(config_file, 'r') as f:
        return json.load(f)


async def get_edgex_mid_price(symbol: str, quote: str, env: dict) -> Optional[Dict]:
    """Fetch mid price from EdgeX."""
    edgex_client = None
    try:
        edgex_client = EdgeXClient(
            base_url=env.get("EDGEX_BASE_URL", "https://pro.edgex.exchange"),
            account_id=int(env["EDGEX_ACCOUNT_ID"]),
            stark_private_key=env["EDGEX_STARK_PRIVATE_KEY"]
        )

        # Get contract details
        contract_name = f"{symbol}{quote}"
        contract_id, tick_size, step_size = await get_edgex_contract_details(edgex_client, contract_name)

        # Get best bid/ask
        best_bid, best_ask = await get_edgex_best_bid_ask(edgex_client, contract_id)

        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
            spread_bps = ((best_ask - best_bid) / mid_price) * 10000
        elif best_bid or best_ask:
            mid_price = best_bid if best_bid else best_ask
            spread_bps = None
        else:
            return None

        return {
            'mid_price': mid_price,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread_bps': spread_bps
        }

    except Exception as e:
        return None
    finally:
        if edgex_client:
            await edgex_client.close()


async def get_lighter_mid_price(symbol: str, env: dict) -> Optional[Dict]:
    """Fetch mid price from Lighter."""
    api_client = None
    try:
        # Initialize Lighter client
        base_url = env.get("LIGHTER_BASE_URL") or env.get("BASE_URL", "https://mainnet.zklighter.elliot.ai")

        api_client = lighter.ApiClient(configuration=lighter.Configuration(host=base_url))
        order_api = lighter.OrderApi(api_client)

        # Get market details
        market_id, price_tick, amount_tick = await get_lighter_market_details(order_api, symbol)

        # Get best bid/ask via WebSocket
        best_bid, best_ask = await get_lighter_best_bid_ask(order_api, symbol, market_id, timeout=10.0)

        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
            spread_bps = ((best_ask - best_bid) / mid_price) * 10000
        elif best_bid or best_ask:
            mid_price = best_bid if best_bid else best_ask
            spread_bps = None
        else:
            return None

        return {
            'mid_price': mid_price,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread_bps': spread_bps
        }

    except Exception as e:
        return None
    finally:
        if api_client:
            await api_client.close()


async def fetch_symbol_spread(symbol: str, quote: str, env: dict, edgex_data: Optional[Dict]) -> Dict:
    """Fetch spread data for a single symbol (EdgeX data provided, fetch Lighter)."""

    # Fetch from Lighter (EdgeX data already provided)
    lighter_data = await get_lighter_mid_price(symbol, env)

    result = {
        'symbol': symbol,
        'edgex_mid': None,
        'edgex_spread_bps': None,
        'lighter_mid': None,
        'lighter_spread_bps': None,
        'cross_spread_bps': None,
        'cross_spread_pct': None,
        'higher_exchange': None,
        'available': True
    }

    # Handle EdgeX data
    if isinstance(edgex_data, dict) and edgex_data:
        result['edgex_mid'] = edgex_data['mid_price']
        result['edgex_spread_bps'] = edgex_data.get('spread_bps')
    else:
        result['available'] = False

    # Handle Lighter data
    if isinstance(lighter_data, dict) and lighter_data:
        result['lighter_mid'] = lighter_data['mid_price']
        result['lighter_spread_bps'] = lighter_data.get('spread_bps')
    else:
        result['available'] = False

    # Calculate cross-exchange spread
    if result['edgex_mid'] and result['lighter_mid']:
        edgex_mid = result['edgex_mid']
        lighter_mid = result['lighter_mid']

        price_diff = abs(edgex_mid - lighter_mid)
        avg_mid = (edgex_mid + lighter_mid) / 2

        result['cross_spread_bps'] = (price_diff / avg_mid) * 10000
        result['cross_spread_pct'] = (price_diff / avg_mid) * 100
        result['higher_exchange'] = "EdgeX" if edgex_mid > lighter_mid else "Lighter"

    return result


def format_price(price: Optional[float]) -> str:
    """Format price with appropriate precision."""
    if price is None:
        return "N/A"
    if price < 0.01:
        return f"${price:.6f}"
    elif price < 1:
        return f"${price:.4f}"
    elif price < 100:
        return f"${price:.2f}"
    else:
        return f"${price:,.2f}"


def format_bps(bps: Optional[float]) -> str:
    """Format basis points."""
    if bps is None:
        return "N/A"
    return f"{bps:.1f}"


def print_summary_table(results: List[Dict]):
    """Print a nicely formatted summary table."""

    # Calculate column widths
    symbol_width = max(len(r['symbol']) for r in results) + 2
    symbol_width = max(symbol_width, 8)

    # Header
    print(f"\n{'='*90}")
    print(f"CROSS-EXCHANGE SPREAD ANALYSIS")
    print(f"{'='*90}")

    # Table header
    header = (
        f"{'SYMBOL':<{symbol_width}} "
        f"{'EdgeX Mid':>14} "
        f"{'Lighter Mid':>14} "
        f"{'Spread (bps)':>12} "
        f"{'Spread (%)':>12} "
        f"{'Higher':>8}"
    )
    print(header)
    print(f"{'-'*90}")

    # Sort results by cross spread (descending)
    sorted_results = sorted(
        [r for r in results if r['available'] and r['cross_spread_bps'] is not None],
        key=lambda x: x['cross_spread_bps'],
        reverse=True
    )

    # Add unavailable symbols at the end
    unavailable = [r for r in results if not r['available'] or r['cross_spread_bps'] is None]

    # Print available symbols
    for result in sorted_results:
        row = (
            f"{result['symbol']:<{symbol_width}} "
            f"{format_price(result['edgex_mid']):>14} "
            f"{format_price(result['lighter_mid']):>14} "
            f"{format_bps(result['cross_spread_bps']):>12} "
            f"{result['cross_spread_pct']:>11.3f}% "
            f"{result['higher_exchange']:>8}"
        )
        print(row)

    # Print unavailable symbols
    if unavailable:
        print(f"{'-'*90}")
        for result in unavailable:
            row = (
                f"{result['symbol']:<{symbol_width}} "
                f"{format_price(result['edgex_mid']):>14} "
                f"{format_price(result['lighter_mid']):>14} "
                f"{'N/A':>12} "
                f"{'N/A':>12} "
                f"{'N/A':>8}"
            )
            print(row)

    print(f"{'='*90}\n")

    # Summary statistics
    if sorted_results:
        print("SUMMARY STATISTICS")
        print(f"{'='*90}")

        avg_cross_spread = sum(r['cross_spread_bps'] for r in sorted_results) / len(sorted_results)
        max_spread = max(sorted_results, key=lambda x: x['cross_spread_bps'])
        min_spread = min(sorted_results, key=lambda x: x['cross_spread_bps'])

        print(f"Total Symbols Analyzed:    {len(sorted_results)}")
        print(f"Average Cross Spread:      {avg_cross_spread:.1f} bps ({avg_cross_spread/100:.3f}%)")
        print(f"Maximum Cross Spread:      {max_spread['cross_spread_bps']:.1f} bps ({max_spread['cross_spread_pct']:.3f}%) - {max_spread['symbol']}")
        print(f"Minimum Cross Spread:      {min_spread['cross_spread_bps']:.1f} bps ({min_spread['cross_spread_pct']:.3f}%) - {min_spread['symbol']}")

        # Count which exchange is higher
        edgex_higher = sum(1 for r in sorted_results if r['higher_exchange'] == 'EdgeX')
        lighter_higher = sum(1 for r in sorted_results if r['higher_exchange'] == 'Lighter')

        print(f"\nEdgeX Higher:              {edgex_higher} symbols")
        print(f"Lighter Higher:            {lighter_higher} symbols")

        if unavailable:
            print(f"\nUnavailable Symbols:       {len(unavailable)}")

        print(f"{'='*90}\n")


async def main():
    # Load environment
    load_dotenv()
    env = os.environ

    # Load bot configuration
    config = load_bot_config()
    symbols = config.get('symbols_to_monitor', [])
    quote = config.get('quote', 'USD')

    if not symbols:
        print("❌ No symbols found in bot_config.json")
        sys.exit(1)

    print(f"\nFetching spread data for {len(symbols)} symbols...")
    print("Step 1/2: Fetching all EdgeX prices concurrently...")

    # Fetch all EdgeX prices concurrently (EdgeX has no rate limits)
    edgex_tasks = [get_edgex_mid_price(symbol, quote, env) for symbol in symbols]
    edgex_results = await asyncio.gather(*edgex_tasks)

    print(f"Step 2/2: Fetching Lighter prices sequentially (avoiding rate limits)...\n")

    # Fetch Lighter prices sequentially with delays to avoid rate limits
    results = []
    for i, (symbol, edgex_data) in enumerate(zip(symbols, edgex_results)):
        if i > 0:
            # Add delay between Lighter API calls to avoid rate limits
            await asyncio.sleep(1.0)

        # Show progress
        print(f"  [{i+1}/{len(symbols)}] Fetching {symbol} from Lighter...", end='\r')

        result = await fetch_symbol_spread(symbol, quote, env, edgex_data)
        results.append(result)

    print(" " * 60, end='\r')  # Clear progress line

    # Print summary table
    print_summary_table(results)


if __name__ == "__main__":
    # Suppress debug logging for cleaner output
    import logging
    logging.getLogger().setLevel(logging.WARNING)

    asyncio.run(main())
