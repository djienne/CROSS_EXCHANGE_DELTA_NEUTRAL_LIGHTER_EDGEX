#!/usr/bin/env python3
"""
Check 24h trading volume for symbols across EdgeX and Lighter exchanges.
Displays results in a formatted table.
"""

import os
import json
import asyncio
from decimal import Decimal
from dotenv import load_dotenv
from edgex_sdk import Client as EdgeXClient
import lighter

# Load environment variables
load_dotenv()


def load_config(config_file="bot_config.json"):
    """Load configuration file"""
    with open(config_file, 'r') as f:
        return json.load(f)


async def get_edgex_volume(symbol: str, quote: str = "USD", debug: bool = False):
    """Get 24h volume from EdgeX for a symbol"""
    client = None
    try:
        # Initialize EdgeX client
        account_id = int(os.getenv("EDGEX_ACCOUNT_ID"))
        stark_private_key = os.getenv("EDGEX_STARK_PRIVATE_KEY")
        base_url = os.getenv("EDGEX_BASE_URL", "https://pro.edgex.exchange")

        client = EdgeXClient(
            base_url=base_url,
            account_id=account_id,
            stark_private_key=stark_private_key
        )

        # Get contract info
        contract_name = f"{symbol}{quote}"
        metadata = await client.get_metadata()
        contracts = metadata.get("data", {}).get("contractList", [])

        contract_id = None
        for contract in contracts:
            if contract.get("contractName") == contract_name:
                contract_id = contract.get("contractId")
                break

        if not contract_id:
            return None, "Contract not found"

        # Get 24h stats from quote endpoint
        try:
            quote_data = await client.quote.get_24_hour_quote(contract_id)
            if quote_data.get("code") == "SUCCESS" and quote_data.get("data"):
                record = quote_data["data"][0]

                if debug:
                    print(f"DEBUG {symbol}: Available fields: {list(record.keys())}")

                # EdgeX uses 'value' field for 24h trading volume in quote currency (USD)
                # and 'size' for base currency volume
                volume = None
                for field in ["value", "volume24h", "volume", "volume24H", "quoteVolume"]:
                    if field in record and record[field]:
                        volume = float(record[field])
                        if debug:
                            print(f"DEBUG {symbol}: Found volume in field '{field}': {volume}")
                        break

                if volume is not None:
                    return volume, None
        except Exception as e:
            if debug:
                print(f"DEBUG {symbol}: Error getting quote data: {e}")

        return None, "Volume data not available"

    except Exception as e:
        return None, f"Error: {str(e)}"
    finally:
        # Clean up client session
        if client and hasattr(client, 'internal_client'):
            try:
                await client.internal_client.close()
            except Exception:
                pass


async def get_lighter_volume(symbol: str, debug: bool = False):
    """Get 24h (daily) volume from Lighter for a symbol using ExchangeStats API"""
    api_client = None
    try:
        # Initialize Lighter client
        base_url = os.getenv("LIGHTER_BASE_URL") or os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")

        api_client = lighter.ApiClient(configuration=lighter.Configuration(host=base_url))
        order_api = lighter.OrderApi(api_client)

        # Get exchange statistics which includes daily volume per market
        stats_response = await order_api.exchange_stats()

        if debug:
            print(f"DEBUG Lighter: Retrieved stats for {len(stats_response.order_book_stats)} markets")

        # Find the market by symbol
        for market_stats in stats_response.order_book_stats:
            if market_stats.symbol.upper() == symbol.upper():
                # Get daily quote token volume (USD volume)
                volume = float(market_stats.daily_quote_token_volume)

                if debug:
                    print(f"DEBUG Lighter {symbol}: daily_quote_volume={volume}, daily_base_volume={market_stats.daily_base_token_volume}, trades={market_stats.daily_trades_count}")

                return volume, None

        # Symbol not found
        return None, "Market not found"

    except Exception as e:
        if debug:
            print(f"DEBUG Lighter {symbol}: Error: {e}")
        return None, f"Error: {str(e)}"
    finally:
        # Clean up client session
        if api_client:
            try:
                await api_client.close()
            except Exception:
                pass


def format_volume(volume):
    """Format volume for display"""
    if volume is None:
        return "N/A"

    if volume >= 1_000_000:
        return f"${volume/1_000_000:.2f}M"
    elif volume >= 1_000:
        return f"${volume/1_000:.2f}K"
    else:
        return f"${volume:.2f}"


async def main():
    """Main function to fetch and display volumes"""
    import sys

    # Check for debug flag
    debug = "--debug" in sys.argv

    # Load config
    config = load_config()
    symbols = config.get("symbols_to_monitor", [])
    quote = config.get("quote", "USD")

    print(f"\n{'='*80}")
    print(f"24H TRADING VOLUME COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Symbol':<12} {'EdgeX Volume':<20} {'Lighter Volume':<20} {'Total':<20}")
    print(f"{'-'*80}")

    total_edgex = 0
    total_lighter = 0

    for symbol in symbols:
        # Fetch volumes concurrently
        edgex_task = get_edgex_volume(symbol, quote, debug=debug)
        lighter_task = get_lighter_volume(symbol, debug=debug)

        edgex_result, lighter_result = await asyncio.gather(edgex_task, lighter_task)

        edgex_volume, edgex_error = edgex_result
        lighter_volume, lighter_error = lighter_result

        # Calculate total
        total_vol = None
        if edgex_volume is not None and lighter_volume is not None:
            total_vol = edgex_volume + lighter_volume
            total_edgex += edgex_volume
            total_lighter += lighter_volume
        elif edgex_volume is not None:
            total_vol = edgex_volume
            total_edgex += edgex_volume
        elif lighter_volume is not None:
            total_vol = lighter_volume
            total_lighter += lighter_volume

        # Format output
        edgex_str = format_volume(edgex_volume) if edgex_error is None else edgex_error
        lighter_str = format_volume(lighter_volume) if lighter_error is None else lighter_error
        total_str = format_volume(total_vol)

        print(f"{symbol:<12} {edgex_str:<20} {lighter_str:<20} {total_str:<20}")

    print(f"{'-'*80}")
    print(f"{'TOTALS':<12} {format_volume(total_edgex):<20} {format_volume(total_lighter):<20} {format_volume(total_edgex + total_lighter):<20}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
