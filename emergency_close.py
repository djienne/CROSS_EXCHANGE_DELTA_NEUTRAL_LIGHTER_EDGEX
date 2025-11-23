#!/usr/bin/env python3
"""
Emergency Position Close Script

Immediately closes ALL open positions on both EdgeX and Lighter exchanges.
Use this script when you need to quickly exit all positions, regardless of
configuration files or normal workflow.

IMPORTANT: This script ONLY works on Linux/macOS due to Lighter SDK limitations.
On Windows, you MUST use Docker:
    docker-compose run emergency_close

Usage (Linux/macOS only):
    python emergency_close.py                    # Close all positions
    python emergency_close.py --dry-run          # Check positions without closing
    python emergency_close.py --cross-ticks 200  # Ultra-aggressive fills
    python emergency_close.py --help             # Show help
"""

import asyncio
import argparse
import sys
import os
import json
import platform
from datetime import datetime, timezone
from dotenv import load_dotenv

# Check if running on Windows
if platform.system() == "Windows":
    print("\n" + "="*70)
    print("ERROR: Cannot run on Windows")
    print("="*70)
    print("\nThe Lighter SDK only supports Linux and macOS.")
    print("Windows is NOT supported due to platform-specific dependencies.")
    print("\nTo close positions on Windows, use Docker:\n")
    print("  docker-compose run emergency_close              # Interactive mode")
    print("  docker-compose run emergency_close --dry-run    # Check positions")
    print("\nOr use hedge_cli.py via Docker:")
    print("  docker-compose run close                        # Close using config")
    print("\n" + "="*70)
    sys.exit(1)

# Import exchange client modules
import lighter_client
import edgex_client

# Load environment variables
load_dotenv()


def load_env():
    """Load and validate environment variables"""
    env = {}

    # EdgeX
    env["EDGEX_BASE_URL"] = os.getenv("EDGEX_BASE_URL", "https://pro.edgex.exchange")
    env["EDGEX_ACCOUNT_ID"] = os.getenv("EDGEX_ACCOUNT_ID")
    env["EDGEX_STARK_PRIVATE_KEY"] = os.getenv("EDGEX_STARK_PRIVATE_KEY")

    # Lighter (with fallbacks)
    env["LIGHTER_BASE_URL"] = os.getenv("LIGHTER_BASE_URL") or os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    env["API_KEY_PRIVATE_KEY"] = os.getenv("API_KEY_PRIVATE_KEY") or os.getenv("LIGHTER_PRIVATE_KEY")
    env["ACCOUNT_INDEX"] = int(os.getenv("ACCOUNT_INDEX", os.getenv("LIGHTER_ACCOUNT_INDEX", 0)))
    env["API_KEY_INDEX"] = int(os.getenv("API_KEY_INDEX", os.getenv("LIGHTER_API_KEY_INDEX", 0)))

    # Validate required fields
    if not env["EDGEX_ACCOUNT_ID"] or not env["EDGEX_STARK_PRIVATE_KEY"]:
        raise ValueError("EdgeX credentials not found in .env file")
    if not env["API_KEY_PRIVATE_KEY"]:
        raise ValueError("Lighter private key not found in .env file")

    return env


async def check_all_positions(env):
    """Check positions on both exchanges"""
    from edgex_sdk import Client as EdgeXClient
    import lighter

    positions = {
        'edgex': [],
        'lighter': []
    }

    # EdgeX positions
    try:
        edgex = EdgeXClient(
            base_url=env["EDGEX_BASE_URL"],
            account_id=int(env["EDGEX_ACCOUNT_ID"]),  # Must be int, not string
            stark_private_key=env["EDGEX_STARK_PRIVATE_KEY"]
        )

        # Get contract metadata to map contractId to market symbol
        metadata = await edgex.get_metadata()
        contract_map = {}  # contractId -> market symbol
        for contract in metadata.get("data", {}).get("contractList", []):
            # Keep contractId as integer (EdgeX SDK expects int, not string)
            # EdgeX uses 'contractName' not 'market'
            contract_map[contract.get("contractId")] = contract.get("contractName", "UNKNOWN")

        positions_response = await edgex.get_account_positions()
        positions_data = positions_response.get("data", {}).get("positionList", [])

        for pos in positions_data:
            # EdgeX API uses 'openSize' field for position size
            size = float(pos.get('openSize', 0))

            # Handle position side (short positions are negative)
            side = pos.get('side') or pos.get('positionSide')
            if side and str(side).lower().startswith('short'):
                size = -abs(size)

            if abs(size) > 1e-8:  # Only include non-zero positions
                # EdgeX API returns contractId as integer - keep it as integer
                contract_id = pos.get('contractId')
                symbol = contract_map.get(contract_id, 'UNKNOWN')

                positions['edgex'].append({
                    'symbol': symbol,
                    'contract_id': contract_id,
                    'size': size,
                    'entry_price': float(pos.get('entryPrice', 0)),
                    'unrealized_pnl': float(pos.get('unrealizedPnl', 0))
                })

        await edgex.close()
    except Exception as e:
        print(f"[X] Error fetching EdgeX positions: {e}")

    # Lighter positions
    try:
        api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
        account_api = lighter.AccountApi(api_client)

        account_index = env["ACCOUNT_INDEX"]
        positions['lighter'] = await lighter_client.get_all_lighter_positions(account_api, account_index)

        await api_client.close()
    except Exception as e:
        print(f"[X] Error fetching Lighter positions: {e}")

    return positions


async def emergency_close_all(dry_run=False, cross_ticks=100):
    """Close all positions on both exchanges"""

    print("\n" + "="*70)
    print("EMERGENCY POSITION CLOSE")
    print("="*70)
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Mode: {'DRY RUN (no trades)' if dry_run else 'LIVE TRADING'}")
    print(f"Aggressiveness: {cross_ticks} ticks")
    print("="*70 + "\n")

    # Load environment
    print("[+] Loading credentials...")
    try:
        env = load_env()
        print("[OK] Credentials loaded\n")
    except Exception as e:
        print(f"[X] Failed to load credentials: {e}")
        return False

    # Fetch positions
    print("[i] Fetching positions...\n")
    positions = await check_all_positions(env)

    # Display positions
    print(f"EdgeX Positions: {len(positions['edgex'])}")
    for pos in positions['edgex']:
        pnl_color = "[+]" if pos['unrealized_pnl'] >= 0 else "[-]"
        print(f"  {pnl_color} {pos['symbol']}: {pos['size']:.4f} @ ${pos['entry_price']:.2f} (PnL: ${pos['unrealized_pnl']:.2f})")

    print(f"\nLighter Positions: {len(positions['lighter'])}")
    for pos in positions['lighter']:
        pnl_color = "[+]" if pos['unrealized_pnl'] >= 0 else "[-]"
        print(f"  {pnl_color} {pos['symbol']}: {pos['size']:.4f} @ ${pos['entry_price']:.2f} (PnL: ${pos['unrealized_pnl']:.2f})")

    # Check if there are any positions
    total_positions = len(positions['edgex']) + len(positions['lighter'])

    if total_positions == 0:
        print("\n[OK] No open positions found. Nothing to close.")
        return True

    print(f"\n[!]  Total positions to close: {total_positions}")

    if dry_run:
        print("\n[?] DRY RUN MODE - No orders will be placed")
        return True

    # Confirm before proceeding
    print("\n" + "="*70)
    print("[!]  WARNING: This will close ALL positions immediately!")
    print("="*70)
    print("\nPress ENTER to confirm and proceed (or Ctrl+C to abort)")
    response = input(">>> ")

    # Any input (including empty) confirms, we just need user interaction
    print("[OK] Confirmed, proceeding with close...")

    # Close positions directly on each exchange
    print("\n[!!] CLOSING ALL POSITIONS...\n")

    from edgex_sdk import Client as EdgeXClient
    import lighter

    # Initialize exchange clients once
    edgex = EdgeXClient(
        base_url=env["EDGEX_BASE_URL"],
        account_id=int(env["EDGEX_ACCOUNT_ID"]),  # Must be int, not string
        stark_private_key=env["EDGEX_STARK_PRIVATE_KEY"]
    )

    api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
    order_api = lighter.OrderApi(api_client)
    account_api = lighter.AccountApi(api_client)
    signer = lighter.SignerClient(
        url=env["LIGHTER_BASE_URL"],
        private_key=env["API_KEY_PRIVATE_KEY"],
        account_index=env["ACCOUNT_INDEX"],
        api_key_index=env["API_KEY_INDEX"],
    )

    success_count = 0
    fail_count = 0

    # Get EdgeX contract metadata once
    try:
        metadata = await edgex.get_metadata()
        contract_lookup = {}  # contractId -> contract details
        for c in metadata.get("data", {}).get("contractList", []):
            # Keep contractId as integer (EdgeX SDK expects int, not string)
            contract_lookup[c.get("contractId")] = {
                'contractName': c.get("contractName"),
                'tickSize': float(c.get("tickSize", 0.01)),
                'stepSize': float(c.get("stepSize", 0.001))
            }
    except Exception as e:
        print(f"[X] Failed to get EdgeX metadata: {e}")
        contract_lookup = {}

    # Close EdgeX positions
    for pos in positions['edgex']:
        symbol_market = pos['symbol']
        size = pos['size']
        contract_id = pos['contract_id']

        print(f"Closing EdgeX {symbol_market}: {size:+.6f}")

        try:
            # Get contract details from lookup
            contract_details = contract_lookup.get(contract_id)
            if not contract_details:
                print(f"  [X] Contract details not found for {symbol_market}")
                fail_count += 1
                continue

            tick_size = contract_details['tickSize']
            step_size = contract_details['stepSize']

            # Use edgex_client.close_position() for proper closing
            order_id = await edgex_client.close_position(
                edgex,
                contract_id,
                tick_size,
                step_size,
                cross_ticks=cross_ticks
            )

            if order_id:
                print(f"  [OK] EdgeX {symbol_market} closed (order: {order_id})")
                success_count += 1
            else:
                print(f"  [OK] EdgeX {symbol_market} already flat")
                success_count += 1

        except Exception as e:
            print(f"  [X] EdgeX {symbol_market} failed: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    # Close Lighter positions
    for pos in positions['lighter']:
        symbol = pos['symbol']
        size = pos['size']

        print(f"\nClosing Lighter {symbol}: {size:+.6f}")

        try:
            # Get market details
            market_id, price_tick, amount_tick = await lighter_client.get_lighter_market_details(order_api, symbol)

            # Get best bid/ask
            best_bid, best_ask = await lighter_client.get_lighter_best_bid_ask(order_api, symbol, market_id)

            # Determine close side and reference price
            close_side = "sell" if size > 0 else "buy"
            ref_price = best_bid if close_side == "sell" else best_ask

            if not ref_price:
                print(f"  [X] No reference price available for {symbol}")
                fail_count += 1
                continue

            # Close position using lighter_client function
            success = await lighter_client.lighter_close_position(
                signer=signer,
                market_id=market_id,
                price_tick=price_tick,
                amount_tick=amount_tick,
                side=close_side,
                size_base=abs(size),
                ref_price=ref_price,
                cross_ticks=cross_ticks
            )

            if success:
                print(f"  [OK] Lighter {symbol} closed")
                success_count += 1
            else:
                print(f"  [X] Lighter {symbol} close order failed")
                fail_count += 1

        except Exception as e:
            print(f"  [X] Lighter {symbol} failed: {e}")
            fail_count += 1

    # Close clients
    await edgex.close()
    await api_client.close()

    print()

    # Summary
    print("="*70)
    print("CLOSE SUMMARY")
    print("="*70)
    print(f"[OK] Successful closes: {success_count}")
    print(f"[X] Failed closes:     {fail_count}")
    print(f"[i] Total positions:   {total_positions}")
    print("="*70)

    if fail_count > 0:
        print("\n[!]  Some positions failed to close. Please check manually!")
        return False
    else:
        print("\n[OK] All positions closed successfully!")
        print("[...] Orders should fill within seconds. Verify on exchanges.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Emergency close all positions on EdgeX and Lighter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python emergency_close.py                    # Close all positions (with confirmation)
  python emergency_close.py --dry-run          # Check positions without closing
  python emergency_close.py --cross-ticks 200  # Ultra-aggressive fills
  python emergency_close.py --cross-ticks 50   # Less aggressive (slower)

Safety:
  - Dry-run mode available to check positions first
  - Requires pressing ENTER to confirm (in live mode)
  - Uses aggressive limit orders for fast execution
  - Higher cross-ticks = faster fills but worse prices
  - Uses lighter_client.py functions for Lighter position closing
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Check positions without placing close orders'
    )

    parser.add_argument(
        '--cross-ticks',
        type=int,
        default=100,
        help='How many ticks to cross the spread (default: 100, higher = faster fills)'
    )

    args = parser.parse_args()

    # Run the emergency close
    try:
        success = asyncio.run(emergency_close_all(
            dry_run=args.dry_run,
            cross_ticks=args.cross_ticks
        ))

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n[X] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[X] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
