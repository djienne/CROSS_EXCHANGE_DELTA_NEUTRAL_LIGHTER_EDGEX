#!/usr/bin/env python3
"""
Debug script to test Lighter position detection
"""
import asyncio
import json
import os
from dotenv import load_dotenv
import lighter

load_dotenv()

# Get env vars
base_url = os.getenv("LIGHTER_BASE_URL") or os.getenv("BASE_URL") or "https://mainnet.zklighter.elliot.ai"
account_index = int(os.getenv("LIGHTER_ACCOUNT_INDEX") or os.getenv("ACCOUNT_INDEX") or "0")

print(f"Using BASE_URL: {base_url}")
print(f"Using account_index: {account_index}")
print()

async def test_positions():
    """Test different ways to get positions"""

    # Method 1: Using AccountApi.account()
    print("="*70)
    print("Method 1: AccountApi.account(by='index', value=str(account_index))")
    print("="*70)

    api_client = lighter.ApiClient(lighter.Configuration(host=base_url))
    account_api = lighter.AccountApi(api_client)

    try:
        response = await account_api.account(by="index", value=str(account_index))

        print(f"Response code: {response.code}")
        print(f"Response message: {response.message}")
        print(f"Total accounts: {response.total}")
        print(f"Number of accounts: {len(response.accounts)}")
        print()

        if response.accounts:
            acc = response.accounts[0]
            print(f"Account Index: {acc.account_index if hasattr(acc, 'account_index') else acc.index}")
            print(f"L1 Address: {acc.l1_address if hasattr(acc, 'l1_address') else 'N/A'}")
            print(f"Number of positions: {len(acc.positions) if acc.positions else 0}")
            print()

            if acc.positions:
                print("Positions:")
                for i, pos in enumerate(acc.positions):
                    print(f"\n  Position {i+1}:")
                    print(f"    market_id: {pos.market_id}")
                    print(f"    symbol: {pos.symbol}")
                    print(f"    position: {pos.position}")
                    print(f"    sign: {pos.sign}")
                    print(f"    avg_entry_price: {pos.avg_entry_price}")
                    print(f"    unrealized_pnl: {pos.unrealized_pnl}")
                    print(f"    open_order_count: {pos.open_order_count}")

                    # Calculate signed size
                    raw_size = float(pos.position or "0")
                    sign = int(pos.sign or 0)
                    signed_size = raw_size * sign
                    print(f"    CALCULATED signed_size: {signed_size}")

                    # Check if would be filtered
                    print(f"    Would be filtered (abs < 1e-8): {abs(signed_size) < 1e-8}")
            else:
                print("No positions returned")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await api_client.close()

    print()
    print("="*70)
    print("Method 2: Check raw JSON response")
    print("="*70)

    # Try making raw API call to see actual response
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/v1/accounts?by=index&value={account_index}"
            print(f"URL: {url}")
            async with session.get(url) as resp:
                text = await resp.text()
                print(f"Status: {resp.status}")
                print(f"Raw response:")
                try:
                    data = json.loads(text)
                    print(json.dumps(data, indent=2))
                except:
                    print(text)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_positions())
