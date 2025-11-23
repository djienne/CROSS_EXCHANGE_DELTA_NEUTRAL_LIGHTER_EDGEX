#!/usr/bin/env python3
"""
Debug script to test Lighter position detection using SignerClient
"""
import asyncio
import json
import os
from dotenv import load_dotenv
import lighter

load_dotenv()

# Get env vars
base_url = os.getenv("LIGHTER_BASE_URL") or os.getenv("BASE_URL") or "https://mainnet.zklighter.elliot.ai"
private_key = os.getenv("LIGHTER_PRIVATE_KEY") or os.getenv("API_KEY_PRIVATE_KEY")
account_index = int(os.getenv("LIGHTER_ACCOUNT_INDEX") or os.getenv("ACCOUNT_INDEX") or "0")
api_key_index = int(os.getenv("LIGHTER_API_KEY_INDEX") or os.getenv("API_KEY_INDEX") or "0")

print(f"Using BASE_URL: {base_url}")
print(f"Using account_index: {account_index}")
print(f"Using api_key_index: {api_key_index}")
print()

async def test_signer_client():
    """Test using SignerClient for authenticated access"""

    print("="*70)
    print("Method: SignerClient with authentication")
    print("="*70)

    try:
        # Create signer client
        client = lighter.SignerClient(
            url=base_url,
            private_key=private_key,
            api_key_index=api_key_index,
            account_index=account_index
        )

        print(f"SignerClient created successfully")
        print()

        # Try getting account by index
        print(f"Fetching account by index ({account_index})...")
        account_details = await client.api.account_api.account(by="index", value=str(account_index))

        print(f"Response code: {account_details.code}")
        print(f"Total accounts: {account_details.total}")
        print()

        if account_details.accounts:
            acc = account_details.accounts[0]
            print(f"Account Index: {acc.account_index}")
            print(f"L1 Address: {acc.l1_address}")
            print(f"Total Asset Value: {acc.total_asset_value}")
            print(f"Cross Asset Value: {acc.cross_asset_value}")
            print(f"Number of positions: {len(acc.positions) if acc.positions else 0}")
            print()

            if acc.positions:
                print("All Positions (showing non-zero only):")
                for i, pos in enumerate(acc.positions):
                    raw_size = float(pos.position or "0")
                    sign = int(pos.sign or 0)
                    signed_size = raw_size * sign

                    # Only show non-zero positions or those with open orders
                    if abs(signed_size) > 1e-8 or pos.open_order_count > 0:
                        print(f"\n  Position: {pos.symbol}")
                        print(f"    market_id: {pos.market_id}")
                        print(f"    position (raw): {pos.position}")
                        print(f"    sign: {pos.sign}")
                        print(f"    signed_size: {signed_size}")
                        print(f"    avg_entry_price: {pos.avg_entry_price}")
                        print(f"    unrealized_pnl: {pos.unrealized_pnl}")
                        print(f"    open_order_count: {pos.open_order_count}")
                        print(f"    pending_order_count: {pos.pending_order_count}")

                # Summary
                non_zero = [(pos.symbol, float(pos.position or "0") * int(pos.sign or 0))
                            for pos in acc.positions
                            if abs(float(pos.position or "0") * int(pos.sign or 0)) > 1e-8]

                if non_zero:
                    print("\n" + "="*70)
                    print("SUMMARY - Non-zero positions:")
                    print("="*70)
                    for symbol, size in non_zero:
                        print(f"  {symbol}: {size}")
                else:
                    print("\n" + "="*70)
                    print("SUMMARY: All positions are ZERO")
                    print("="*70)
            else:
                print("No positions array or empty")

        await client.api.close()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_signer_client())
