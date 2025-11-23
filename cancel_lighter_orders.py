#!/usr/bin/env python3
"""
Cancel all open orders on Lighter
"""
import asyncio
import os
from dotenv import load_dotenv
import lighter

load_dotenv()

# Get env vars
base_url = os.getenv("LIGHTER_BASE_URL") or os.getenv("BASE_URL") or "https://mainnet.zklighter.elliot.ai"
private_key = os.getenv("LIGHTER_PRIVATE_KEY") or os.getenv("API_KEY_PRIVATE_KEY")
account_index = int(os.getenv("LIGHTER_ACCOUNT_INDEX") or os.getenv("ACCOUNT_INDEX") or "0")
api_key_index = int(os.getenv("LIGHTER_API_KEY_INDEX") or os.getenv("API_KEY_INDEX") or "0")

print(f"Lighter Cancel All Orders")
print(f"Account Index: {account_index}")
print("="*70)

async def cancel_all_lighter_orders():
    """Cancel all open orders on Lighter"""
    try:
        # Create signer client
        client = lighter.SignerClient(
            url=base_url,
            private_key=private_key,
            api_key_index=api_key_index,
            account_index=account_index
        )

        print("Canceling all open orders...")

        # Cancel all orders with immediate time-in-force
        import time
        current_time = int(time.time())

        try:
            result = await client.cancel_all_orders(
                time_in_force=client.CANCEL_ALL_TIF_IMMEDIATE,
                time=current_time
            )
            print(f"Result: {result}")
        except Exception as e:
            # SDK may return None or error, but order might still be canceled
            print(f"Cancel result: {e}")

        # Verify orders were canceled by checking account
        print("\nVerifying...")
        api_client = lighter.ApiClient(lighter.Configuration(host=base_url))
        account_api = lighter.AccountApi(api_client)

        account_details = await account_api.account(by="index", value=str(account_index))

        if account_details.accounts:
            acc = account_details.accounts[0]
            total_open_orders = sum(pos.open_order_count for pos in acc.positions if pos.open_order_count > 0)

            print(f"Total open orders after cancel: {total_open_orders}")

            if total_open_orders > 0:
                print("\n⚠ Orders still exist:")
                for pos in acc.positions:
                    if pos.open_order_count > 0:
                        print(f"  {pos.symbol}: {pos.open_order_count} open orders")
            else:
                print("✓ All orders canceled successfully")

        await api_client.close()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(cancel_all_lighter_orders())
