#!/usr/bin/env python3
"""
Cancel specific orders on Lighter by getting order IDs first
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

print(f"Lighter Cancel Specific Orders")
print(f"Account Index: {account_index}")
print("="*70)

async def cancel_orders():
    """Get and cancel specific orders"""
    try:
        # Create API client to get orders
        api_client = lighter.ApiClient(lighter.Configuration(host=base_url))
        order_api = lighter.OrderApi(api_client)

        # Get active orders for SOL (market_id=2)
        # We know from earlier debug that SOL has 1 open order
        print("Fetching active orders for SOL (market_id=2)...")
        orders_response = await order_api.account_active_orders(
            account_index=account_index,
            market_id=2  # SOL market
        )

        print(f"Response code: {orders_response.code}")
        print(f"Total orders: {orders_response.total}")

        if orders_response.total > 0 and orders_response.orders:
            print(f"\nFound {len(orders_response.orders)} active orders:")
            for order in orders_response.orders:
                print(f"  Order ID: {order.id}, Symbol: {order.symbol if hasattr(order, 'symbol') else 'SOL'}, Size: {order.size}")

            # Create signer client to cancel orders
            print("\nCreating signer client to cancel orders...")
            client = lighter.SignerClient(
                url=base_url,
                private_key=private_key,
                api_key_index=api_key_index,
                account_index=account_index
            )

            import time
            for order in orders_response.orders:
                print(f"\nCanceling order {order.id} ({order.symbol})...")
                try:
                    result = await client.cancel_order(
                        order_id=order.id,
                        time=int(time.time())
                    )
                    print(f"  Result: {result}")
                except Exception as e:
                    print(f"  Error: {e}")

            print("\n" + "="*70)
            print("Verifying cancellation...")

            # Check again
            orders_after = await order_api.account_active_orders(
                account_index=account_index
            )
            print(f"Active orders remaining: {orders_after.total}")

        else:
            print("No active orders found")

        await api_client.close()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(cancel_orders())
