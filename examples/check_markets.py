#!/usr/bin/env python3
"""Quick script to check what markets are available on Lighter."""
import asyncio
import os
from dotenv import load_dotenv
import lighter

async def main():
    load_dotenv()

    lighter_base_url = os.getenv("LIGHTER_BASE_URL", "https://mainnet.zklighter.elliot.ai")

    api_client = lighter.ApiClient(configuration=lighter.Configuration(host=lighter_base_url))
    order_api = lighter.OrderApi(api_client)

    print("\nFetching Lighter markets...")
    print("=" * 80)

    try:
        resp = await order_api.order_books()

        print(f"\nFound {len(resp.order_books)} markets:\n")

        for ob in resp.order_books:
            symbol = ob.symbol
            bids = getattr(ob, "bids", [])
            asks = getattr(ob, "asks", [])

            has_liquidity = "✓" if (bids and asks) else "✗"
            bid_count = len(bids)
            ask_count = len(asks)

            best_bid = float(bids[0].price) if bids else None
            best_ask = float(asks[0].price) if asks else None

            if best_bid and best_ask:
                mid = (best_bid + best_ask) / 2
                print(f"{has_liquidity} {symbol:10} - Bids: {bid_count:3}, Asks: {ask_count:3}, Mid: ${mid:,.2f}")
            else:
                print(f"{has_liquidity} {symbol:10} - Bids: {bid_count:3}, Asks: {ask_count:3}, No prices")

        print("\n" + "=" * 80)
        print("Legend: ✓ = Has liquidity, ✗ = Empty order book\n")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await api_client.close()

if __name__ == "__main__":
    asyncio.run(main())
