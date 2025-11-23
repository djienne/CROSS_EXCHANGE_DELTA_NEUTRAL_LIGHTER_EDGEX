#!/usr/bin/env python3
"""
Cancel all orders on Lighter using REST API endpoint
"""
import asyncio
import os
import aiohttp
import json
from dotenv import load_dotenv
import lighter
from eth_account import Account
from eth_account.messages import encode_defunct

load_dotenv()

# Get env vars
base_url = os.getenv("LIGHTER_BASE_URL") or os.getenv("BASE_URL") or "https://mainnet.zklighter.elliot.ai"
private_key = os.getenv("LIGHTER_PRIVATE_KEY") or os.getenv("API_KEY_PRIVATE_KEY")
account_index = int(os.getenv("LIGHTER_ACCOUNT_INDEX") or os.getenv("ACCOUNT_INDEX") or "0")
api_key_index = int(os.getenv("LIGHTER_API_KEY_INDEX") or os.getenv("API_KEY_INDEX") or "0")

print(f"Lighter Cancel All Orders (REST API)")
print(f"Account Index: {account_index}")
print("="*70)

async def cancel_all_via_rest():
    """Cancel all orders using REST API endpoint"""
    try:
        # Create signer client for authentication
        client = lighter.SignerClient(
            url=base_url,
            private_key=private_key,
            api_key_index=api_key_index,
            account_index=account_index
        )

        # Create authentication token
        import time
        deadline = int(time.time()) + 60  # 60 seconds from now
        auth_token = client.create_auth_token_with_expiry(deadline=deadline)

        print(f"Auth token created (deadline: {deadline})")
        print(f"Calling cancel all endpoint...")

        # Call the REST API endpoint directly
        url = f"{base_url}/api/v1/private/order/cancelAllOrder"

        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }

        # Empty body means cancel ALL orders
        body = {
            "filterCoinIdList": [],
            "filterContractIdList": [],
            "filterOrderTypeList": [],
            "filterOrderStatusList": [],
            "filterIsPositionTpsl": []
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=body) as resp:
                text = await resp.text()
                print(f"\nResponse status: {resp.status}")
                print(f"Response: {text}")

                try:
                    data = json.loads(text)
                    print(json.dumps(data, indent=2))

                    if resp.status == 200:
                        print("\n✓ Cancel all orders request sent successfully")
                    else:
                        print(f"\n⚠ Request returned status {resp.status}")
                except:
                    pass

        # Verify by checking account
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
    asyncio.run(cancel_all_via_rest())
