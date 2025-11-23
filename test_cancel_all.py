#!/usr/bin/env python3
"""
Test the cancel_all_lighter_orders function
"""
import asyncio
import os
from dotenv import load_dotenv
from lighter_client import cancel_all_lighter_orders

load_dotenv()

async def main():
    env = dict(os.environ)
    result = await cancel_all_lighter_orders(env)
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())
