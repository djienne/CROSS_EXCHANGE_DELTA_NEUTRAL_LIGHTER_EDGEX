import asyncio
import logging
import os
from dotenv import load_dotenv
from lighter_edgex_hedge import scan_all_account_positions, BotConfig, load_env

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    print("Loading environment...")
    env = load_env()
    
    print("Loading config...")
    config = BotConfig.load_from_file("bot_config.json")
    
    print("\nTesting scan_all_account_positions...")
    try:
        positions = await scan_all_account_positions(env, config)
        
        print(f"\nFound {len(positions)} positions:")
        for pos in positions:
            print(f"  Symbol: {pos['symbol']}")
            print(f"    EdgeX Size: {pos['edgex_size']}")
            print(f"    Lighter Size: {pos['lighter_size']}")
            print(f"    EdgeX Contract ID: {pos['edgex_contract_id']}")
            print(f"    Lighter Market ID: {pos['lighter_market_id']}")
            
    except Exception as e:
        print(f"\n❌ Error during scan: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
