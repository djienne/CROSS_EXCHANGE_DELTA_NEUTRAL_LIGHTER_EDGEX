import asyncio
import json
import logging
import os
import platform
import signal
import sys
import time
import math
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

from edgex_sdk import Client, OrderSide, OrderType, TimeInForce, WebSocketManager, CreateOrderParams, CancelOrderParams, GetActiveOrderParams, GetOrderBookDepthParams, OrderFillTransactionParams

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging(debug_to_file: bool = True, log_file: str = "log.txt"):
    """Setup logging configuration with optional file debug logging."""
    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug_to_file else logging.INFO)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (DEBUG level) - optional
    if debug_to_file:
        try:
            # Use Path for cross-platform file handling
            log_path = Path(log_file)
            file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

            # Log startup message
            startup_msg = f"\n{'='*80}\nTRADING BOT STARTUP - {datetime.now().isoformat()}\n{'='*80}"
            root_logger.debug(startup_msg)

        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")

# Setup logging based on environment variable
debug_logging = os.getenv("EDGEX_DEBUG_LOGGING", "true").lower() in ("true", "1", "yes", "on")
log_file_path = os.getenv("EDGEX_LOG_FILE", "log.txt")
setup_logging(debug_to_file=debug_logging, log_file=log_file_path)

logger = logging.getLogger(__name__)


# --- TRADING CONFIGURATION ---
# Default leverage for trading
LEVERAGE = 3.0
# Time in seconds between each trading loop iteration
REFRESH_INTERVAL = 10
# Minimum time in seconds to wait between placing orders
MIN_ORDER_INTERVAL = 3.0
# Time in seconds to wait after startup before placing the first order
STARTUP_DELAY = 2
# The amount to trade. Unit is determined by ORDER_SIZE_IN_QUOTE.
# This value is used as a fallback if CAPITAL_USAGE_RATIO is set to None or 0.
TRADE_AMOUNT = 45.0
# The percentage of available capital to use for each trade (e.g., 0.95 for 95%).
# If set to a value between 0 and 1, this overrides TRADE_AMOUNT.
# Set to None or 0 to use the fixed TRADE_AMOUNT.
CAPITAL_USAGE_RATIO = 0.4
# Set to True to interpret TRADE_AMOUNT as quote currency (USD), False for base currency (e.g., PAXG)
ORDER_SIZE_IN_QUOTE = True
# Time to wait after cancelling an order before placing a new one, to allow balance to update
POST_CANCELLATION_DELAY = 0.25
# Balance snapshot file path
BALANCE_SNAPSHOT_FILE = "balance_snapshots.txt"

# --- SPREAD CONFIGURATION ---
# Set to True to use the advanced spread calculation model.
USE_ADVANCED_SPREAD = True
# Path to the spread parameters JSON file.
SPREAD_PARAMS_FILE = "params/spread_parameters_PAXG.json"
# Fallback spread in ticks if advanced calculation fails.
FALLBACK_TICK_SPREAD = 10
# --- END TRADING CONFIGURATION ---



class EdgeXTradingBot:
    def __init__(self, base_url: str, ws_url: str, account_id: int, stark_private_key: str, leverage: float = LEVERAGE, use_advanced_spread: bool = USE_ADVANCED_SPREAD, spread_params_file: str = SPREAD_PARAMS_FILE):
        """
        Initialize the EdgeX Trading Bot.
        
        Args:
            base_url: EdgeX API base URL
            ws_url: EdgeX WebSocket URL
            account_id: Your account ID
            stark_private_key: Your Stark private key
            leverage: Trading leverage (default: from config)
            use_advanced_spread: Whether to use the advanced spread model
            spread_params_file: Path to the spread parameters file
        """
        self.base_url = base_url
        self.ws_url = ws_url
        self.account_id = account_id
        self.stark_private_key = stark_private_key
        self.leverage = leverage
        self.use_advanced_spread = use_advanced_spread
        self.spread_params_file = spread_params_file
        self.spread_params = None
        
        # Trading state
        self.contract_id = None  # Will be set for PAXGUSD
        self.current_position = None
        self.active_orders: Dict[str, Optional[str]] = {"BUY": None, "SELL": None}
        self.last_order_price = None
        self.current_bid = None
        self.current_ask = None
        self.order_book = None
        self.market_mid_price = None

        # Contract specifications
        self.tick_size = None  # Price precision
        self.step_size = None  # Quantity precision
        self.min_order_size = None
        self.taker_fee_rate = None  # Fee rate for market orders
        self.maker_fee_rate = None  # Fee rate for limit orders
        self.price_decimal_places = 0

        # Account state
        self.account_balance = None
        self.available_balance = None
        self.account_data = None
        
        # Clients
        self.client = None
        self.ws_manager = None
        
        # Control flags
        self.running = False
        self.stopping = False
        self.trading_allowed = True  # Set to False when volume limit is exceeded
        self.refresh_interval = REFRESH_INTERVAL

        # Timing control
        self.last_order_time = None  # Track when we last placed an order
        self.min_order_interval = MIN_ORDER_INTERVAL
        self.startup_delay = STARTUP_DELAY
        self.bot_start_time = None  # Track when bot started
        self.last_ticker_update_time: Optional[float] = None
        self.last_balance_snapshot_time = None  # Track when we last saved a balance snapshot

    async def initialize(self):
        """Initialize the trading bot."""
        logger.debug(f"Starting bot initialization with base_url={self.base_url}, ws_url={self.ws_url}, account_id={self.account_id}")
        try:
            # Initialize REST client only if we have valid credentials
            if self.account_id and self.stark_private_key and str(self.account_id) != "0":
                logger.debug(f"Initializing REST client with account_id={self.account_id}")
                self.client = Client(
                    base_url=self.base_url,
                    account_id=self.account_id,
                    stark_private_key=self.stark_private_key
                )
                logger.debug("REST client created successfully")
            else:
                logger.warning("No valid credentials provided - running in market data mode only")
                logger.debug(f"Credential check failed: account_id={self.account_id}, has_private_key={bool(self.stark_private_key)}")
                self.client = None
            
            # Test authentication by getting metadata
            try:
                logger.debug("Testing authentication by fetching metadata")
                metadata = await self.client.get_metadata()
                logger.debug(f"Metadata response: {metadata}")
                if metadata.get("code") == "SUCCESS":
                    logger.info("REST client authenticated successfully")
                    logger.debug(f"Metadata contains {len(metadata.get('data', {}).get('contractList', []))} contracts")
                else:
                    logger.warning(f"Authentication issue: {metadata.get('msg', 'Unknown error')}")
                    logger.warning(f"Continuing with limited functionality (market data only)")
                    logger.debug(f"Full metadata response: {metadata}")
            except Exception as e:
                logger.warning(f"Authentication failed: {e}")
                logger.debug(f"Authentication exception details: {type(e).__name__}: {e}")
                logger.warning(f"Continuing with limited functionality (market data only)")
                # Don't raise here, continue with WebSocket for market data
            
            # Get contract ID for PAXGUSD
            await self._find_paxgusd_contract()

            # Load spread parameters if using the advanced model
            if self.use_advanced_spread:
                self._load_spread_parameters()

            # Check account state (positions and balance)
            if self.client:
                # Fetch position first, as it affects the balance check
                logger.info("🔍 Fetching initial position state...")
                self.current_position = await self._get_current_position()
                if self.current_position:
                    size = self.current_position.get("size", "0")
                    logger.info(f"✅ Found existing position at startup. Size: {size}")
                else:
                    logger.info("✅ No existing position found at startup.")

                # Now check balance
                logger.info("🔍 Checking account balance...")
                balance_success = await self._get_account_balance()
                if not balance_success:
                    logger.warning("⚠️ Could not retrieve account balance - continuing anyway")
                elif self.available_balance and self.available_balance < 15.0:
                    if not self.current_position:
                        logger.error(f"❌ INSUFFICIENT CAPITAL: Available balance ${self.available_balance:.2f} USD is below minimum required $15 USD and no open position to close.")
                        logger.error("🛑 Trading bot stopped due to insufficient capital.")
                        raise Exception(f"Insufficient capital for trading: ${self.available_balance:.2f} USD < $15 USD minimum and no open position.")
                    else:
                        logger.warning(f"⚠️ Low available balance: ${self.available_balance:.2f} USD, but allowing bot to run to close open position.")
                elif self.available_balance and self.available_balance < 50.0:
                    logger.warning(f"⚠️ Low available balance: {self.available_balance} USD - may not be sufficient for trading")

            # Set leverage if it differs from the current setting on the exchange
            current_leverage = None
            if self.account_data:
                trade_settings = self.account_data.get("contractIdToTradeSetting", {})
                contract_settings = trade_settings.get(self.contract_id)
                if contract_settings and contract_settings.get("isSetMaxLeverage"):
                    current_leverage = float(contract_settings.get("maxLeverage", "1.0"))
            
            if current_leverage is None:
                logger.warning("Could not determine current leverage from API. Will attempt to set leverage if configured value is not 1.0.")
                if self.leverage != 1.0:
                    await self._set_leverage()
            elif float(self.leverage) != current_leverage:
                logger.info(f"Current exchange leverage is {current_leverage}x, configured leverage is {self.leverage}x. Updating...")
                await self._set_leverage()
            else:
                logger.info(f"Leverage is already correctly set to {self.leverage}x. No action needed.")
            
            # Initialize WebSocket manager with proper credentials if available
            if self.client and self.account_id and self.stark_private_key:
                self.ws_manager = WebSocketManager(
                    base_url=self.ws_url,
                    account_id=self.account_id,
                    stark_pri_key=self.stark_private_key
                )
            else:
                # Use dummy values for public connection only
                self.ws_manager = WebSocketManager(
                    base_url=self.ws_url,
                    account_id=0,  # Use 0 for public data
                    stark_pri_key=""  # Empty for public data
                )
            
            # Connect to WebSocket
            await self._setup_websocket()

            # Determine the contract name for logging
            final_contract_name = "PAXGUSD"
            if self.contract_id != "10000001": # A bit of a hardcode, but reflects the fallback logic
                try:
                    metadata = await self.client.get_metadata()
                    contracts = metadata.get("data", {}).get("contractList", [])
                    for contract in contracts:
                        if contract.get("contractId") == self.contract_id:
                            final_contract_name = contract.get("contractName")
                            break
                except Exception:
                    pass # Stick with default name if metadata fails

            logger.info(f"🚀 Trading bot initialized for {final_contract_name} (Contract ID: {self.contract_id})")

            if not self.client:
                logger.info("⚠️  Running in MARKET DATA mode only - fix credentials to enable trading")

            self.last_ticker_update_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}")
            raise

    async def _find_paxgusd_contract(self):
        """Find the contract ID for PAXGUSD."""
        logger.debug("Searching for PAXGUSD contract")
        try:
            metadata = await self.client.get_metadata()
            contracts = metadata.get("data", {}).get("contractList", [])
            logger.debug(f"Found {len(contracts)} contracts in metadata")

            for i, contract in enumerate(contracts):
                contract_name = contract.get("contractName")
                contract_id = contract.get("contractId")
                logger.debug(f"Contract {i}: {contract_name} (ID: {contract_id})")

                if contract_name == "PAXGUSD":
                    self.contract_id = contract_id
                    self.tick_size = float(contract.get("tickSize", "0.01"))
                    self.step_size = float(contract.get("stepSize", "0.001"))
                    self.min_order_size = float(contract.get("minOrderSize", "0.01"))
                    self.taker_fee_rate = float(contract.get("defaultTakerFeeRate", "0.00038"))
                    self.maker_fee_rate = float(contract.get("defaultMakerFeeRate", "0.00015"))

                    # Determine the number of decimal places for price formatting
                    if self.tick_size > 0:
                        tick_size_str = f"{self.tick_size:.10f}".rstrip('0')
                        if '.' in tick_size_str:
                            self.price_decimal_places = len(tick_size_str.split('.')[1])
                        else:
                            self.price_decimal_places = 0

                    logger.info(f"Found PAXGUSD contract ID: {self.contract_id}")
                    logger.debug(f"Contract specs - tick_size: {self.tick_size}, step_size: {self.step_size}, min_order_size: {self.min_order_size}, price_decimals: {self.price_decimal_places}")
                    logger.debug(f"Fee rates - taker: {self.taker_fee_rate}, maker: {self.maker_fee_rate}")
                    logger.debug(f"PAXGUSD contract details: {contract}")
                    return

            # Fallback: use a common contract ID if PAXGUSD not found
            logger.warning("PAXGUSD contract not found, using fallback contract ID")
            logger.debug(f"Available contracts: {[c.get('contractName') for c in contracts]}")
            self.contract_id = "10000001"  # Example fallback

        except Exception as e:
            logger.error(f"Failed to find PAXGUSD contract: {e}")
            logger.debug(f"Contract search exception: {type(e).__name__}: {e}")
            raise

    def _round_price_to_tick_size(self, price: float) -> float:
        """Round price to the contract's tick size."""
        if not self.tick_size:
            logger.warning("No tick size available, using price as-is")
            return price
        return round(price / self.tick_size) * self.tick_size

    def _round_size_to_step_size(self, size: float) -> float:
        """Round order size to the contract's step size."""
        if not self.step_size:
            logger.warning("No step size available, using size as-is")
            return size
        return round(size / self.step_size) * self.step_size

    async def _get_account_balance(self) -> bool:
        """Get account balance and update state. Returns True if successful."""
        if not self.client:
            logger.warning("No client available for balance check")
            return False

        try:
            logger.debug("Using get_account_asset() method to get balance information")
            balance_response = await self.client.get_account_asset()
            logger.debug(f"Balance response: {balance_response}")

            if balance_response.get("code") != "SUCCESS":
                logger.warning(f"Failed to get account balance: {balance_response.get('msg', 'Unknown error')}")
                return False

            balance_data = balance_response.get("data", {})
            self.account_data = balance_data.get("account")
            logger.debug(f"Full account data object: {self.account_data}")

            # --- Total & Available Balance Calculation ---
            # Prioritize using collateralAssetModelList as it seems to represent actual wallet funds
            # without unrealized PnL, which is safer for risk calculations.
            
            collateral_asset_list = balance_data.get("collateralAssetModelList", [])
            if collateral_asset_list:
                logger.debug("Calculating balance from 'collateralAssetModelList'.")
                total_balance_from_assets = 0.0
                available_balance_from_assets = 0.0
                
                for asset in collateral_asset_list:
                    if asset.get("coinId") == "1000": # USD
                        total_balance_from_assets += float(asset.get("amount", "0"))
                        available_balance_from_assets += float(asset.get("availableAmount", "0"))

                self.account_balance = total_balance_from_assets
                self.available_balance = available_balance_from_assets
                logger.debug(f"Derived from assets - Total: {self.account_balance}, Available: {self.available_balance}")

            else:
                # Fallback to the old method if collateralAssetModelList is not present
                logger.warning("Could not find 'collateralAssetModelList'. Falling back to potentially inaccurate balance sources.")
                total_balance_str = None
                if self.account_data and 'totalWalletBalance' in self.account_data:
                    total_balance_str = self.account_data.get('totalWalletBalance')
                    logger.debug(f"Using 'totalWalletBalance' from account object for total balance: {total_balance_str}")
                else:
                    logger.debug("Field 'totalWalletBalance' not found. Falling back to collateralList.")
                    collateral_list = balance_data.get("collateralList", [])
                    for balance in collateral_list:
                        if balance.get("coinId") == "1000":  # USD
                            total_balance_str = balance.get("amount", "0")
                            logger.warning("Using 'amount' from collateralList for total balance. This may be equity, not wallet balance.")
                            break
                
                if total_balance_str is None:
                    logger.error("Could not determine total account balance.")
                    return False
                
                self.account_balance = float(total_balance_str)
                # Also attempt to set available balance, though it might be missing
                self.available_balance = self.account_balance 
                logger.warning("Available balance is assumed to be equal to total balance in fallback mode.")

            logger.info(f"💰 Account Balance - Total: {self.account_balance:.4f} USD, Available: {self.available_balance:.4f} USD")
            return True

        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            logger.debug(f"Balance check exception: {type(e).__name__}: {e}")
            return False

    def _calculate_order_fee(self, price: float, size: float, is_maker: bool = True) -> float:
        """Calculate the fee for an order."""
        if not self.maker_fee_rate or not self.taker_fee_rate:
            logger.warning("No fee rates available, using default estimate")
            fee_rate = 0.00015 if is_maker else 0.00038
        else:
            fee_rate = self.maker_fee_rate if is_maker else self.taker_fee_rate

        notional_value = price * size
        fee = notional_value * fee_rate

        logger.debug(f"Order fee calculation - notional: {notional_value}, rate: {fee_rate}, fee: {fee}")
        return fee

    def _save_balance_snapshot(self):
        """Save account balance snapshot to file when no position exists (max once every 5 minutes)."""
        try:
            if not self.account_balance:
                logger.debug("No account balance available for snapshot")
                return

            current_time = time.time()

            # Check if 5 minutes have passed since last snapshot
            if self.last_balance_snapshot_time is not None:
                time_since_last_snapshot = current_time - self.last_balance_snapshot_time
                if time_since_last_snapshot < 300:  # 300 seconds = 5 minutes
                    logger.debug(f"Skipping balance snapshot - only {time_since_last_snapshot:.1f}s since last snapshot (need 300s)")
                    return

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            snapshot_line = f"[{timestamp}] Total: {self.account_balance:.4f} USD (No Position - Placing Buy Orders)\n"

            # Append to file
            with open(BALANCE_SNAPSHOT_FILE, 'a', encoding='utf-8') as f:
                f.write(snapshot_line)

            self.last_balance_snapshot_time = current_time
            logger.info(f"💾 Balance snapshot saved: Total: {self.account_balance:.4f} USD")

        except Exception as e:
            logger.warning(f"Failed to save balance snapshot: {e}")

    def _check_sufficient_balance(self, price: float, size: float, is_buy: bool = True) -> bool:
        """Check if account has sufficient balance for the order."""
        if not self.available_balance:
            logger.warning("No balance information available")
            return False

        if is_buy:
            # For buy orders, need USD to cover the initial margin (notional / leverage) + fees
            notional_value = price * size
            fee = self._calculate_order_fee(price, size, is_maker=True)

            # With leverage, the required margin is the notional value divided by the leverage ratio.
            required_margin = notional_value / float(self.leverage)
            required_balance = required_margin + fee

            logger.debug(f"Buy order check - Notional: {notional_value:.2f}, Leverage: {self.leverage}x, Required Margin: {required_margin:.2f}, Fee: {fee:.4f}, Required Balance: {required_balance:.2f}, Available: {self.available_balance:.2f}")

            if self.available_balance >= required_balance:
                logger.debug("✅ Sufficient balance for buy order")
                return True
            else:
                logger.warning(f"❌ Insufficient balance - need {required_balance:.2f}, have {self.available_balance:.2f}")
                return False
        else:
            # For sell orders, mainly need USD for fees (assuming we have the position)
            fee = self._calculate_order_fee(price, size, is_maker=True)

            logger.debug(f"Sell order check - fee: {fee}, available: {self.available_balance}")

            if self.available_balance >= fee:
                logger.debug("✅ Sufficient balance for sell order fees")
                return True
            else:
                logger.warning(f"❌ Insufficient balance for fees - need {fee}, have {self.available_balance}")
                return False

    async def _set_leverage(self):
        """Set the leverage for the contract."""
        logger.debug(f"Setting leverage to {self.leverage} for contract {self.contract_id}")
        try:
            # WORKAROUND: The SDK's update_leverage_setting method is broken.
            # We are re-implementing its logic here using the internal async_client that is known to work.
            logger.info("Applying workaround for broken SDK method 'update_leverage_setting'.")
            
            path = "/api/v1/private/account/updateLeverageSetting"
            data = {
                "accountId": str(self.client.internal_client.get_account_id()),
                "contractId": self.contract_id,
                "leverage": str(self.leverage)
            }

            # Use the internal client to make a properly authenticated POST request.
            response = await self.client.internal_client.make_authenticated_request(
                method="POST",
                path=path,
                data=data
            )

            logger.debug(f"Leverage setting response: {response}")

            if response.get("code") == "SUCCESS":
                logger.info(f"✅ Successfully set leverage to {self.leverage} for contract {self.contract_id}")
            else:
                logger.warning(f"⚠️ Failed to set leverage: {response.get('msg', 'Unknown error')}")

        except Exception as e:
            logger.warning(f"Failed to set leverage: {e}")
            logger.debug(f"Leverage setting exception: {type(e).__name__}: {e}")

    async def _setup_websocket(self):
        """Set up WebSocket connections and subscriptions."""
        logger.debug("Setting up WebSocket connections")
        try:
            # Connect to public WebSocket for market data
            logger.debug("Connecting to public WebSocket")
            self.ws_manager.connect_public()
            logger.debug("Public WebSocket connection initiated")

            # Ticker data will now be fetched via REST API.
            logger.info("Skipping public ticker WebSocket subscription.")

            # The subscribe_depth method is not available in the current SDK version.
            # The bot will rely on the ticker feed for market data.
            # try:
            #     logger.debug(f"Subscribing to depth data for contract {self.contract_id}")
            #     self.ws_manager.subscribe_depth(self.contract_id, self._handle_order_book_update)
            #     logger.info(f"Subscribed to depth data for contract {self.contract_id}")
            # except Exception as e:
            #     logger.warning(f"Depth subscription failed: {e}")
            #     logger.debug(f"Depth subscription exception: {type(e).__name__}: {e}")
            
            # Connect to private WebSocket for account updates (only if we have valid credentials)
            if self.client and self.account_id and self.stark_private_key:
                try:
                    logger.debug("Connecting to private WebSocket...")
                    self.ws_manager.connect_private()
                    logger.debug("Private WebSocket connection initiated.")
                    # NOTE: Real-time subscriptions for orders/positions are currently not functional due to
                    # unknown method names in the SDK. The bot will rely on fast REST polling
                    # via the order monitor task for timely fill detection.
                    logger.warning("⚠️ No functional WebSocket subscription for order/position updates found. Relying on REST polling.")
                except Exception as e:
                    logger.error(f"Failed to connect to private WebSocket: {e}")
            else:
                logger.info("Skipping private WebSocket (no valid credentials)")
                logger.debug(f"Private WS skip reason: client={bool(self.client)}, account_id={self.account_id}, has_key={bool(self.stark_private_key)}")
            
            logger.info("WebSocket connections established")
            
        except Exception as e:
            logger.error(f"Failed to setup WebSocket: {e}")
            raise

    async def _refresh_position_and_balance(self):
        """Asynchronously fetches the latest position and balance."""
        logger.debug("Executing REST API refresh for position and balance.")
        try:
            await self._update_account_state()
            logger.info("✅ Position and balance state successfully refreshed via REST API.")
        except Exception as e:
            logger.error(f"Error during state refresh: {e}")

    async def _update_account_state(self):
        """Fetches the latest position and balance from the REST API concurrently."""
        if not self.client:
            return

        logger.debug("Updating account state (position and balance)...")
        # Fetch position and balance concurrently for efficiency
        position_task = asyncio.create_task(self._get_current_position())
        balance_task = asyncio.create_task(self._get_account_balance())

        # Wait for both to complete
        self.current_position = await position_task
        await balance_task
        
        if self.current_position:
            logger.debug(f"Position updated: size {self.current_position.get('size')}")
        else:
            logger.debug("No active position found.")
        logger.debug("Account state update complete.")

    def _load_spread_parameters(self) -> bool:
        """Load spread parameters from the JSON file."""
        logger.debug(f"Attempting to load spread parameters from {self.spread_params_file}")
        try:
            with open(self.spread_params_file, 'r') as f:
                self.spread_params = json.load(f)
            logger.info(f"✅ Successfully loaded spread parameters from {self.spread_params_file}")
            return True
        except FileNotFoundError:
            logger.error(f"Spread parameters file not found at: {self.spread_params_file}")
            self.spread_params = None
            return False
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing spread parameters file: {e}")
            self.spread_params = None
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading spread parameters: {e}")
            self.spread_params = None
            return False

    async def _update_prices_from_rest_api(self) -> bool:
        """Fetch the latest 24-hour quote via REST API and determine mid-price."""
        logger.debug("Fetching 24-hour quote via REST API...")
        try:
            quote_response = await self.client.quote.get_24_hour_quote(self.contract_id)
            logger.debug(f"24-hour quote response: {quote_response}")

            if quote_response.get("code") != "SUCCESS":
                logger.warning(f"Failed to fetch 24-hour quote: {quote_response.get('msg')}")
                return False

            quote_data_list = quote_response.get("data", [])
            if not isinstance(quote_data_list, list) or not quote_data_list:
                logger.warning("Quote response data is not a list or is empty.")
                return False
            
            quote_data = quote_data_list[0]

            # --- Mid-price Calculation ---
            best_bid_str = quote_data.get("bestBid")
            best_ask_str = quote_data.get("bestAsk")
            market_mid_price = None

            if best_bid_str and best_ask_str:
                market_mid_price = (float(best_bid_str) + float(best_ask_str)) / 2.0
            else:
                last_price_str = quote_data.get("lastPrice")
                if last_price_str:
                    market_mid_price = float(last_price_str)
                else:
                    logger.error("Could not find best bid/ask or lastPrice in quote data.")
                    return False
            
            self.market_mid_price = market_mid_price

            if not self.tick_size:
                logger.error("Cannot create synthetic spread without tick_size.")
                return False

            # --- Spread Calculation ---
            inventory = float(self.current_position.get("size", "0")) if self.current_position else 0.0
            log_method = ""
            
            if self.use_advanced_spread and self.spread_params:
                log_method = "Advanced"
                p = self.spread_params
                
                gamma, tau, sigma = p['gamma'], p['tau'], p['latest_volatility']
                k_bid, k_ask = p['k_bid'], p['k_ask']
                c_AS_bid, c_AS_ask = p['c_AS_bid'], p['c_AS_ask']
                maker_fee_bps = p['maker_fee_bps']
                beta_alpha = p.get('beta_alpha', 0.0) # Use .get for safety

                # NOTE: Microprice deviation is simplified to 0 as we are not fetching the full order book
                micro_price_deviation = 0.0
                reservation_price = market_mid_price + (beta_alpha * micro_price_deviation) - (inventory * gamma * (sigma**2) * tau)
                
                k_mid = (k_bid + k_ask) / 2.0
                k_price_based = k_mid * market_mid_price / 10000.0
                
                phi_inventory = (1.0 / gamma) * math.log(1 + gamma / k_price_based) if gamma > 0 and k_price_based > 0 else 0.01
                phi_adverse_sel = 0.5 * gamma * (sigma**2) * tau
                phi_adverse_cost = (c_AS_bid + c_AS_ask) / 2.0
                phi_fee = maker_fee_bps / 10000.0 * market_mid_price
                half_spread = phi_inventory + phi_adverse_sel + phi_adverse_cost + phi_fee

                delta_bid_as = c_AS_bid - phi_adverse_cost
                delta_ask_as = c_AS_ask - phi_adverse_cost

                self.current_bid = self._round_price_to_tick_size(reservation_price - half_spread - delta_bid_as)
                self.current_ask = self._round_price_to_tick_size(reservation_price + half_spread + delta_ask_as)
            else:
                log_method = "Fallback"
                self.current_bid = self._round_price_to_tick_size(market_mid_price - (FALLBACK_TICK_SPREAD * self.tick_size))
                self.current_ask = self._round_price_to_tick_size(market_mid_price + (FALLBACK_TICK_SPREAD * self.tick_size))

            # Log the synthesized market
            log_msg = f"📊 Quotes (Mid={market_mid_price:.4f}, Inv={inventory:.4f}, Method={log_method}):"
            if self.current_bid and self.current_ask and market_mid_price > 0:
                bid_pct_diff = ((market_mid_price - self.current_bid) / market_mid_price) * 100
                ask_pct_diff = ((self.current_ask - market_mid_price) / market_mid_price) * 100
                log_msg += f" Bid={self.current_bid} (-{bid_pct_diff:.3f}%), Ask={self.current_ask} (+{ask_pct_diff:.3f}%)"
            else:
                log_msg += f" Bid={self.current_bid}, Ask={self.current_ask}"
            logger.info(log_msg)
            
            return True

        except Exception as e:
            logger.error(f"Error updating prices from REST API: {e}")
            logger.debug(f"Price update exception details: {type(e).__name__}: {e}")
            return False


    async def _get_current_position(self) -> Optional[Dict[str, Any]]:
        """Get current position via REST API."""
        logger.debug("Fetching current position via REST API")
        try:
            positions_response = await self.client.get_account_positions()
            logger.debug(f"Full positions API response: {positions_response}")

            # Based on API response, 'positionList' contains the size info.
            position_data = positions_response.get("data", {})
            positions = position_data.get("positionList", [])

            if not positions:
                logger.debug("No active positions found in 'positionList'.")
                return None

            logger.debug(f"Found {len(positions)} items in 'positionList'.")

            for position in positions:
                contract_id = position.get("contractId")
                # The size field is named 'openSize' in 'positionList'
                size = position.get("openSize", "0")

                if contract_id == self.contract_id:
                    logger.debug(f"Found position for target contract {contract_id} with openSize={size}")
                    size_float = float(size)
                    if size_float != 0:
                        # The bot's logic may expect a 'size' key, so add it for consistency.
                        position['size'] = size
                        logger.debug(f"Returning active position: {position}")
                        return position
            
            logger.debug("No active position found for the target contract in the list.")
            return None

        except Exception as e:
            logger.error(f"Failed to get current position: {e}")
            logger.debug(f"Position fetch exception: {type(e).__name__}: {e}")
            return None


    async def _cancel_active_orders(self):
        """Cancel all active orders for the contract using the bulk cancel endpoint."""
        if not self.client or not self.contract_id:
            logger.warning("Cannot cancel orders without client or contract_id.")
            return

        logger.info(f"Cancelling all active orders for contract {self.contract_id}...")
        try:
            # Use the SDK's ability to cancel all orders for a contract
            cancel_params = CancelOrderParams(contract_id=self.contract_id)
            cancel_response = await self.client.cancel_order(cancel_params)
            
            logger.debug(f"Cancel all orders response: {cancel_response}")

            if cancel_response.get("code") == "SUCCESS":
                # The response for cancelAll may contain details about what was cancelled
                success_data = cancel_response.get("data", {})
                if success_data:
                     logger.info(f"✅ Successfully sent request to cancel all orders.")
                else:
                    logger.info("✅ Cancel all request successful. No active orders to cancel.")
            else:
                error_msg = cancel_response.get("msg", "Unknown error")
                logger.warning(f"⚠️ Failed to cancel all orders: {error_msg}")

        except Exception as e:
            logger.error(f"An exception occurred while cancelling all orders: {e}")
            logger.debug(f"Cancel all orders exception: {type(e).__name__}: {e}")

    async def _execute_trading_decision(self):
        """Manage buy and sell orders to maintain a spread."""
        logger.debug("Executing market maker trading decision")

        # Always cancel existing orders before placing new ones to ensure the state is clean.
        await self._cancel_active_orders()
        await asyncio.sleep(POST_CANCELLATION_DELAY)  # Allow time for balance updates post-cancellation.
        
        # Concurrently manage buy and sell orders
        await asyncio.gather(
            self._place_buy_order(),
            self._place_sell_order()
        )

        # Update last order time
        self.last_order_time = time.time()

    async def _place_buy_order(self):
        """Places a limit buy order based on the calculated bid price."""
        logger.debug("Attempting to place buy order")
        if not self.current_bid:
            logger.warning("No bid price available, cannot place buy order")
            return

        if not self.client:
            logger.warning("No authenticated client available for placing orders")
            return

        try:
            rounded_price = self._round_price_to_tick_size(self.current_bid)

            await self._update_account_state()

            if CAPITAL_USAGE_RATIO and self.available_balance and 0 < CAPITAL_USAGE_RATIO <= 1:
                trade_amount = self.available_balance * float(CAPITAL_USAGE_RATIO)
            else:
                trade_amount = float(TRADE_AMOUNT)
            
            if ORDER_SIZE_IN_QUOTE:
                if not self.current_bid or self.current_bid <= 0:
                    logger.error("Cannot calculate order size: Invalid current bid price.")
                    return
                notional_trade_value = trade_amount * float(self.leverage)
                config_order_size = notional_trade_value / self.current_bid
            else:
                config_order_size = trade_amount

            min_size = self.min_order_size or 0.0
            raw_order_size = max(config_order_size, min_size)
            order_size = self._round_size_to_step_size(raw_order_size)

            if self.available_balance is not None and not self._check_sufficient_balance(rounded_price, order_size, is_buy=True):
                logger.error("❌ Insufficient balance for buy order - skipping")
                return

            order_params = CreateOrderParams(
                contract_id=self.contract_id,
                size=str(order_size),
                price=f"{rounded_price:.{self.price_decimal_places}f}",
                type=OrderType.LIMIT,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.POST_ONLY
            )
            
            metadata = await self.client.get_metadata()
            order_response = await self.client.order.create_order(order_params, metadata.get("data", {}))

            if order_response.get("code") == "SUCCESS":
                order_id = order_response.get("data", {}).get("orderId")
                self.active_orders["BUY"] = order_id
                logger.info(f"✅ Placed BUY order at {rounded_price} (size: {order_size}) - ID: {order_id}")
            else:
                logger.error(f"❌ Failed to place buy order: {order_response.get('msg', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Error placing buy order: {e}")

    async def _place_sell_order(self):
        """Places a limit sell order based on the calculated ask price."""
        logger.debug("Attempting to place sell order")
        if not self.current_ask:
            logger.warning("No ask price available, cannot place sell order")
            return

        if not self.client:
            logger.warning("No authenticated client available for placing orders")
            return

        try:
            rounded_price = self._round_price_to_tick_size(self.current_ask)

            await self._update_account_state()

            if CAPITAL_USAGE_RATIO and self.available_balance and 0 < CAPITAL_USAGE_RATIO <= 1:
                trade_amount = self.available_balance * float(CAPITAL_USAGE_RATIO)
            else:
                trade_amount = float(TRADE_AMOUNT)

            if ORDER_SIZE_IN_QUOTE:
                if not self.current_ask or self.current_ask <= 0:
                    logger.error("Cannot calculate order size: Invalid current ask price.")
                    return
                notional_trade_value = trade_amount * float(self.leverage)
                config_order_size = notional_trade_value / self.current_ask
            else:
                config_order_size = trade_amount
            
            min_size = self.min_order_size or 0.0
            raw_order_size = max(config_order_size, min_size)
            order_size = self._round_size_to_step_size(raw_order_size)

            if self.available_balance is not None and not self._check_sufficient_balance(rounded_price, order_size, is_buy=False):
                logger.error("❌ Insufficient balance for sell order fees - skipping")
                return

            order_params = CreateOrderParams(
                contract_id=self.contract_id,
                size=str(order_size),
                price=f"{rounded_price:.{self.price_decimal_places}f}",
                type=OrderType.LIMIT,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.POST_ONLY
            )

            metadata = await self.client.get_metadata()
            order_response = await self.client.order.create_order(order_params, metadata.get("data", {}))

            if order_response.get("code") == "SUCCESS":
                order_id = order_response.get("data", {}).get("orderId")
                self.active_orders["SELL"] = order_id
                logger.info(f"✅ Placed SELL order at {rounded_price} (size: {order_size}) - ID: {order_id}")
            else:
                logger.error(f"❌ Failed to place sell order: {order_response.get('msg', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Error placing sell order: {e}")

    async def _trading_loop(self):
        """Main trading loop."""
        logger.info("Starting trading loop...")
        logger.debug(f"Trading loop config - refresh_interval: {self.refresh_interval}s, contract_id: {self.contract_id}")

        # --- Wait for startup delay ---
        if self.startup_delay > 0:
            logger.info(f"⏳ Applying startup delay of {self.startup_delay}s...")
            await asyncio.sleep(self.startup_delay)

        loop_count = 0
        while self.running:
            loop_count += 1
            logger.debug(f"=== Trading loop iteration {loop_count} === ")

            try:
                # Update account state (position and balance) first, as it's needed for spread calculation
                await self._update_account_state()

                # --- Update market prices and calculate spreads ---
                prices_updated = await self._update_prices_from_rest_api()
                if not prices_updated:
                    logger.warning("Could not update prices, skipping this iteration.")
                    await asyncio.sleep(self.refresh_interval)
                    continue

                # Log current market data
                if self.current_bid and self.current_ask:
                    spread = self.current_ask - self.current_bid
                    spread_pct = (spread / self.current_bid) * 100 if self.current_bid > 0 else 0
                    logger.info(f"📊 Market: Bid={self.current_bid}, Ask={self.current_ask}, Spread={spread:.6f} ({spread_pct:.3f}%)")
                else:
                    logger.warning("⚠️ No market data available yet")
                    logger.debug(f"Market data state - bid: {self.current_bid}, ask: {self.current_ask}")

                # Decide action based on position (only if we have valid credentials and trading is allowed)
                if self.client and self.trading_allowed:
                    # Check order cooldown
                    if self.last_order_time is not None:
                        time_since_last_order = time.time() - self.last_order_time
                        if time_since_last_order < self.min_order_interval:
                            remaining = self.min_order_interval - time_since_last_order
                            logger.info(f"⏳ Order cooldown: waiting {remaining:.1f}s before next order...")
                            await asyncio.sleep(remaining)

                    # If all checks pass, execute trade
                    await self._execute_trading_decision()
                elif not self.client:
                    logger.info("🔄 Running in market data mode only (no trading due to auth issues)")
                else:
                    logger.info("🛑 Trading is currently disabled.")

                # Wait for the main refresh interval
                logger.debug(f"Sleeping for {self.refresh_interval} seconds before next iteration")
                await asyncio.sleep(self.refresh_interval)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                logger.debug(f"Trading loop exception: {type(e).__name__}: {e}")
                await asyncio.sleep(5)  # Short delay before retry

    async def start(self):
        """Start the trading bot."""
        logger.debug("Starting EdgeX trading bot")
        try:
            logger.debug("Initializing bot components")
            await self.initialize()
            self.running = True
            logger.debug("Bot initialization complete, starting trading loop")

            # Start trading loop
            await self._trading_loop()

        except KeyboardInterrupt:
            logger.info("Received shutdown signal...")
            logger.debug("Keyboard interrupt detected")
        except Exception as e:
            logger.error(f"Trading bot error: {e}")
            logger.debug(f"Bot startup exception: {type(e).__name__}: {e}")
        finally:
            logger.debug("Entering cleanup phase")
            await self.stop()

    async def stop(self):
        """Stop the trading bot. This method is idempotent."""
        if getattr(self, 'stopping', False):
            return
        self.stopping = True

        logger.info("Stopping trading bot...")
        logger.debug("Setting running flag to False")
        self.running = False

        try:
            # Cancel any active orders
            logger.debug("Cancelling any remaining active orders")
            await self._cancel_active_orders()

            # Disconnect WebSocket
            if self.ws_manager:
                logger.debug("Disconnecting WebSocket connections")
                self.ws_manager.disconnect_all()
            else:
                logger.debug("No WebSocket manager to disconnect")

            # Close client
            if self.client:
                logger.debug("Closing REST client")
                await self.client.close()
            else:
                logger.debug("No REST client to close")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            logger.debug(f"Shutdown exception: {type(e).__name__}: {e}")

        logger.info("Trading bot stopped")
        logger.debug("Bot shutdown complete")


async def main():
    """
    Main function to run the trading bot.
    Includes signal handling for graceful shutdown.
    """
    logger.debug("=== MAIN FUNCTION START ===")

    # Configuration from environment variables
    base_url = os.getenv("EDGEX_BASE_URL", "https://pro.edgex.exchange")
    ws_url = os.getenv("EDGEX_WS_URL", "wss://quote.edgex.exchange")
    account_id = int(os.getenv("EDGEX_ACCOUNT_ID", "0"))
    stark_private_key = os.getenv("EDGEX_STARK_PRIVATE_KEY", "")
    leverage = LEVERAGE
    use_advanced_spread = os.getenv("EDGEX_USE_ADVANCED_SPREAD", str(USE_ADVANCED_SPREAD)).lower() in ("true", "1", "yes", "on")
    spread_params_file = os.getenv("EDGEX_SPREAD_PARAMS_FILE", SPREAD_PARAMS_FILE)

    logger.debug(f"Configuration loaded:")
    logger.debug(f"  base_url: {base_url}")
    logger.debug(f"  ws_url: {ws_url}")
    logger.debug(f"  account_id: {account_id}")
    logger.debug(f"  has_private_key: {bool(stark_private_key)}")
    logger.debug(f"  leverage: {leverage}")
    logger.debug(f"  debug_logging: {debug_logging}")
    logger.debug(f"  log_file: {log_file_path}")
    logger.debug(f"  use_advanced_spread: {use_advanced_spread}")
    logger.debug(f"  spread_params_file: {spread_params_file}")

    if not account_id or not stark_private_key:
        logger.error("Please set EDGEX_ACCOUNT_ID and EDGEX_STARK_PRIVATE_KEY environment variables")
        logger.debug("Missing required credentials - exiting")
        return

    # Create and start trading bot
    logger.debug("Creating EdgeXTradingBot instance")
    bot = EdgeXTradingBot(
        base_url=base_url,
        ws_url=ws_url,
        account_id=account_id,
        stark_private_key=stark_private_key,
        leverage=leverage,
        use_advanced_spread=use_advanced_spread,
        spread_params_file=spread_params_file
    )
    logger.debug("Bot instance created")

    main_task = asyncio.create_task(bot.start())

    if platform.system() != "Windows":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, main_task.cancel)

    try:
        await main_task
    except asyncio.CancelledError:
        logger.info("Main task cancelled. Shutdown complete.")

    logger.debug("=== MAIN FUNCTION END ===")


if __name__ == "__main__":
    asyncio.run(main())
