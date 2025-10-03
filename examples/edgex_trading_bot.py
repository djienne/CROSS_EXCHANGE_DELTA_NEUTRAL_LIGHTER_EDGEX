import asyncio
import json
import logging
import os
import platform
import signal
import sys
import time
from decimal import Decimal
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
LEVERAGE = 2.0
# Time in seconds between each trading loop iteration
REFRESH_INTERVAL = 15
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
CAPITAL_USAGE_RATIO = 0.95
# Set to True to interpret TRADE_AMOUNT as quote currency (USD), False for base currency (e.g., PAXG)
ORDER_SIZE_IN_QUOTE = True
# Time to wait after cancelling an order before placing a new one, to allow balance to update
POST_CANCELLATION_DELAY = 0.25
# Maximum total trading volume in USD before the bot stops trading.
# This limit is only active if USE_MAX_VOLUME_LIMIT is set to True.
MAX_TOTAL_VOLUME_USD = 80500.0
# Set to True to enable the maximum total volume limit. Off by default.
USE_MAX_VOLUME_LIMIT = False
# Balance snapshot file path
BALANCE_SNAPSHOT_FILE = "balance_snapshots.txt"

# --- SPREAD CONFIGURATION ---
# Set to True to use Avellaneda-based deltas for spread calculation.
USE_AVELLANEDA_SPREAD = True
# Path to the Avellaneda parameters JSON file.
AVELLANEDA_PARAMS_FILE = "edgeX_data_collector/avellaneda_parameters_PAXG.json"
# Number of ticks to place orders away from the last price when creating a synthetic spread (used if USE_AVELLANEDA_SPREAD is False).
TICK_SPREAD_FACTOR = 10
# --- END TRADING CONFIGURATION ---



class EdgeXTradingBot:
    def __init__(self, base_url: str, ws_url: str, account_id: int, stark_private_key: str, leverage: float = LEVERAGE, use_avellaneda_spread: bool = USE_AVELLANEDA_SPREAD, avellaneda_params_file: str = AVELLANEDA_PARAMS_FILE):
        """
        Initialize the EdgeX Trading Bot.
        
        Args:
            base_url: EdgeX API base URL
            ws_url: EdgeX WebSocket URL
            account_id: Your account ID
            stark_private_key: Your Stark private key
            leverage: Trading leverage (default: from config)
            use_avellaneda_spread: Whether to use Avellaneda deltas for spread
            avellaneda_params_file: Path to Avellaneda parameters file
        """
        self.base_url = base_url
        self.ws_url = ws_url
        self.account_id = account_id
        self.stark_private_key = stark_private_key
        self.leverage = leverage
        self.use_avellaneda_spread = use_avellaneda_spread
        self.avellaneda_params_file = avellaneda_params_file
        
        # Trading state
        self.contract_id = None  # Will be set for PAXGUSD
        self.current_position = None
        self.active_order_id = None
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
        self.order_monitor_task: Optional[asyncio.Task] = None
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
                elif self.available_balance and self.available_balance < Decimal("15"):
                    if not self.current_position:
                        logger.error(f"❌ INSUFFICIENT CAPITAL: Available balance ${self.available_balance:.2f} USD is below minimum required $15 USD and no open position to close.")
                        logger.error("🛑 Trading bot stopped due to insufficient capital.")
                        raise Exception(f"Insufficient capital for trading: ${self.available_balance:.2f} USD < $15 USD minimum and no open position.")
                    else:
                        logger.warning(f"⚠️ Low available balance: ${self.available_balance:.2f} USD, but allowing bot to run to close open position.")
                elif self.available_balance and self.available_balance < Decimal("50"):
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

            # Display initial total traded volume
            if self.client and USE_MAX_VOLUME_LIMIT:
                logger.info("🔍 Calculating initial total traded volume...")
                _, should_continue = await self._calculate_total_traded_volume()
                if not should_continue:
                    self.trading_allowed = False
                    logger.error("❌ Maximum trading volume already exceeded at startup - trading disabled")

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
                    self.tick_size = Decimal(str(contract.get("tickSize", "0.01")))
                    self.step_size = Decimal(str(contract.get("stepSize", "0.001")))
                    self.min_order_size = Decimal(str(contract.get("minOrderSize", "0.01")))
                    self.taker_fee_rate = Decimal(str(contract.get("defaultTakerFeeRate", "0.00038")))
                    self.maker_fee_rate = Decimal(str(contract.get("defaultMakerFeeRate", "0.00015")))

                    logger.info(f"Found PAXGUSD contract ID: {self.contract_id}")
                    logger.debug(f"Contract specs - tick_size: {self.tick_size}, step_size: {self.step_size}, min_order_size: {self.min_order_size}")
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

    def _round_price_to_tick_size(self, price: Decimal) -> Decimal:
        """Round price to the contract's tick size."""
        if not self.tick_size:
            logger.warning("No tick size available, using price as-is")
            return price

        # Round to the nearest tick
        rounded_price = (price / self.tick_size).quantize(Decimal('1'), rounding='ROUND_HALF_UP') * self.tick_size
        logger.debug(f"Rounded price {price} to {rounded_price} (tick_size: {self.tick_size})")
        return rounded_price

    def _round_size_to_step_size(self, size: Decimal) -> Decimal:
        """Round order size to the contract's step size."""
        if not self.step_size:
            logger.warning("No step size available, using size as-is")
            return size

        # Round to the nearest step
        rounded_size = (size / self.step_size).quantize(Decimal('1'), rounding='ROUND_HALF_UP') * self.step_size
        logger.debug(f"Rounded size {size} to {rounded_size} (step_size: {self.step_size})")
        return rounded_size

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

            # --- Total Balance Calculation ---
            # The 'amount' in collateralList appears to be EQUITY (Collateral + PnL), which can be negative.
            # A more stable value is the actual wallet balance, which might be in the main 'account' object.
            # We'll try to find a field like 'totalWalletBalance' and fall back to the old method if not found.
            
            total_balance_str = None
            if self.account_data and 'totalWalletBalance' in self.account_data:
                total_balance_str = self.account_data.get('totalWalletBalance')
                logger.debug(f"Using 'totalWalletBalance' from account object for total balance: {total_balance_str}")
            else:
                logger.debug("Field 'totalWalletBalance' not found in account object. Falling back to collateralList.")
                collateral_list = balance_data.get("collateralList", [])
                for balance in collateral_list:
                    if balance.get("coinId") == "1000":  # USD
                        total_balance_str = balance.get("amount", "0")
                        logger.warning(f"Using 'amount' from collateralList for total balance: {total_balance_str}. This may be equity, not wallet balance.")
                        break
            
            if total_balance_str is None:
                logger.error("Could not determine total account balance.")
                return False
            
            self.account_balance = Decimal(str(total_balance_str))

            # --- Available Balance Calculation ---
            available_balance_str = None
            collateral_asset_list = balance_data.get("collateralAssetModelList", [])
            for asset in collateral_asset_list:
                if asset.get("coinId") == "1000": # USD
                    available_balance_str = asset.get("availableAmount", "0")
                    break
            
            if available_balance_str is None:
                logger.warning("Could not find available balance in collateralAssetModelList. This may cause issues.")
                self.available_balance = None
            else:
                self.available_balance = Decimal(str(available_balance_str))

            logger.info(f"💰 Account Balance - Total: {self.account_balance:.4f} USD, Available: {self.available_balance:.4f} USD")
            return True

        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            logger.debug(f"Balance check exception: {type(e).__name__}: {e}")
            return False

    def _calculate_order_fee(self, price: Decimal, size: Decimal, is_maker: bool = True) -> Decimal:
        """Calculate the fee for an order."""
        if not self.maker_fee_rate or not self.taker_fee_rate:
            logger.warning("No fee rates available, using default estimate")
            fee_rate = Decimal("0.00015") if is_maker else Decimal("0.00038")
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

    def _check_sufficient_balance(self, price: Decimal, size: Decimal, is_buy: bool = True) -> bool:
        """Check if account has sufficient balance for the order."""
        if not self.available_balance:
            logger.warning("No balance information available")
            return False

        if is_buy:
            # For buy orders, need USD to cover the initial margin (notional / leverage) + fees
            notional_value = price * size
            fee = self._calculate_order_fee(price, size, is_maker=True)

            # With leverage, the required margin is the notional value divided by the leverage ratio.
            required_margin = notional_value / Decimal(str(self.leverage))
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

    def _load_avellaneda_deltas(self) -> Optional[tuple[Decimal, Decimal]]:
        """Load delta_a and delta_b from the Avellaneda parameters JSON file."""
        logger.debug(f"Attempting to load Avellaneda deltas from {self.avellaneda_params_file}")
        try:
            with open(self.avellaneda_params_file, 'r') as f:
                params = json.load(f)
            
            delta_a = Decimal(str(params["limit_orders"]["delta_a"]))
            delta_b = Decimal(str(params["limit_orders"]["delta_b"]))

            logger.debug(f"Loaded Avellaneda deltas: delta_a={delta_a}, delta_b={delta_b}")
            return delta_a, delta_b
        except FileNotFoundError:
            logger.error(f"Avellaneda parameters file not found at: {self.avellaneda_params_file}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing Avellaneda parameters file: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading Avellaneda deltas: {e}")
            return None

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
            # Try to get best bid/ask from quote data to calculate mid-price
            best_bid_str = quote_data.get("bestBid")
            best_ask_str = quote_data.get("bestAsk")
            market_mid_price = None

            if best_bid_str and best_ask_str:
                best_bid = Decimal(best_bid_str)
                best_ask = Decimal(best_ask_str)
                market_mid_price = (best_bid + best_ask) / 2
                logger.info(f"Calculated market mid-price: {market_mid_price:.4f} from quote's Best Bid/Ask.")
            else:
                # Fallback to lastPrice if best bid/ask are not available
                last_price_str = quote_data.get("lastPrice")
                if last_price_str:
                    market_mid_price = Decimal(last_price_str)
                    logger.info(f"Using lastPrice as mid-price: {market_mid_price:.4f} (best bid/ask not in quote).")
                else:
                    logger.error("Could not find best bid/ask or lastPrice in quote data.")
                    self.market_mid_price = None
                    return False
            
            self.market_mid_price = market_mid_price

            if not self.tick_size:
                logger.error("Cannot create synthetic spread without tick_size.")
                return False

            # --- Spread Calculation using market mid-price ---
            log_method = ""
            use_tick_spread = True
            if self.use_avellaneda_spread:
                deltas = self._load_avellaneda_deltas()
                if deltas:
                    delta_a, delta_b = deltas
                    self.current_bid = self._round_price_to_tick_size(market_mid_price - delta_b)
                    self.current_ask = self._round_price_to_tick_size(market_mid_price + delta_a)
                    log_method = "Avellaneda"
                    use_tick_spread = False
                else:
                    logger.warning("Failed to load Avellaneda deltas. Falling back to TICK_SPREAD_FACTOR.")

            if use_tick_spread:
                self.current_bid = self._round_price_to_tick_size(market_mid_price - (TICK_SPREAD_FACTOR * self.tick_size))
                self.current_ask = self._round_price_to_tick_size(market_mid_price + (TICK_SPREAD_FACTOR * self.tick_size))
                log_method = "Ticks"

            # Log the synthesized market with percentage differences
            if market_mid_price > 0:
                bid_pct_diff = ((market_mid_price - self.current_bid) / market_mid_price) * 100
                ask_pct_diff = ((self.current_ask - market_mid_price) / market_mid_price) * 100
                logger.info(f"📊 Synthesized Market from {log_method} (midPrice={market_mid_price:.4f}): Bid={self.current_bid} (-{bid_pct_diff:.3f}%), Ask={self.current_ask} (+{ask_pct_diff:.3f}%)")
            else:
                logger.info(f"📊 Synthesized Market from {log_method} (midPrice={market_mid_price:.4f}): Bid={self.current_bid}, Ask={self.current_ask}")
            
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
                    size_decimal = Decimal(str(size))
                    if size_decimal != 0:
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


    async def _calculate_total_traded_volume(self) -> tuple[Decimal, bool]:
        """
        Calculate total traded volume across all contracts for this account.

        Returns:
            tuple: (total_volume, should_continue_trading)
        """
        if not self.client:
            logger.warning("No client available for volume calculation")
            return Decimal("0"), True

        try:
            logger.debug("Fetching order fill transactions for total volume calculation...")
            total_volume = Decimal("0")
            offset_data = ""

            # Get all order fill transactions with pagination.
            # The API appears to provide complete historical data, so we retrieve all pages.
            while True:
                params = OrderFillTransactionParams(
                    size="100",
                    offset_data=offset_data
                )

                response = await self.client.get_order_fill_transactions(params)
                logger.debug(f"Order fill transactions response: {response}")

                if response.get("code") != "SUCCESS":
                    logger.warning(f"Failed to get order fill transactions: {response.get('msg')}")
                    break

                data = response.get("data", {})
                transactions = data.get("dataList", [])

                if not transactions:
                    logger.debug("No more transactions found")
                    break

                # Sum up all fillValue amounts for target contract
                page_volume = Decimal("0")
                target_contract_count = 0
                for transaction in transactions:
                    contract_id = transaction.get("contractId")
                    if contract_id == self.contract_id:
                        fill_value = Decimal(str(transaction.get("fillValue", "0")))
                        page_volume += fill_value
                        target_contract_count += 1

                total_volume += page_volume
                logger.debug(f"Page: {len(transactions)} total transactions, {target_contract_count} for target contract, ${page_volume:,.2f} page volume")

                # Check if there are more pages
                if not data.get("hasNext", False):
                    break

                offset_data = data.get("offsetData", "")
                if not offset_data:
                    break

            max_volume = Decimal(str(MAX_TOTAL_VOLUME_USD))
            should_continue = total_volume < max_volume

            if should_continue:
                remaining = max_volume - total_volume
                percentage_used = (total_volume / max_volume) * 100
                logger.info(f"💹 Total Traded Volume: ${total_volume:,.2f} USD ({percentage_used:.1f}% of ${max_volume:,.2f} limit, ${remaining:,.2f} remaining)")
            else:
                logger.warning(f"🛑 MAXIMUM VOLUME REACHED: ${total_volume:,.2f} USD >= ${max_volume:,.2f} USD - STOPPING TRADING")

            return total_volume, should_continue

        except Exception as e:
            logger.error(f"Error calculating total traded volume: {e}")
            logger.debug(f"Volume calculation exception: {type(e).__name__}: {e}")
            return Decimal("0"), True

    async def _cancel_active_orders(self):
        """Cancel all active orders for the contract."""
        logger.debug("Cancelling active orders")
        try:
            params = GetActiveOrderParams(size="100", offset_data="")
            logger.debug(f"Getting active orders with params: {params}")
            active_orders_response = await self.client.get_active_orders(params)
            logger.debug(f"Active orders response: {active_orders_response}")

            if active_orders_response.get("code") != "SUCCESS":
                error_msg = active_orders_response.get("msg", "Unknown error")
                logger.warning(f"Failed to get active orders: {error_msg}")
                return

            active_orders = active_orders_response.get("data", {}).get("dataList", [])
            logger.debug(f"Found {len(active_orders)} active orders")

            # Filter orders for our specific contract
            our_orders = []
            for order in active_orders:
                order_contract_id = order.get("contractId")
                # Try both 'id' and 'orderId' fields as the API response uses 'id'
                order_id = order.get("id") or order.get("orderId")
                logger.debug(f"Checking order: Contract {order_contract_id}, ID {order_id}")

                if order_contract_id == self.contract_id:
                    our_orders.append(order)
                    logger.debug(f"Found order {order_id} for our contract {self.contract_id}")

            if not our_orders:
                logger.debug(f"No active orders found for contract {self.contract_id}")
                return

            # Cancel each order individually
            cancelled_count = 0
            for order in our_orders:
                # Try both 'id' and 'orderId' fields as the API response uses 'id'
                order_id = order.get("id") or order.get("orderId")
                if not order_id:
                    logger.warning(f"Order missing id/orderId: {order}")
                    continue

                logger.debug(f"Cancelling order {order_id} for contract {self.contract_id}")
                cancel_params = CancelOrderParams(order_id=str(order_id))
                cancel_response = await self.client.cancel_order(cancel_params)
                logger.debug(f"Cancel response for order {order_id}: {cancel_response}")

                if cancel_response.get("code") == "SUCCESS":
                    logger.info(f"✅ Cancelled order: {order_id}")
                    cancelled_count += 1
                else:
                    error_msg = cancel_response.get("msg", "Unknown error")
                    logger.warning(f"Failed to cancel order {order_id}: {error_msg}")

            logger.info(f"Successfully cancelled {cancelled_count}/{len(our_orders)} orders")

        except Exception as e:
            logger.error(f"Failed to cancel active orders: {e}")
            logger.debug(f"Cancel orders exception: {type(e).__name__}: {e}")
            # Don't raise the exception, just log it to prevent initialization failure

    async def _execute_trading_decision(self):
        """Execute the appropriate trading decision based on current position."""
        if self.current_position is None:
            # No position - save balance snapshot and place buy order
            self._save_balance_snapshot()
            logger.info("💰 No position found, attempting to place buy order...")
            logger.debug("Trading decision: BUY (no position)")
            await self._place_buy_order()
        else:
            # Have position - place sell order
            position_size = self.current_position.get("size", "0")
            logger.info(f"📈 Position found (size: {position_size}), attempting to place sell order to close...")
            logger.debug(f"Trading decision: SELL (close position of size {position_size})")
            await self._place_sell_order()

        # Update last order time
        self.last_order_time = time.time()

    async def _start_order_monitor(self, order_id: str):
        """Starts a new background task to monitor an active order."""
        # Cancel any existing monitor task first
        if self.order_monitor_task and not self.order_monitor_task.done():
            logger.debug("Cancelling previous order monitor task.")
            self.order_monitor_task.cancel()

        self.order_monitor_task = asyncio.create_task(self._monitor_active_order(order_id))

    async def _monitor_active_order(self, order_id: str):
        """
        Polls the status of a specific active order until it's no longer active (filled or cancelled).
        """
        logger.info(f"👁️ Starting to monitor order {order_id} for fills...")
        
        # Give a brief moment for the order to appear in the system
        await asyncio.sleep(1)

        while self.running and self.active_order_id == order_id:
            try:
                params = GetActiveOrderParams(size="100")
                active_orders_response = await self.client.get_active_orders(params)

                if active_orders_response.get("code") != "SUCCESS":
                    logger.warning(f"Failed to get active orders while monitoring {order_id}. Retrying...")
                    await asyncio.sleep(5) # Wait longer on API error
                    continue

                active_orders = active_orders_response.get("data", {}).get("dataList", [])
                
                order_is_still_active = any(order.get("id") == order_id for order in active_orders)
                
                if order_is_still_active:
                    logger.debug(f"Order {order_id} is still active. Continuing to monitor.")
                    await asyncio.sleep(3)  # Poll every 3 seconds
                else:
                    # Order is no longer in the active list, so it was likely filled or cancelled.
                    logger.info(f"✅ Order {order_id} is no longer active. Triggering state refresh.")
                    await self._refresh_position_and_balance()
                    # Stop this monitoring task
                    break 

            except Exception as e:
                logger.error(f"An error occurred while monitoring order {order_id}: {e}")
                await asyncio.sleep(5) # Wait longer on exception
        
        logger.info(f"👁️ Stopped monitoring order {order_id}.")

    async def _place_buy_order(self):
        """Places a limit buy order based on the calculated bid price."""
        logger.debug("Attempting to place buy order")
        if not self.current_bid:
            logger.warning("No bid price available, cannot place buy order")
            logger.debug(f"Current market data - bid: {self.current_bid}, ask: {self.current_ask}")
            return

        if not self.client:
            logger.warning("No authenticated client available for placing orders")
            return

        try:
            # Round price to tick size for comparison
            rounded_price = self._round_price_to_tick_size(self.current_bid)

            # Check if we need to cancel existing orders first
            params = GetActiveOrderParams(size="100", offset_data="")
            active_orders_response = await self.client.get_active_orders(params)

            if active_orders_response.get("code") == "SUCCESS":
                active_orders = active_orders_response.get("data", {}).get("dataList", [])
                our_orders = [order for order in active_orders if order.get("contractId") == self.contract_id]

                if our_orders:
                    # Check if a buy order already exists at the target price
                    for order in our_orders:
                        order_price = Decimal(str(order.get("price", "0")))
                        order_side = order.get("side")  # Assuming 'BUY' or 'SELL'
                        if order_side == "BUY" and order_price == rounded_price:
                            logger.info(f"✅ Buy order at {rounded_price} already exists. No action needed.")
                            return

                    logger.info(f"🔄 Price has moved or order is wrong type. Cancelling {len(our_orders)} active orders to replace.")
                    await self._cancel_active_orders()
                    logger.info(f"⏳ Waiting for {POST_CANCELLATION_DELAY}s for balance to update after cancellation...")
                    await asyncio.sleep(POST_CANCELLATION_DELAY)
                else:
                    logger.debug("✅ No active orders to cancel, proceeding with order placement")
            
            # Get current state AFTER any cancellations
            await self._update_account_state()

            # --- Calculate Order Size ---
            if CAPITAL_USAGE_RATIO and self.available_balance and 0 < CAPITAL_USAGE_RATIO <= 1:
                # Dynamically calculate trade amount based on available capital.
                # This 'trade_amount' is the margin we are willing to commit.
                trade_amount = self.available_balance * Decimal(str(CAPITAL_USAGE_RATIO))
                logger.info(f"Using {CAPITAL_USAGE_RATIO*100:.0f}% of available capital (${self.available_balance:.2f}) for trade margin: ${trade_amount:.2f}")
            else:
                # Use the fixed trade amount from configuration
                trade_amount = Decimal(str(TRADE_AMOUNT))
            
            if ORDER_SIZE_IN_QUOTE:
                # Convert the margin (trade_amount) to a notional value using leverage, then to base currency size.
                if not self.current_bid or self.current_bid <= 0:
                    logger.error("Cannot calculate order size in base currency: Invalid current bid price.")
                    return
                
                notional_trade_value = trade_amount * Decimal(str(self.leverage))
                config_order_size = notional_trade_value / self.current_bid
                logger.debug(f"Calculated order size: Notional Value ({notional_trade_value} USD) / Price ({self.current_bid}) = {config_order_size:.6f} in base currency.")
            else:
                # Trade amount is already in base currency
                config_order_size = trade_amount

            min_size = Decimal(str(self.min_order_size)) if self.min_order_size else Decimal("0")
            
            # Ensure order size is at least the minimum required by the contract
            raw_order_size = max(config_order_size, min_size)
            if raw_order_size == min_size and config_order_size < min_size:
                logger.warning(f"Calculated order size {config_order_size:.6f} is below minimum {min_size}. Using minimum size.")

            order_size = self._round_size_to_step_size(raw_order_size)
            logger.debug(f"Using order size: {order_size} (rounded from {raw_order_size})")

            logger.debug(f"Using rounded price: {rounded_price} (original: {self.current_bid})")

            # Check if we have sufficient balance (if balance info is available)
            if self.available_balance is not None:
                if not self._check_sufficient_balance(rounded_price, order_size, is_buy=True):
                    logger.error("❌ Insufficient balance for buy order - skipping")
                    return
            else:
                logger.debug("⚠️ No balance info available - placing order anyway")

            # Create buy order at rounded bid price
            order_params = CreateOrderParams(
                contract_id=self.contract_id,
                size=str(order_size),
                price=str(rounded_price),
                type=OrderType.LIMIT,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.POST_ONLY
            )
            logger.debug(f"Created order params: {order_params}")

            # Get metadata and place order
            try:
                logger.debug("Getting metadata for order placement")
                metadata = await self.client.get_metadata()
                logger.debug(f"Metadata for order: {metadata}")

                if metadata.get("code") != "SUCCESS":
                    logger.error(f"Cannot get metadata for order: {metadata.get('msg')}")
                    return

                logger.debug("Placing buy order")
                order_response = await self.client.order.create_order(order_params, metadata.get("data", {}))
                logger.debug(f"Order response: {order_response}")

                if order_response.get("code") == "SUCCESS":
                    self.active_order_id = order_response.get("data", {}).get("orderId")
                    self.last_order_price = rounded_price
                    log_msg = f"✅ Placed BUY order at {rounded_price} (size: {order_size})"
                    if self.market_mid_price and self.market_mid_price > 0:
                        pct_diff = ((self.market_mid_price - rounded_price) / self.market_mid_price) * 100
                        log_msg += f" (-{pct_diff:.3f}% from mid)"
                    log_msg += f" - Order ID: {self.active_order_id}"
                    logger.info(log_msg)
                    
                    # Start monitoring this new order for fills
                    await self._start_order_monitor(self.active_order_id)

                    logger.debug(f"Order details: {order_response.get('data', {})}")
                else:
                    logger.error(f"❌ Failed to place buy order: {order_response.get('msg', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error during order placement: {e}")
                logger.debug(f"Order placement exception: {type(e).__name__}: {e}")

        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            logger.debug(f"Buy order exception: {type(e).__name__}: {e}")

    async def _place_sell_order(self):
        """Places a limit sell order to close the current position."""
        logger.debug("Attempting to place sell order")
        if not self.current_ask or not self.current_position:
            logger.warning("No ask price or position available, cannot place sell order")
            logger.debug(f"Current ask: {self.current_ask}, position: {self.current_position}")
            return

        if not self.client:
            logger.warning("No authenticated client available for placing orders")
            return

        try:
            # Round ask price to tick size for comparison
            rounded_ask = self._round_price_to_tick_size(self.current_ask)

            # Check if we need to cancel existing orders first
            params = GetActiveOrderParams(size="100", offset_data="")
            active_orders_response = await self.client.get_active_orders(params)

            if active_orders_response.get("code") == "SUCCESS":
                active_orders = active_orders_response.get("data", {}).get("dataList", [])
                our_orders = [order for order in active_orders if order.get("contractId") == self.contract_id]

                if our_orders:
                    # Check if a sell order already exists at the target price
                    for order in our_orders:
                        order_price = Decimal(str(order.get("price", "0")))
                        order_side = order.get("side")
                        if order_side == "SELL" and order_price == rounded_ask:
                            logger.info(f"✅ Sell order at {rounded_ask} already exists. No action needed.")
                            return

                    logger.info(f"🔄 Price has moved or order is wrong type. Cancelling {len(our_orders)} active orders to replace.")
                    await self._cancel_active_orders()
                    logger.info(f"⏳ Waiting for {POST_CANCELLATION_DELAY}s for balance to update after cancellation...")
                    await asyncio.sleep(POST_CANCELLATION_DELAY)
                else:
                    logger.debug("✅ No active orders to cancel, proceeding with sell order placement")
            
            # Get current state AFTER any cancellations
            await self._update_account_state()

            # Get position size and round to step size
            raw_position_size = abs(Decimal(str(self.current_position.get("size", "0"))))
            position_size = self._round_size_to_step_size(raw_position_size)
            logger.debug(f"Position size to sell: {position_size} (rounded from {raw_position_size})")

            logger.debug(f"Using rounded ask price: {rounded_ask} (original: {self.current_ask})")

            # Check if we have sufficient balance for fees (if balance info is available)
            if self.available_balance is not None:
                if not self._check_sufficient_balance(rounded_ask, position_size, is_buy=False):
                    logger.error("❌ Insufficient balance for sell order fees - skipping")
                    return
            else:
                logger.debug("⚠️ No balance info available - placing sell order anyway")

            # Create sell order at rounded ask price
            order_params = CreateOrderParams(
                contract_id=self.contract_id,
                size=str(position_size),
                price=str(rounded_ask),
                type=OrderType.LIMIT,
                side=OrderSide.SELL,
                reduce_only=True,
                time_in_force=TimeInForce.POST_ONLY
            )
            logger.debug(f"Created sell order params: {order_params}")

            # Get metadata and place order
            try:
                logger.debug("Getting metadata for sell order placement")
                metadata = await self.client.get_metadata()
                logger.debug(f"Metadata for sell order: {metadata}")

                if metadata.get("code") != "SUCCESS":
                    logger.error(f"Cannot get metadata for order: {metadata.get('msg')}")
                    return

                logger.debug("Placing sell order")
                order_response = await self.client.order.create_order(order_params, metadata.get("data", {}))
                logger.debug(f"Sell order response: {order_response}")

                if order_response.get("code") == "SUCCESS":
                    self.active_order_id = order_response.get("data", {}).get("orderId")
                    self.last_order_price = rounded_ask
                    log_msg = f"✅ Placed SELL order at {rounded_ask} (size: {position_size})"
                    if self.market_mid_price and self.market_mid_price > 0:
                        pct_diff = ((rounded_ask - self.market_mid_price) / self.market_mid_price) * 100
                        log_msg += f" (+{pct_diff:.3f}% from mid)"
                    log_msg += f" - Order ID: {self.active_order_id}"
                    logger.info(log_msg)

                    # Start monitoring this new order for fills
                    await self._start_order_monitor(self.active_order_id)

                    logger.debug(f"Sell order details: {order_response.get('data', {})}")
                else:
                    logger.error(f"❌ Failed to place sell order: {order_response.get('msg', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error during sell order placement: {e}")
                logger.debug(f"Sell order placement exception: {type(e).__name__}: {e}")

        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
            logger.debug(f"Sell order exception: {type(e).__name__}: {e}")

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
                # --- Update market prices via REST API ---
                prices_updated = await self._update_prices_from_rest_api()
                if not prices_updated:
                    logger.warning("Could not update prices, skipping this iteration.")
                    await asyncio.sleep(self.refresh_interval)
                    continue

                # Update account state (position and balance)
                await self._update_account_state()

                # Calculate and display total traded volume (periodically)
                if USE_MAX_VOLUME_LIMIT and loop_count % 10 == 1:  # Check trading volume every 10th iteration
                    _, should_continue = await self._calculate_total_traded_volume()
                    if not should_continue and self.trading_allowed:
                        self.trading_allowed = False
                        logger.info("🛑 Volume limit reached. Cancelling active orders and preparing to close position.")
                        await self._cancel_active_orders()

                # Log current market data
                if self.current_bid and self.current_ask:
                    spread = self.current_ask - self.current_bid
                    spread_pct = (spread / self.current_bid) * 100 if self.current_bid > 0 else 0
                    logger.info(f"📊 Market: Bid={self.current_bid}, Ask={self.current_ask}, Spread={spread:.6f} ({spread_pct:.3f}%)")
                else:
                    logger.warning("⚠️ No market data available yet")
                    logger.debug(f"Market data state - bid: {self.current_bid}, ask: {self.current_ask}")

                # Decide action based on position (only if we have valid credentials and trading is allowed)
                if self.client:
                    if not self.trading_allowed:
                        logger.info("🛑 Trading disabled due to maximum volume exceeded.")
                        if self.current_position:
                            position_size = self.current_position.get("size", "0")
                            logger.info(f"📈 Position found (size: {position_size}), attempting to place final sell order to close...")
                            await self._place_sell_order()
                        else:
                            logger.info("✅ No remaining position. Stopping bot.")
                            await self.stop()
                            break  # Exit trading loop
                    else:
                        # Check order cooldown
                        if self.last_order_time is not None:
                            time_since_last_order = time.time() - self.last_order_time
                            if time_since_last_order < self.min_order_interval:
                                remaining = self.min_order_interval - time_since_last_order
                                logger.info(f"⏳ Order cooldown: waiting {remaining:.1f}s before next order...")
                                await asyncio.sleep(remaining)

                        # If all checks pass, execute trade
                        await self._execute_trading_decision()

                else:
                    logger.info("🔄 Running in market data mode only (no trading due to auth issues)")

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
    leverage = float(os.getenv("EDGEX_LEVERAGE", str(LEVERAGE)))
    use_avellaneda_spread = os.getenv("EDGEX_USE_AVELLANEDA_SPREAD", str(USE_AVELLANEDA_SPREAD)).lower() in ("true", "1", "yes", "on")
    avellaneda_params_file = os.getenv("EDGEX_AVELLANEDA_PARAMS_FILE", AVELLANEDA_PARAMS_FILE)

    logger.debug(f"Configuration loaded:")
    logger.debug(f"  base_url: {base_url}")
    logger.debug(f"  ws_url: {ws_url}")
    logger.debug(f"  account_id: {account_id}")
    logger.debug(f"  has_private_key: {bool(stark_private_key)}")
    logger.debug(f"  leverage: {leverage}")
    logger.debug(f"  debug_logging: {debug_logging}")
    logger.debug(f"  log_file: {log_file_path}")
    logger.debug(f"  use_avellaneda_spread: {use_avellaneda_spread}")
    logger.debug(f"  avellaneda_params_file: {avellaneda_params_file}")

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
        use_avellaneda_spread=use_avellaneda_spread,
        avellaneda_params_file=avellaneda_params_file
    )
    logger.debug("Bot instance created")

    # Set up signal handlers for graceful shutdown (cross-platform)
    def signal_handler():
        logger.info("Received shutdown signal...")
        asyncio.create_task(bot.stop())

    if platform.system() == "Windows":
        # Windows doesn't support SIGTERM and has limited signal handling
        # Use signal.signal() for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, lambda s, f: signal_handler())
    else:
        # Unix-like systems (Linux, macOS)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

    await bot.start()
    logger.debug("=== MAIN FUNCTION END ===")


if __name__ == "__main__":
    asyncio.run(main())