"""
lighter_connector.py
--------------------
Lighter exchange connector - handles all Lighter-specific operations.
Extracted from hedge_cli.py and auto_rotation_bot.py for better modularity.
"""

import asyncio
import json
import logging
import time
import websockets
from typing import Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
import lighter

logger = logging.getLogger(__name__)


# ==================== Utility Functions ====================

def _round_to_tick(value: float, tick: float) -> float:
    """Round value to nearest tick size using Decimal for precision."""
    if not tick or tick <= 0:
        return value
    d_value = Decimal(str(value))
    d_tick = Decimal(str(tick))
    return float((d_value / d_tick).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * d_tick)


def _floor_to_tick(value: float, tick: float) -> float:
    """Floor value to tick size using Decimal for precision."""
    if not tick or tick <= 0:
        return value
    d_value = Decimal(str(value))
    d_tick = Decimal(str(tick))
    from decimal import ROUND_DOWN
    return float((d_value / d_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * d_tick)


def cross_price(side: str, ref_bid: float, ref_ask: float, tick: float, cross_ticks: int = 10) -> float:
    """
    Calculate aggressive limit order price that crosses the spread.

    Args:
        side: "buy" or "sell"
        ref_bid: Best bid price
        ref_ask: Best ask price
        tick: Tick size for rounding
        cross_ticks: Number of ticks to cross the spread

    Returns:
        Aggressive price (buy = ask + N ticks, sell = bid - N ticks)
    """
    if side.lower() == "buy":
        # Buy at ask + cross_ticks (ensures fill)
        base = ref_ask if ref_ask else ref_bid
        aggressive = base + (cross_ticks * tick) if base else 0.0
    else:
        # Sell at bid - cross_ticks (ensures fill)
        base = ref_bid if ref_bid else ref_ask
        aggressive = base - (cross_ticks * tick) if base else 0.0
    return _round_to_tick(aggressive, tick)


# ==================== Balance Functions ====================

async def get_balance(env: dict) -> Tuple[float, float]:
    """
    Get Lighter total and available USD balance via WebSocket.

    Args:
        env: Environment dict with Lighter credentials

    Returns:
        Tuple of (total_usd, available_usd)
    """
    try:
        account_index = int(env.get("ACCOUNT_INDEX", env.get("LIGHTER_ACCOUNT_INDEX", "0")))
        ws_url = env["LIGHTER_WS_URL"]

        available, portfolio_value = await get_available_capital_ws(
            ws_url=ws_url,
            account_index=account_index,
            timeout=10.0
        )

        if available is None or portfolio_value is None:
            raise Exception("Lighter WebSocket returned None values")

        # Return (total, available) - portfolio_value is the total
        return portfolio_value, available

    except Exception as e:
        logger.error(f"Error fetching Lighter balance: {type(e).__name__}: {e}", exc_info=True)
        raise


async def get_available_capital_ws(ws_url: str, account_index: int, timeout: float = 10.0) -> Tuple[Optional[float], Optional[float]]:
    """
    Connect to Lighter WebSocket to get available balance and portfolio value.

    Args:
        ws_url: WebSocket URL
        account_index: Account index
        timeout: Timeout in seconds

    Returns:
        Tuple of (available_balance, portfolio_value)
    """
    sub = {"type": "subscribe", "channel": f"user_stats/{account_index}"}
    start = time.time()
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps(sub))
            while (time.time() - start) < timeout:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout - (time.time() - start))
                data = json.loads(msg)
                t = data.get("type")
                if t in ("update/user_stats", "subscribed/user_stats"):
                    stats = data.get("stats", {})
                    avail = float(stats.get("available_balance", 0) or 0)
                    portv = float(stats.get("portfolio_value", 0) or 0)
                    if avail > 0 or portv > 0:
                        return avail, portv
    except Exception as e:
        logger.error(f"Lighter WebSocket error: {e}")
        return None, None
    return None, None


# ==================== Market Discovery ====================

async def get_market_details(order_api, symbol: str) -> Tuple[int, float, float]:
    """
    Get market ID, price tick, and amount tick for a symbol.

    Args:
        order_api: Lighter OrderApi instance
        symbol: Symbol name (e.g., "PAXG")

    Returns:
        Tuple of (market_id, price_tick, amount_tick)
    """
    resp = await order_api.order_books()
    for ob in resp.order_books:
        if ob.symbol.upper() == symbol.upper():
            market_id = ob.market_id
            price_tick = 10 ** -ob.supported_price_decimals
            amount_tick = 10 ** -ob.supported_size_decimals
            return market_id, price_tick, amount_tick
    raise RuntimeError(f"Lighter: symbol {symbol} not found.")


# ==================== Price Functions ====================

class OrderBookFetcher:
    """Helper class to fetch order book snapshot from Lighter WebSocket."""

    def __init__(self, symbol: str, market_id: int):
        self.symbol = symbol
        self.market_id = market_id
        self.best_bid = None
        self.best_ask = None
        self.received_event = asyncio.Event()
        self.update_count = 0

    def on_order_book_update(self, mid, order_book):
        """Callback for order book updates."""
        self.update_count += 1
        logger.debug(f"Lighter callback: update #{self.update_count}, market_id={mid}, target={self.market_id}")

        if int(mid) == int(self.market_id):
            try:
                bids = order_book.get('bids', [])
                asks = order_book.get('asks', [])
                logger.debug(f"Lighter {self.symbol}: Received {len(bids)} bids, {len(asks)} asks")

                if bids and asks:
                    self.best_bid = float(bids[0]['price'])
                    self.best_ask = float(asks[0]['price'])
                    logger.debug(f"Lighter {self.symbol}: bid={self.best_bid}, ask={self.best_ask}")
                    self.received_event.set()
                else:
                    logger.warning(f"Lighter {self.symbol}: Empty order book")
                    self.received_event.set()  # Set even if empty
            except Exception as e:
                logger.error(f"Error parsing Lighter order book: {e}")
                self.received_event.set()

    def on_account_update(self, account_id, update):
        """Callback for account updates (not used)."""
        pass


async def get_best_bid_ask(order_api, symbol: str, market_id: int, timeout: float = 10.0) -> Tuple[Optional[float], Optional[float]]:
    """
    Get best bid/ask from Lighter using WebSocket.

    Args:
        order_api: Lighter OrderApi instance
        symbol: Symbol name
        market_id: Market ID
        timeout: Timeout in seconds

    Returns:
        Tuple of (best_bid, best_ask) or (None, None) if unavailable
    """
    logger.debug(f"Fetching Lighter prices for {symbol} (market_id={market_id}) via WebSocket...")

    fetcher = OrderBookFetcher(symbol, market_id)

    try:
        # Create WebSocket client for this market only
        ws_client = lighter.WsClient(
            order_book_ids=[market_id],
            account_ids=[],
            on_order_book_update=fetcher.on_order_book_update,
            on_account_update=fetcher.on_account_update,
        )

        # Run WebSocket in background and wait for first order book update
        ws_task = asyncio.create_task(ws_client.run_async())

        try:
            await asyncio.wait_for(fetcher.received_event.wait(), timeout=timeout)
            logger.debug(f"Lighter: Received {fetcher.update_count} updates for {symbol}")
        except asyncio.TimeoutError:
            logger.warning(f"Lighter: Timeout waiting for {symbol} order book update")
        finally:
            # Cancel WebSocket task and close client
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass

            # Ensure WebSocket connection is closed
            try:
                if hasattr(ws_client, 'close'):
                    await ws_client.close()
                elif hasattr(ws_client, 'disconnect'):
                    await ws_client.disconnect()
            except Exception as e:
                logger.debug(f"Error closing WsClient: {e}")

        return fetcher.best_bid, fetcher.best_ask

    except Exception as e:
        logger.error(f"Lighter WebSocket error for {symbol}: {e}", exc_info=True)
        return None, None


async def get_mark_price(order_api, symbol: str, market_id: int) -> Optional[float]:
    """
    Get current mark price (mid of bid/ask).

    Args:
        order_api: Lighter OrderApi instance
        symbol: Symbol name
        market_id: Market ID

    Returns:
        Mark price or None if unavailable
    """
    bid, ask = await get_best_bid_ask(order_api, symbol, market_id)
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    elif bid is not None or ask is not None:
        return float(bid or ask)
    return None


# ==================== Leverage Management ====================

async def set_leverage(signer: lighter.SignerClient, market_id: int, leverage: int, margin_mode: str = "cross") -> None:
    """
    Set leverage for a market.

    Args:
        signer: Lighter SignerClient instance
        market_id: Market ID
        leverage: Target leverage (e.g., 3)
        margin_mode: "cross" or "isolated"
    """
    mmode = signer.CROSS_MARGIN_MODE if margin_mode == "cross" else signer.ISOLATED_MARGIN_MODE
    _, _, err = await signer.update_leverage(market_id, mmode, int(leverage))
    if err:
        raise RuntimeError(f"Lighter: leverage update failed: {err}")


async def get_leverage(signer: lighter.SignerClient, market_id: int) -> Optional[float]:
    """
    Get current leverage (not supported by Lighter API).

    Returns:
        None (Lighter doesn't expose current leverage)
    """
    # Lighter doesn't provide an easy way to query current leverage
    return None


# ==================== Position Management ====================

async def get_open_size(account_api: lighter.AccountApi, account_index: int, market_id: int) -> float:
    """
    Get open position size for a market (signed: positive=long, negative=short).

    Args:
        account_api: Lighter AccountApi instance
        account_index: Account index
        market_id: Market ID

    Returns:
        Position size (0.0 if no position)
    """
    logger.debug(f"Lighter: Fetching position for account {account_index}, market {market_id}")
    try:
        account_details_response = await account_api.account(by="index", value=str(account_index))
        logger.debug(f"Lighter account details response: {account_details_response}")

        if not (account_details_response and account_details_response.accounts):
            logger.debug("Lighter: Account details response is empty or has no accounts.")
            return 0.0

        acc = account_details_response.accounts[0]
        if not acc.positions:
            logger.debug(f"Lighter: No positions found for account {account_index}.")
            return 0.0

        for pos in acc.positions:
            if pos.market_id == market_id:
                size = float(pos.position)
                sign = int(pos.sign)

                if size == 0:
                    return 0.0

                # sign is 1 for Long, -1 for Short
                signed_size = size * sign
                logger.debug(f"Lighter: Found position for market {market_id}: size={size}, sign={sign}, signed_size={signed_size}")
                return signed_size

        logger.debug(f"Lighter: No position found for market {market_id} in account {account_index}.")
        return 0.0

    except Exception as e:
        logger.error(f"Lighter: Failed to get account position size: {e}", exc_info=True)
        return 0.0


async def get_position_details(account_api: lighter.AccountApi, account_index: int, market_id: int) -> dict:
    """
    Get detailed position information.

    Args:
        account_api: Lighter AccountApi instance
        account_index: Account index
        market_id: Market ID

    Returns:
        Dict with position details (size, side, entry_price, unrealized_pnl, etc.)
    """
    try:
        account_details_response = await account_api.account(by="index", value=str(account_index))

        if not (account_details_response and account_details_response.accounts):
            return {'size': 0.0, 'unrealized_pnl': 0.0, 'entry_price': 0.0, 'leverage': 0.0, 'open_timestamp': None, 'side': 'none'}

        acc = account_details_response.accounts[0]
        if not acc.positions:
            return {'size': 0.0, 'unrealized_pnl': 0.0, 'entry_price': 0.0, 'leverage': 0.0, 'open_timestamp': None, 'side': 'none'}

        for pos in acc.positions:
            if pos.market_id == market_id:
                size = float(pos.position)
                sign = int(pos.sign)
                signed_size = size * sign if size != 0 else 0.0

                side = "long" if sign > 0 else "short" if sign < 0 else "none"
                entry_price = float(pos.entry_price) if hasattr(pos, 'entry_price') else 0.0
                unrealized_pnl = float(pos.unrealized_pnl) if hasattr(pos, 'unrealized_pnl') else 0.0
                leverage = float(pos.leverage) if hasattr(pos, 'leverage') else 0.0
                open_timestamp = pos.open_timestamp if hasattr(pos, 'open_timestamp') else None

                return {
                    'size': signed_size,
                    'side': side,
                    'entry_price': entry_price,
                    'unrealized_pnl': unrealized_pnl,
                    'leverage': leverage,
                    'open_timestamp': open_timestamp
                }

        return {'size': 0.0, 'unrealized_pnl': 0.0, 'entry_price': 0.0, 'leverage': 0.0, 'open_timestamp': None, 'side': 'none'}

    except Exception as e:
        logger.error(f"Lighter: Failed to get position details: {e}", exc_info=True)
        return {'size': 0.0, 'unrealized_pnl': 0.0, 'entry_price': 0.0, 'leverage': 0.0, 'open_timestamp': None, 'side': 'none'}


async def get_position_pnl(account_api: lighter.AccountApi, account_index: int, market_id: int) -> float:
    """
    Get unrealized PnL for a position.

    Args:
        account_api: Lighter AccountApi instance
        account_index: Account index
        market_id: Market ID

    Returns:
        Unrealized PnL in USD
    """
    details = await get_position_details(account_api, account_index, market_id)
    return details.get('unrealized_pnl', 0.0)


# ==================== Order Execution ====================

async def place_aggressive_order(
    signer: lighter.SignerClient,
    market_id: int,
    price_tick: float,
    amount_tick: float,
    side: str,
    size_base: float,
    ref_price: float,
    cross_ticks: int = 10
) -> str:
    """
    Place aggressive limit order (crosses spread for immediate fill).

    Args:
        signer: Lighter SignerClient instance
        market_id: Market ID
        price_tick: Price tick size
        amount_tick: Size tick size
        side: "buy" or "sell"
        size_base: Order size in base units (should be pre-rounded)
        ref_price: Reference price
        cross_ticks: Number of ticks to cross spread

    Returns:
        Client order ID
    """
    if ref_price is None:
        raise RuntimeError(f"Lighter: ref_price is None for {side} order")

    px = cross_price(side, ref_bid=None if side == 'buy' else ref_price, ref_ask=ref_price if side == 'buy' else None, tick=price_tick, cross_ticks=cross_ticks)

    # Size should already be rounded - just scale to integer units
    base_scaled = int(round(size_base / amount_tick))
    price_scaled = int(px / price_tick)

    logger.debug(f"Lighter order: {side} {base_scaled} units @ {price_scaled} scaled ({px:.4f} actual)")

    client_order_id = int(asyncio.get_running_loop().time() * 1_000_000) % 1_000_000

    # Use create_order with GOOD_TILL_TIME for aggressive crossing orders
    tx, tx_hash, err = await signer.create_order(
        market_index=market_id,
        client_order_index=client_order_id,
        base_amount=base_scaled,
        price=price_scaled,
        is_ask=True if side == "sell" else False,
        order_type=lighter.SignerClient.ORDER_TYPE_LIMIT,
        time_in_force=lighter.SignerClient.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
        reduce_only=0,  # 0 = False, 1 = True
        trigger_price=0,
    )
    if err:
        raise RuntimeError(f"Lighter: order error: {err}")

    logger.info(f"Lighter order placed: tx_hash={getattr(tx_hash, 'tx_hash', tx_hash)}")
    return str(client_order_id)


async def close_position(
    signer: lighter.SignerClient,
    market_id: int,
    price_tick: float,
    amount_tick: float,
    side: str,
    size_base: float,
    ref_price: float,
    cross_ticks: int = 10
) -> None:
    """
    Close position with reduce-only aggressive limit order.

    Args:
        signer: Lighter SignerClient instance
        market_id: Market ID
        price_tick: Price tick size
        amount_tick: Size tick size
        side: "buy" to close short, "sell" to close long
        size_base: Position size to close
        ref_price: Reference price
        cross_ticks: Number of ticks to cross spread
    """
    logger.debug(f"Lighter: Closing position - market_id={market_id}, side={side}, size={size_base}")

    if ref_price is None:
        logger.error("Lighter close failed: ref_price is None.")
        raise RuntimeError(f"Lighter: ref_price is None for {side} reduce-only order")

    # Use a large size so reduce-only will close whatever position exists
    base_scaled = int(round(size_base / amount_tick))

    # For a closing order, we want to cross the spread to get filled
    px = cross_price(side, ref_bid=None if side == 'buy' else ref_price, ref_ask=ref_price if side == 'buy' else None, tick=price_tick, cross_ticks=cross_ticks)
    price_scaled = int(px / price_tick)

    logger.debug(f"Lighter: base_scaled={base_scaled}, aggressive_price={px}, price_scaled={price_scaled}")

    order_id = int(asyncio.get_running_loop().time() * 1_000_000) % 1_000_000

    order_params = {
        "market_index": market_id,
        "client_order_index": order_id,
        "base_amount": base_scaled,
        "price": price_scaled,
        "is_ask": True if side == "sell" else False,
        "order_type": lighter.SignerClient.ORDER_TYPE_LIMIT,
        "time_in_force": lighter.SignerClient.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
        "reduce_only": 1,  # 1 = True
        "trigger_price": 0,
    }

    tx, tx_hash, err = await signer.create_order(**order_params)

    if err is not None:
        logger.warning(f"Lighter reduce-only limit {side.upper()} returned error: {err}")
        err_str = str(err).lower()
        if 'no open position' not in err_str and 'no position' not in err_str:
            raise RuntimeError(f"Lighter {side.upper()} reduce-only limit order failed: {err}")

    logger.info(f"Lighter reduce-only limit order sent successfully: tx={tx}, tx_hash={tx_hash}")


# ==================== Funding Rate Info ====================

async def get_funding_info(api_client: lighter.ApiClient, market_id: int, symbol: str, quote: str) -> Optional[float]:
    """
    Get funding rate for a market (annualized %).

    Args:
        api_client: Lighter ApiClient instance
        market_id: Market ID
        symbol: Base symbol (e.g., "PAXG")
        quote: Quote currency (e.g., "USD")

    Returns:
        Funding rate as annualized percentage or None
    """
    try:
        candles_api = lighter.CandlesticksApi(api_client)
        # Fetch recent candles to get funding rate data
        # Note: Implementation depends on available Lighter API endpoints
        # This is a placeholder - actual implementation may vary
        logger.warning("Lighter funding rate fetch not fully implemented")
        return None
    except Exception as e:
        logger.error(f"Error getting Lighter funding: {e}")
        return None
