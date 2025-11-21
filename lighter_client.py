#!/usr/bin/env python3
"""
lighter_client.py
-----------------
Lighter exchange connector with helper functions for balance, positions, and order management.
"""

import asyncio
import json
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from typing import Dict, Optional, Tuple

import lighter

logger = logging.getLogger(__name__)


class BalanceFetchError(Exception):
    """Raised when balance retrieval fails."""
    pass


def _round_to_tick(value: float, tick: float) -> float:
    """Round value to nearest tick."""
    if not tick or tick <= 0:
        return value
    d_value = Decimal(str(value))
    d_tick = Decimal(str(tick))
    return float((d_value / d_tick).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * d_tick)


def _ceil_to_tick(value: float, tick: float) -> float:
    """Round value up to nearest tick."""
    if not tick or tick <= 0:
        return value
    d_value = Decimal(str(value))
    d_tick = Decimal(str(tick))
    return float((d_value / d_tick).quantize(Decimal('1'), rounding=ROUND_UP) * d_tick)


def _floor_to_tick(value: float, tick: float) -> float:
    """Round value down to nearest tick."""
    if not tick or tick <= 0:
        return value
    d_value = Decimal(str(value))
    d_tick = Decimal(str(tick))
    return float((d_value / d_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * d_tick)


def cross_price(side: str, ref_bid: Optional[float], ref_ask: Optional[float], tick: float, cross_pct: float = 3.0) -> float:
    """
    Return an aggressive price based on percentage from mid price.
    - BUY orders: mid * (1 + cross_pct/100)
    - SELL orders: mid * (1 - cross_pct/100)
    Falls back to available prices if bid/ask is missing.

    Args:
        side: "buy" or "sell"
        ref_bid: Best bid price
        ref_ask: Best ask price
        tick: Tick size for rounding
        cross_pct: Percentage to cross from mid price (default 3%, well under 5% exchange limit)
    """
    # Calculate mid price
    if ref_bid is not None and ref_ask is not None:
        mid = (ref_bid + ref_ask) / 2.0
    elif ref_ask is not None:
        mid = ref_ask
    elif ref_bid is not None:
        mid = ref_bid
    else:
        return 0.0

    # Apply percentage adjustment
    cross_pct = max(0.0, cross_pct)  # Ensure non-negative
    if side == "buy":
        price = mid * (1.0 + cross_pct / 100.0)
        return _ceil_to_tick(price, tick)
    else:  # sell
        price = mid * (1.0 - cross_pct / 100.0)
        return _floor_to_tick(price, tick)


class LighterOrderBookFetcher:
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
        logger.debug(f"Lighter callback triggered: update #{self.update_count}, market_id={mid}, target={self.market_id}")

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
                    logger.warning(f"Lighter {self.symbol}: Empty order book (bids={len(bids)}, asks={len(asks)})")
                    self.received_event.set()  # Set even if empty
            except Exception as e:
                logger.error(f"Error parsing Lighter order book: {e}")
                logger.error(f"Order book structure: {order_book}")
                self.received_event.set()

    def on_account_update(self, account_id, update):
        """Callback for account updates (not used)."""
        pass


async def get_lighter_balance(ws_url: str, account_index: int, timeout: float = 10.0) -> Tuple[float, float]:
    """
    Get Lighter total and available USD balance via WebSocket.

    Args:
        ws_url: Lighter WebSocket URL
        account_index: Account index
        timeout: Timeout in seconds

    Returns:
        Tuple of (available_balance, portfolio_value) - NOTE: available first, then portfolio
    """
    import time
    import websockets

    sub = {"type": "subscribe", "channel": f"user_stats/{account_index}"}
    start = time.time()
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps(sub))
            while (time.time() - start) < timeout:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout - (time.time() - start))
                data = json.loads(msg)
                t = data.get("type")
                # Check for the correct message types from Lighter WebSocket
                if t in ("update/user_stats", "subscribed/user_stats"):
                    stats = data.get("stats", {})
                    avail = float(stats.get("available_balance", 0) or 0)
                    portv = float(stats.get("portfolio_value", 0) or 0)
                    if avail > 0 or portv > 0:
                        logger.debug(f"Lighter balance: available=${avail:.2f}, portfolio=${portv:.2f}")
                        return avail, portv
        raise BalanceFetchError("Lighter WebSocket: timeout waiting for user_stats")
    except asyncio.TimeoutError:
        raise BalanceFetchError("Lighter WebSocket: timeout")
    except Exception as e:
        logger.error(f"Error fetching Lighter balance: {e}", exc_info=True)
        raise BalanceFetchError(f"Lighter balance fetch failed: {e}") from e


async def get_lighter_market_details(order_api, symbol: str) -> Tuple[int, float, float]:
    """
    Get market details for a Lighter symbol.

    Args:
        order_api: Lighter OrderApi instance
        symbol: Symbol to query

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
    raise ValueError(f"Symbol {symbol} not found on Lighter")


async def get_lighter_best_bid_ask(order_api, symbol: str, market_id: int, timeout: float = 10.0) -> Tuple[Optional[float], Optional[float]]:
    """
    Get best bid/ask from Lighter using WebSocket (REST API returns empty order books).
    Connects briefly to WebSocket, waits for order book update, then returns prices.

    Args:
        order_api: Lighter OrderApi instance (not used, kept for compatibility)
        symbol: Symbol to query
        market_id: Market ID
        timeout: Timeout in seconds

    Returns:
        Tuple of (best_bid, best_ask)
    """
    logger.info(f"Fetching Lighter prices for {symbol} (market_id={market_id}) via WebSocket...")

    fetcher = LighterOrderBookFetcher(symbol, market_id)

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
            logger.info(f"Lighter: Received {fetcher.update_count} updates for {symbol}")
        except asyncio.TimeoutError:
            logger.warning(f"Lighter: Timeout waiting for {symbol} order book update ({fetcher.update_count} updates received)")
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


async def lighter_set_leverage(
    signer: lighter.SignerClient,
    market_id: int,
    leverage: int,
    margin_mode: str = "cross"
) -> None:
    """
    Update leverage configuration for a market. Margin mode defaults to cross.
    """
    mmode = signer.CROSS_MARGIN_MODE if margin_mode == "cross" else signer.ISOLATED_MARGIN_MODE
    _, _, err = await signer.update_leverage(market_id, mmode, int(leverage))
    if err:
        raise RuntimeError(f"Lighter: leverage update failed: {err}")


async def get_lighter_open_size(account_api, account_index: int, market_id: int, symbol: Optional[str] = None) -> float:
    """
    Get signed position size for a given market on Lighter.

    Args:
        account_api: Lighter AccountApi instance
        account_index: Account index
        market_id: Market ID

    Returns:
        Signed position size (positive=long, negative=short)
    """
    market_label = f"{symbol} (market {market_id})" if symbol else f"market {market_id}"
    logger.info(f"Lighter: Fetching position for account {account_index}, {market_label}")
    try:
        # The API expects `value` as a string for the account index
        account_details_response = await account_api.account(by="index", value=str(account_index))
        logger.debug(f"Lighter account details response: {account_details_response}")

        # The response is a DetailedAccounts object which has an 'accounts' list
        if not (account_details_response and account_details_response.accounts):
            logger.warning("Lighter: Account details response is empty or has no accounts.")
            return 0.0

        # We expect one account when querying by index
        acc = account_details_response.accounts[0]
        if not acc.positions:
            logger.info(f"Lighter: No positions found for account {account_index}.")
            return 0.0

        for pos in acc.positions:
            if pos.market_id == market_id:
                size = float(pos.position)
                sign = int(pos.sign)

                if size == 0:
                    return 0.0

                # sign is 1 for Long, -1 for Short
                signed_size = size * sign
                logger.info(
                    "Lighter: Found position for %s: size=%s, sign=%s, signed_size=%s",
                    market_label,
                    size,
                    sign,
                    signed_size,
                )
                return signed_size

        logger.info(f"Lighter: No position found for {market_label} in account {account_index}.")
        return 0.0

    except Exception as e:
        logger.error(f"Lighter: Failed to get account position size due to an error: {e}", exc_info=True)
        raise RuntimeError(f"Lighter: Failed to get account position size: {e}") from e


async def get_lighter_position_pnl(account_api, account_index: int, market_id: int) -> float:
    """
    Get unrealized PnL for a Lighter position.

    Args:
        account_api: Lighter AccountApi instance
        account_index: Account index
        market_id: Market ID

    Returns:
        Unrealized PnL in USD
    """
    try:
        account_details = await account_api.account(by="index", value=str(account_index))

        if not (account_details and account_details.accounts):
            return 0.0

        acc = account_details.accounts[0]
        if not acc.positions:
            return 0.0

        for pos in acc.positions:
            if pos.market_id == market_id:
                return float(pos.unrealized_pnl or "0")

        return 0.0

    except Exception as e:
        logger.error(f"Lighter: Failed to get PnL: {e}")
        return 0.0


async def get_lighter_position_details(
    account_api,
    account_index: int,
    market_id: int,
) -> Optional[Dict[str, float]]:
    """
    Fetch detailed position info (size, entry price, unrealized PnL) for a market.

    Args:
        account_api: Lighter AccountApi instance
        account_index: Account index
        market_id: Market ID

    Returns:
        Dict with keys: size (signed), abs_size, side, entry_price, unrealized_pnl
        or None if no position is found.
    """
    try:
        account_details = await account_api.account(by="index", value=str(account_index))

        if not (account_details and account_details.accounts):
            return None

        acc = account_details.accounts[0]
        positions = getattr(acc, "positions", None)
        if not positions:
            return None

        for pos in positions:
            if pos.market_id != market_id:
                continue

            raw_size = float(pos.position or "0")
            sign = int(pos.sign or 0)
            signed_size = raw_size * sign
            entry_price = float(pos.avg_entry_price or "0")
            unrealized_pnl = float(pos.unrealized_pnl or "0")
            try:
                imf = float(pos.initial_margin_fraction or "0")
            except Exception:
                imf = 0.0
            margin_mode = int(getattr(pos, "margin_mode", 0))
            leverage = (100.0 / imf) if imf > 0 else 0.0

            return {
                "size": signed_size,
                "abs_size": abs(signed_size),
                "side": "LONG" if signed_size > 0 else ("SHORT" if signed_size < 0 else "FLAT"),
                "entry_price": entry_price,
                "unrealized_pnl": unrealized_pnl,
                "initial_margin_fraction": imf,
                "margin_mode": margin_mode,
                "leverage": leverage,
            }

        return None

    except Exception as e:
        logger.error(f"Lighter: Failed to fetch detailed position info: {e}", exc_info=True)
        raise RuntimeError(f"Lighter: Failed to fetch detailed position info: {e}") from e


async def get_all_lighter_positions(account_api, account_index: int) -> list:
    """
    Get all non-zero positions on Lighter.

    Args:
        account_api: Lighter AccountApi instance
        account_index: Account index

    Returns:
        List of dicts with keys: symbol, size (signed), entry_price, unrealized_pnl
    """
    positions = []
    try:
        account_details_response = await account_api.account(by="index", value=str(account_index))

        if not (account_details_response and account_details_response.accounts):
            logger.info(f"Lighter: No account found for index {account_index}")
            return positions

        acc = account_details_response.accounts[0]
        if not acc.positions:
            logger.info(f"Lighter: No positions found for account {account_index}")
            return positions

        for pos in acc.positions:
            # Extract position data using correct attribute names
            raw_size = float(pos.position or "0") if hasattr(pos, 'position') else 0.0
            sign = int(pos.sign or 0) if hasattr(pos, 'sign') else 1
            signed_size = raw_size * sign

            # Only include non-zero positions
            if abs(signed_size) > 1e-8:
                positions.append({
                    'symbol': pos.symbol if hasattr(pos, 'symbol') else 'UNKNOWN',
                    'size': signed_size,
                    'entry_price': float(pos.avg_entry_price or "0") if hasattr(pos, 'avg_entry_price') else 0.0,
                    'unrealized_pnl': float(pos.unrealized_pnl or "0") if hasattr(pos, 'unrealized_pnl') else 0.0
                })

        logger.info(f"Lighter: Found {len(positions)} non-zero positions for account {account_index}")

    except Exception as e:
        logger.error(f"Lighter: Failed to get all positions: {e}", exc_info=True)

    return positions


async def get_lighter_funding_rate(api_or_client, market_id: int) -> Optional[float]:
    """
    Get current funding rate from Lighter for a specific market.

    Args:
        api_or_client: Lighter FundingApi instance or ApiClient
        market_id: Market ID

    Returns:
        Hourly funding rate as percentage (e.g., 0.01 = 0.01% per hour)
    """
    try:
        if isinstance(api_or_client, lighter.FundingApi):
            funding_api = api_or_client
        else:
            funding_api = lighter.FundingApi(api_or_client)

        resp = await funding_api.funding_rates()
        if not resp or not getattr(resp, "funding_rates", None):
            return None

        for rate_entry in resp.funding_rates:
            try:
                if int(rate_entry.market_id) == int(market_id) and getattr(rate_entry, "exchange", "").lower() == "lighter":
                    return float(rate_entry.rate)
            except Exception:
                continue
        return None
    except Exception as e:
        logger.warning(f"Could not fetch Lighter funding rate for market {market_id}: {e}")
        return None


async def lighter_place_aggressive_order(
    signer: lighter.SignerClient,
    market_id: int,
    price_tick: float,
    amount_tick: float,
    side: str,  # "buy" or "sell"
    size_base: float,
    bid: Optional[float],
    ask: Optional[float],
    cross_pct: float = 3.0  # Percentage from mid price (default 3%, well under 5% exchange limit)
) -> Optional[str]:
    """
    Place an aggressive limit order that crosses the spread to emulate a market order.
    Uses percentage-based pricing from mid price for consistent execution across all assets.

    Args:
        signer: Lighter SignerClient
        market_id: Market ID
        price_tick: Minimum price increment
        amount_tick: Minimum size increment
        side: "buy" or "sell"
        size_base: Order size in base currency (should already be rounded to tick)
        bid: Best bid price
        ask: Best ask price
        cross_pct: Percentage from mid price (default 5%)

    Returns:
        Order ID if successful, None otherwise
    """
    try:
        if bid is None and ask is None:
            logger.error(f"Lighter: both bid and ask are None for {side} order")
            return None

        # Calculate mid price for logging
        mid = (bid + ask) / 2.0 if (bid and ask) else (bid or ask)

        # Calculate aggressive price that crosses from mid
        px = cross_price(
            side,
            ref_bid=bid,
            ref_ask=ask,
            tick=price_tick,
            cross_pct=cross_pct
        )

        # Size should already be rounded - just scale to integer units
        base_scaled = int(round(size_base / amount_tick))
        price_scaled = int(px / price_tick)

        logger.info(f"Lighter order: {side} {base_scaled} units @ {price_scaled} scaled ({px:.4f} actual, mid={mid:.2f}, cross_pct={cross_pct:.1f}%)")

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
            logger.error(f"Lighter: order error: {err}")
            return None

        logger.info(f"Lighter order placed: tx_hash={getattr(tx_hash, 'tx_hash', tx_hash)}")
        return str(client_order_id)

    except Exception as e:
        logger.error(f"Lighter: Failed to place order: {e}", exc_info=True)
        return None


async def lighter_close_position(
    signer: lighter.SignerClient,
    market_id: int,
    price_tick: float,
    amount_tick: float,
    side: str,  # "buy" to close short, "sell" to close long
    size_base: float,
    bid: Optional[float],
    ask: Optional[float],
    cross_pct: float = 3.0  # Percentage from mid price (default 3%, well under 5% exchange limit)
) -> bool:
    """
    Close a position with a reduce-only aggressive limit order.
    Uses percentage-based pricing from mid price for consistent execution across all assets.

    Args:
        signer: Lighter SignerClient
        market_id: Market ID
        price_tick: Minimum price increment
        amount_tick: Minimum size increment
        side: "buy" to close short, "sell" to close long
        size_base: Position size to close (absolute value)
        bid: Best bid price
        ask: Best ask price
        cross_pct: Percentage from mid price (default 5%)

    Returns:
        True if order placed successfully, False otherwise
    """
    logger.info("--- lighter_close_position called ---")
    logger.info(f"Inputs: market_id={market_id}, price_tick={price_tick}, amount_tick={amount_tick}, side='{side}', size_base={size_base}, bid={bid}, ask={ask}, cross_pct={cross_pct}")

    try:
        if bid is None and ask is None:
            logger.error("Lighter close failed: both bid and ask are None.")
            return False

        # Calculate mid price for logging
        mid = (bid + ask) / 2.0 if (bid and ask) else (bid or ask)

        # Use a large size so reduce-only will close whatever position exists
        base_scaled = int(round(size_base / amount_tick))

        # For a closing order, we want to cross from mid price to get filled
        px = cross_price(
            side,
            ref_bid=bid,
            ref_ask=ask,
            tick=price_tick,
            cross_pct=cross_pct
        )
        price_scaled = int(px / price_tick)

        logger.info(f"Calculated values: base_scaled={base_scaled}, aggressive_price={px}, price_scaled={price_scaled}, mid={mid:.2f}, cross_pct={cross_pct:.1f}%")

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
        logger.info(f"Calling signer.create_order with params: {order_params}")

        tx, tx_hash, err = await signer.create_order(**order_params)

        logger.info(f"signer.create_order response: tx={tx}, tx_hash={tx_hash}, err={err}")

        if err:
            logger.error(f"Lighter close error: {err}")
            return False

        logger.info(f"Lighter close order placed: tx_hash={getattr(tx_hash, 'tx_hash', tx_hash)}")
        return True

    except Exception as e:
        logger.error(f"Lighter: Failed to close position: {e}", exc_info=True)
        return False


async def cancel_all_lighter_orders(env: dict) -> bool:
    """
    Cancel all open orders on Lighter.

    Args:
        env: Environment dict with Lighter credentials

    Returns:
        True if successful, False otherwise
    """
    import lighter

    base_url = env.get("LIGHTER_BASE_URL") or env.get("BASE_URL") or "https://mainnet.zklighter.elliot.ai"
    private_key = env.get("LIGHTER_PRIVATE_KEY") or env.get("API_KEY_PRIVATE_KEY")
    account_index = int(env.get("LIGHTER_ACCOUNT_INDEX") or env.get("ACCOUNT_INDEX") or "0")
    api_key_index = int(env.get("LIGHTER_API_KEY_INDEX") or env.get("API_KEY_INDEX") or "0")

    try:
        logger.info("Lighter: Canceling all open orders...")

        # Create signer client
        client = lighter.SignerClient(
            url=base_url,
            private_key=private_key,
            api_key_index=api_key_index,
            account_index=account_index
        )

        # Cancel all orders (using time=0 as per market_maker_v2.py)
        tx, tx_hash, err = await client.cancel_all_orders(
            time_in_force=client.CANCEL_ALL_TIF_IMMEDIATE,
            time=0
        )

        if err is not None:
            logger.error(f"Lighter: Error canceling all orders: {err}")
            return False

        logger.info(f"Lighter: Successfully canceled all orders: tx_hash={getattr(tx_hash, 'tx_hash', tx_hash) if tx_hash else 'OK'}")

        # Verify by checking account
        api_client = lighter.ApiClient(lighter.Configuration(host=base_url))
        account_api = lighter.AccountApi(api_client)

        account_details = await account_api.account(by="index", value=str(account_index))

        total_open_orders = 0
        if account_details.accounts:
            acc = account_details.accounts[0]
            total_open_orders = sum(pos.open_order_count for pos in acc.positions if pos.open_order_count > 0)

            if total_open_orders > 0:
                logger.warning(f"Lighter: {total_open_orders} orders still exist after cancel:")
                for pos in acc.positions:
                    if pos.open_order_count > 0:
                        logger.warning(f"  {pos.symbol}: {pos.open_order_count} open orders")

        await api_client.close()

        if total_open_orders == 0:
            logger.info("Lighter: All orders canceled successfully (verified)")
            return True
        else:
            logger.warning(f"Lighter: {total_open_orders} orders remain after cancel")
            return False

    except Exception as e:
        logger.error(f"Lighter: Failed to cancel all orders: {e}", exc_info=True)
        return False
