"""
edgex_connector.py
------------------
EdgeX exchange connector - handles all EdgeX-specific operations.
Extracted from hedge_cli.py and auto_rotation_bot.py for better modularity.
"""

import asyncio
import logging
from typing import Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from edgex_sdk import Client as EdgeXClient, OrderSide as EdgeXSide, OrderType as EdgeXType, TimeInForce as EdgeXTIF, CreateOrderParams

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
    Get EdgeX total and available USD balance.

    Args:
        env: Environment dict with EdgeX credentials

    Returns:
        Tuple of (total_usd, available_usd)
    """
    edgex = None
    try:
        edgex = EdgeXClient(
            base_url=env["EDGEX_BASE_URL"],
            account_id=int(env["EDGEX_ACCOUNT_ID"]),
            stark_private_key=env["EDGEX_STARK_PRIVATE_KEY"]
        )

        # Get account asset info
        resp = await edgex.get_account_asset()
        logger.debug(f"EdgeX account asset response: {resp}")

        data = resp.get("data", {})
        total = 0.0
        avail = 0.0

        # Try collateralAssetModelList first (most reliable)
        asset_list = data.get("collateralAssetModelList", [])
        if asset_list:
            for a in asset_list:
                if a.get("coinId") == "1000":  # USD
                    # Use totalEquity for total (includes positions), availableAmount for available
                    if "totalEquity" in a:
                        total += float(a.get("totalEquity", "0"))
                    else:
                        total += float(a.get("amount", "0"))
                    avail += float(a.get("availableAmount", "0"))
            logger.debug(f"EdgeX balance from collateralAssetModelList: total={total}, available={avail}")
            return total, avail

        # Fallback: try account.totalWalletBalance
        account = data.get("account", {})
        if "totalWalletBalance" in account:
            total = float(account.get("totalWalletBalance") or 0.0)
            logger.debug(f"EdgeX balance from totalWalletBalance: total={total}, available={total}")
            return total, total

        # Fallback: try collateralList
        coll_list = data.get("collateralList", [])
        for b in coll_list:
            if b.get("coinId") == "1000":
                total = float(b.get("amount", "0"))
                logger.debug(f"EdgeX balance from collateralList: total={total}, available={total}")
                return total, total

        logger.warning(f"EdgeX balance not found in any known fields. Response data keys: {list(data.keys())}")
        return 0.0, 0.0

    except Exception as e:
        logger.error(f"Error fetching EdgeX balance: {e}", exc_info=True)
        raise
    finally:
        if edgex:
            await edgex.close()


# ==================== Contract Discovery ====================

async def find_contract_id(client: EdgeXClient, contract_name: str) -> Tuple[str, float, float]:
    """
    Find contract ID, tick price, and tick size for a contract name.

    Args:
        client: Authenticated EdgeX client
        contract_name: Contract name (e.g., "PAXGUSD")

    Returns:
        Tuple of (contract_id, tick_price, tick_size)
    """
    metadata = await client.get_metadata()
    data = metadata.get("data", {})
    contract_list = data.get("contractList", [])
    for c in contract_list:
        if c.get("contractName") == contract_name:
            return (
                c["contractId"],
                float(c.get("tickPrice", "0.01")),
                float(c.get("stepSize", "0.01"))
            )
    raise ValueError(f"Contract {contract_name} not found on EdgeX")


# ==================== Price Functions ====================

async def get_best_bid_ask(client: EdgeXClient, contract_id: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Get best bid and ask for a contract.

    Args:
        client: Authenticated EdgeX client
        contract_id: Contract ID

    Returns:
        Tuple of (best_bid, best_ask) or (None, None) if unavailable
    """
    try:
        quote_resp = await client.quote.get_24_hour_quote(contract_id)
        if quote_resp.get("code") == "SUCCESS":
            data_list = quote_resp.get("data", [])
            if isinstance(data_list, list) and len(data_list) > 0:
                d = data_list[0]
                bid_str = d.get("bestBid")
                ask_str = d.get("bestAsk")
                bid = float(bid_str) if bid_str else None
                ask = float(ask_str) if ask_str else None
                return bid, ask
    except Exception as e:
        logger.error(f"Error fetching EdgeX bid/ask: {e}")
    return None, None


async def get_mark_price(client: EdgeXClient, contract_id: str) -> Optional[float]:
    """
    Get current mark price for a contract.

    Args:
        client: Authenticated EdgeX client
        contract_id: Contract ID

    Returns:
        Mark price or None if unavailable
    """
    try:
        quote_resp = await client.quote.get_24_hour_quote(contract_id)
        if quote_resp.get("code") == "SUCCESS" and quote_resp.get("data"):
            quote_data = quote_resp["data"][0]
            # Prefer last price, fall back to mark price, then mid of bid/ask
            if quote_data.get("lastPrice"):
                return float(quote_data["lastPrice"])
            elif quote_data.get("markPrice"):
                return float(quote_data["markPrice"])
            elif quote_data.get("bestBid") and quote_data.get("bestAsk"):
                return (float(quote_data["bestBid"]) + float(quote_data["bestAsk"])) / 2.0
    except Exception as e:
        logger.error(f"Error fetching EdgeX mark price: {e}")
    return None


# ==================== Leverage Management ====================

async def set_leverage(client: EdgeXClient, account_id: str, contract_id: str, leverage: float) -> None:
    """
    Set leverage for a contract (internal authenticated endpoint).

    Args:
        client: Authenticated EdgeX client
        account_id: Account ID
        contract_id: Contract ID
        leverage: Target leverage (e.g., 3.0)
    """
    logger.debug(f"Setting EdgeX leverage to {leverage}x for contract {contract_id}")
    await client.update_leverage(account_id=account_id, contract_id=contract_id, leverage=int(leverage))


async def get_leverage(client: EdgeXClient, contract_id: str) -> Optional[float]:
    """
    Get current leverage for a contract from positions API.

    Args:
        client: Authenticated EdgeX client
        contract_id: Contract ID

    Returns:
        Current leverage or None if no position exists
    """
    try:
        positions_resp = await client.get_account_positions()
        positions = positions_resp.get("data", {}).get("positionList", [])
        for p in positions:
            if p.get("contractId") == contract_id:
                lev = p.get("leverage")
                if lev:
                    return float(lev)
    except Exception as e:
        logger.error(f"Error getting EdgeX leverage: {e}")
    return None


# ==================== Position Management ====================

async def get_open_size(client: EdgeXClient, contract_id: str) -> float:
    """
    Get open position size for a contract (signed: positive=long, negative=short).

    Args:
        client: Authenticated EdgeX client
        contract_id: Contract ID

    Returns:
        Position size (0.0 if no position)
    """
    try:
        positions_resp = await client.get_account_positions()
        positions = positions_resp.get("data", {}).get("positionList", [])
        for p in positions:
            if p.get("contractId") == contract_id:
                size = float(p.get("openSize", "0"))
                side = p.get("side") or p.get("positionSide")
                if side and str(side).lower().startswith("short"):
                    size = -abs(size)
                return size
    except Exception as e:
        logger.error(f"Error getting EdgeX position size: {e}")
    return 0.0


async def get_position_details(client: EdgeXClient, contract_id: str, current_price: float) -> dict:
    """
    Get detailed position information.

    Args:
        client: Authenticated EdgeX client
        contract_id: Contract ID
        current_price: Current market price for PnL calculation

    Returns:
        Dict with position details (size, side, entry_price, unrealized_pnl, etc.)
    """
    try:
        positions_resp = await client.get_account_positions()
        positions = positions_resp.get("data", {}).get("positionList", [])
        for pos in positions:
            if pos.get("contractId") == contract_id:
                size = float(pos.get("openSize", "0"))
                side = pos.get("side") or pos.get("positionSide")
                is_short = side and str(side).lower().startswith("short")

                if is_short:
                    size = -abs(size)

                open_value = float(pos.get("openValue", "0"))
                entry_price = abs(open_value / size) if size != 0 else 0.0

                # Calculate unrealized PnL
                unrealized_pnl = 0.0
                if size != 0 and current_price > 0:
                    current_value = current_price * size
                    unrealized_pnl = current_value - open_value

                return {
                    "size": size,
                    "side": "short" if is_short else "long",
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "unrealized_pnl": unrealized_pnl,
                    "open_value": open_value,
                    "leverage": float(pos.get("leverage", "0")),
                    "margin_ratio": float(pos.get("marginRatio", "0"))
                }
    except Exception as e:
        logger.error(f"Error getting EdgeX position details: {e}")

    return {"size": 0.0, "side": "none", "entry_price": 0.0, "current_price": current_price,
            "unrealized_pnl": 0.0, "open_value": 0.0, "leverage": 0.0, "margin_ratio": 0.0}


async def get_position_pnl(client: EdgeXClient, contract_id: str) -> float:
    """
    Get unrealized PnL for a position (calculated manually).

    Args:
        client: Authenticated EdgeX client
        contract_id: Contract ID

    Returns:
        Unrealized PnL in USD
    """
    try:
        positions_resp = await client.get_account_positions()
        positions = positions_resp.get("data", {}).get("positionList", [])

        for pos in positions:
            if pos.get("contractId") == contract_id:
                # Get current market price using quote API
                quote_resp = await client.quote.get_24_hour_quote(contract_id)
                data_list = quote_resp.get("data", [])

                current_price = 0.0
                if isinstance(data_list, list) and data_list:
                    d = data_list[0]
                    # Prefer last price, fall back to mark price, then mid of bid/ask
                    if d.get("lastPrice"):
                        current_price = float(d["lastPrice"])
                    elif d.get("markPrice"):
                        current_price = float(d["markPrice"])
                    elif d.get("bestBid") and d.get("bestAsk"):
                        current_price = (float(d["bestBid"]) + float(d["bestAsk"])) / 2.0

                # Calculate PnL manually
                size = float(pos.get("openSize", "0"))
                side = pos.get("side") or pos.get("positionSide")
                if side and str(side).lower().startswith("short"):
                    size = -abs(size)

                open_value = float(pos.get("openValue", "0"))
                if abs(open_value) > 0 and size != 0 and current_price > 0:
                    current_value = current_price * size
                    upnl = current_value - open_value
                    return upnl
    except Exception as e:
        logger.error(f"Error calculating EdgeX PnL: {e}")

    return 0.0


# ==================== Order Execution ====================

async def place_aggressive_order(
    client: EdgeXClient,
    contract_id: str,
    side: str,
    size: float,
    tick_price: float,
    tick_size: float,
    ref_bid: float,
    ref_ask: float,
    cross_ticks: int = 10
) -> Optional[str]:
    """
    Place aggressive limit order (crosses spread for immediate fill).

    Args:
        client: Authenticated EdgeX client
        contract_id: Contract ID
        side: "buy" or "sell"
        size: Order size in base units
        tick_price: Price tick size
        tick_size: Size tick size
        ref_bid: Current best bid
        ref_ask: Current best ask
        cross_ticks: Number of ticks to cross spread

    Returns:
        Order ID or None if failed
    """
    try:
        price = cross_price(side, ref_bid, ref_ask, tick_price, cross_ticks)
        size_rounded = _round_to_tick(size, tick_size)

        logger.debug(f"EdgeX placing {side} order: size={size_rounded}, price={price}")

        params = CreateOrderParams(
            contract_id=contract_id,
            side=EdgeXSide.BUY if side.lower() == "buy" else EdgeXSide.SELL,
            size=str(size_rounded),
            price=str(price),
            order_type=EdgeXType.LIMIT,
            time_in_force=EdgeXTIF.IOC,
            reduce_only=False,
            post_only=False
        )
        resp = await client.create_order(params)

        if resp.get("code") == "SUCCESS":
            order_id = resp.get("data", {}).get("orderId")
            logger.info(f"EdgeX order placed: {order_id}")
            return order_id
        else:
            logger.error(f"EdgeX order failed: {resp}")
            return None
    except Exception as e:
        logger.error(f"Error placing EdgeX order: {e}", exc_info=True)
        return None


async def close_position(client: EdgeXClient, contract_id: str, tick_size: float, step_size: float, cross_ticks: int = 10) -> Optional[str]:
    """
    Close existing position with aggressive offsetting order.

    Args:
        client: Authenticated EdgeX client
        contract_id: Contract ID
        tick_size: Price tick size
        step_size: Size tick size
        cross_ticks: Number of ticks to cross spread

    Returns:
        Order ID or None if failed
    """
    try:
        # Get current position
        size = await get_open_size(client, contract_id)
        if abs(size) < 1e-8:
            logger.info("EdgeX: no position to close")
            return None

        # Determine offsetting side
        side = "sell" if size > 0 else "buy"
        abs_size = abs(size)

        # Get current bid/ask
        bid, ask = await get_best_bid_ask(client, contract_id)
        if bid is None:
            bid = ask if ask else 0.0
        if ask is None:
            ask = bid if bid else 0.0

        # Place offsetting order
        logger.info(f"EdgeX closing position: {side} {abs_size}")
        return await place_aggressive_order(
            client=client,
            contract_id=contract_id,
            side=side,
            size=abs_size,
            tick_price=tick_size,
            tick_size=step_size,
            ref_bid=bid,
            ref_ask=ask,
            cross_ticks=cross_ticks
        )
    except Exception as e:
        logger.error(f"Error closing EdgeX position: {e}", exc_info=True)
        return None


# ==================== Funding Rate Info ====================

async def get_funding_info(client: EdgeXClient, contract_id: str, base_url: str, symbol: str, quote: str) -> Optional[float]:
    """
    Get funding rate for a contract (annualized %).

    Args:
        client: Authenticated EdgeX client
        contract_id: Contract ID
        base_url: EdgeX base URL
        symbol: Base symbol (e.g., "PAXG")
        quote: Quote currency (e.g., "USD")

    Returns:
        Funding rate as annualized percentage or None
    """
    try:
        # Try quote API first
        quote_resp = await client.quote.get_24_hour_quote(contract_id)
        if quote_resp.get("code") == "SUCCESS" and quote_resp.get("data"):
            data_list = quote_resp["data"]
            if isinstance(data_list, list) and data_list:
                d = data_list[0]
                funding_rate = d.get("fundingRate")
                if funding_rate is not None:
                    rate_decimal = float(funding_rate)
                    # EdgeX funding is 8-hourly, annualize: (rate * 3 payments/day * 365 days)
                    annualized_pct = rate_decimal * 3 * 365 * 100
                    logger.debug(f"EdgeX funding: {annualized_pct:.2f}% APR")
                    return annualized_pct

        # Fallback: try historical endpoint
        # Note: This requires additional API implementation
        logger.warning("EdgeX funding rate not available from quote API")
        return None

    except Exception as e:
        logger.error(f"Error getting EdgeX funding: {e}")
        return None
