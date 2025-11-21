#!/usr/bin/env python3
"""
edgex_client.py
----------------
EdgeX exchange connector providing balance, market, position, and order helpers.

The interface mirrors the helpers offered by `lighter_client.py` so higher-level
modules can interact with both exchanges in a symmetric way.
"""

import logging
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, ROUND_UP
from typing import Dict, List, Optional, Tuple

from edgex_sdk import (
    Client as EdgeXClient,
    CreateOrderParams,
    OrderSide as EdgeXSide,
    OrderType as EdgeXType,
    TimeInForce as EdgeXTIF,
)

logger = logging.getLogger(__name__)


# ==================== Rounding Helpers ====================

def _round_to_tick(value: float, tick: float) -> float:
    """Round `value` to the nearest multiple of `tick`."""
    if not tick or tick <= 0:
        return value
    d_value = Decimal(str(value))
    d_tick = Decimal(str(tick))
    return float((d_value / d_tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * d_tick)


def _ceil_to_tick(value: float, tick: float) -> float:
    """Round `value` up to the nearest multiple of `tick`."""
    if not tick or tick <= 0:
        return value
    d_value = Decimal(str(value))
    d_tick = Decimal(str(tick))
    return float((d_value / d_tick).quantize(Decimal("1"), rounding=ROUND_UP) * d_tick)


def _floor_to_tick(value: float, tick: float) -> float:
    """Round `value` down to the nearest multiple of `tick`."""
    if not tick or tick <= 0:
        return value
    d_value = Decimal(str(value))
    d_tick = Decimal(str(tick))
    return float((d_value / d_tick).quantize(Decimal("1"), rounding=ROUND_DOWN) * d_tick)


def cross_price(
    side: str,
    ref_bid: Optional[float],
    ref_ask: Optional[float],
    tick: float,
    cross_pct: float = 3.0,  # Percentage from mid price (default 3%, well under 5% exchange limit)
) -> float:
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
        cross_pct: Percentage to cross from mid price (default 5%)
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


# ==================== Market Metadata ====================

async def get_edgex_contract_details(client: EdgeXClient, contract_name: str) -> Tuple[str, float, float]:
    """
    Return `(contract_id, tick_size, step_size)` for an EdgeX contract.

    Args:
        client: Authenticated EdgeX client.
        contract_name: Symbol+quote (e.g., `PAXGUSD`).
    """
    metadata = await client.get_metadata()
    contracts = metadata.get("data", {}).get("contractList", [])
    for contract in contracts:
        if contract.get("contractName") == contract_name:
            contract_id = contract.get("contractId")
            tick_size = float(contract.get("tickSize", "0.01"))
            step_size = float(contract.get("stepSize", "0.001"))
            return contract_id, tick_size, step_size
    raise RuntimeError(f"EdgeX: contract {contract_name} not found.")


async def get_edgex_best_bid_ask(client: EdgeXClient, contract_id: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetch best bid/ask for a contract. Falls back to synthetic prices around last trade.
    """
    quote = await client.quote.get_24_hour_quote(contract_id)
    logger.debug("EdgeX quote response: %s", quote)
    data_list = quote.get("data", [])
    if isinstance(data_list, list) and data_list:
        record = data_list[0]
        bid = float(record.get("bestBid")) if record.get("bestBid") else None
        ask = float(record.get("bestAsk")) if record.get("bestAsk") else None

        if bid:
            logger.info("EdgeX best bid: %s", bid)
        if ask:
            logger.info("EdgeX best ask: %s", ask)

        if bid and ask:
            return bid, ask

        last_price_str = record.get("lastPrice")
        if last_price_str:
            last_price = float(last_price_str)
            synthetic_bid = last_price * 0.9995
            synthetic_ask = last_price * 1.0005
            logger.info(
                "EdgeX missing bid/ask for %s; using synthetic spread around %s",
                contract_id,
                last_price,
            )
            return synthetic_bid, synthetic_ask

    logger.warning("EdgeX: No price data available for contract %s", contract_id)
    return None, None


# ==================== Leverage Helpers ====================

async def set_edgex_leverage(client: EdgeXClient, account_id: str, contract_id: str, leverage: float) -> None:
    """
    Update leverage setting for a contract via the private leverage endpoint.
    """
    path = "/api/v1/private/account/updateLeverageSetting"
    data = {"accountId": str(account_id), "contractId": contract_id, "leverage": str(leverage)}
    resp = await client.internal_client.make_authenticated_request(
        method="POST",
        path=path,
        data=data,
    )
    if resp.get("code") != "SUCCESS":
        raise RuntimeError(f"EdgeX: leverage update failed: {resp.get('msg', 'Unknown error')}")


async def get_edgex_leverage(client: EdgeXClient, contract_id: str) -> Optional[float]:
    """
    Return the current leverage configured for a contract if a position exists.
    """
    try:
        positions_response = await client.get_account_positions()
        positions = positions_response.get("data", {}).get("positionList", [])
        for position in positions:
            if position.get("contractId") == contract_id:
                leverage = position.get("leverage")
                if leverage:
                    return float(leverage)
        return None
    except Exception as exc:
        logger.debug("Could not get EdgeX leverage: %s", exc)
        return None


# ==================== Balances & Positions ====================

async def get_edgex_balance(client: EdgeXClient) -> Tuple[float, float]:
    """
    Return `(total_usd, available_usd)` using `get_account_asset`.
    """
    try:
        resp = await client.get_account_asset()
        data = resp.get("data", {})
        total = 0.0
        available = 0.0

        asset_list = data.get("collateralAssetModelList", [])
        if asset_list:
            for asset in asset_list:
                if asset.get("coinId") == "1000":
                    if "totalEquity" in asset:
                        total += float(asset.get("totalEquity", "0"))
                    else:
                        total += float(asset.get("amount", "0"))
                    available += float(asset.get("availableAmount", "0"))
            return total, available

        account = data.get("account", {})
        if "totalWalletBalance" in account:
            total = float(account.get("totalWalletBalance") or 0.0)
            return total, total

        collateral_list = data.get("collateralList", [])
        for entry in collateral_list:
            if entry.get("coinId") == "1000":
                total = float(entry.get("amount", "0"))
                return total, total

        return 0.0, 0.0
    except Exception as exc:
        logger.error("Error fetching EdgeX balance: %s", exc, exc_info=True)
        raise


async def get_edgex_open_size(client: EdgeXClient, contract_id: str) -> float:
    """
    Return the signed open size for a contract (positive=long, negative=short).
    """
    try:
        positions_resp = await client.get_account_positions()
        positions = positions_resp.get("data", {}).get("positionList", [])
        for position in positions:
            if position.get("contractId") != contract_id:
                continue
            size = float(position.get("openSize", "0") or 0.0)
            side = position.get("side") or position.get("positionSide")
            if side and str(side).lower().startswith("short"):
                size = -abs(size)
            return size
        return 0.0
    except Exception as exc:
        logger.error("EdgeX: Failed to fetch open size: %s", exc, exc_info=True)
        raise RuntimeError(f"EdgeX: Failed to fetch open size: {exc}") from exc


async def get_edgex_position_details(
    client: EdgeXClient,
    contract_id: str,
    current_price: float,
) -> Dict[str, float]:
    """
    Return a dict with size, unrealized PnL, entry price, and leverage for a contract.
    """
    details = {"size": 0.0, "unrealized_pnl": 0.0, "entry_price": 0.0, "leverage": 0.0}
    try:
        positions_resp = await client.get_account_positions()
        positions = positions_resp.get("data", {}).get("positionList", [])
        for position in positions:
            if position.get("contractId") != contract_id:
                continue

            size = float(position.get("openSize", "0") or 0.0)
            side = position.get("side") or position.get("positionSide")
            if side and str(side).lower().startswith("short"):
                size = -abs(size)

            open_value = float(position.get("openValue", "0") or 0.0)
            leverage = float(position.get("leverage", "0") or 0.0)

            entry_price = 0.0
            unrealized = 0.0
            if abs(open_value) > 0 and size != 0:
                entry_price = abs(open_value) / abs(size)
                current_value = current_price * size
                unrealized = current_value - open_value

            details.update(
                {
                    "size": size,
                    "unrealized_pnl": unrealized,
                    "entry_price": entry_price,
                    "leverage": leverage,
                }
            )
            break
    except Exception as exc:
        logger.error("EdgeX: Failed to get position details: %s", exc, exc_info=True)
        raise RuntimeError(f"EdgeX: Failed to get position details: {exc}") from exc
    return details


async def get_edgex_funding_rate(client: EdgeXClient, contract_id: str) -> Optional[float]:
    """
    Fetch the latest funding rate (percentage per period) for a contract.
    """
    try:
        quote = await client.quote.get_24_hour_quote(contract_id)
        if quote.get("code") == "SUCCESS" and quote.get("data"):
            record = quote["data"][0]
            rate = record.get("fundingRate")
            return float(rate) if rate is not None else None
    except Exception as exc:
        logger.warning("Could not fetch EdgeX funding rate for %s: %s", contract_id, exc)
    return None


async def get_edgex_funding_payments(
    client: EdgeXClient,
    contract_id: str,
    start_time_ms: Optional[int] = None,
) -> Tuple[float, int, Optional[int]]:
    """
    Return `(total_funding_usd, payment_count, newest_timestamp_ms)` for funding payments.
    """
    params: Dict[str, str] = {"contractId": contract_id}
    if start_time_ms:
        params["startTime"] = str(start_time_ms)

    resp = await client.get_funding_records(params)
    if resp.get("code") != "SUCCESS":
        raise RuntimeError(f"EdgeX funding history failed: {resp.get('msg', 'Unknown error')}")

    data_list = resp.get("data", {}).get("dataList", [])
    total_funding = 0.0
    count = 0
    newest_ts: Optional[int] = None

    for entry in data_list:
        amount = float(entry.get("amount", "0") or 0.0)
        total_funding += amount
        count += 1
        funding_time = entry.get("fundingTime")
        if funding_time:
            ts = int(funding_time)
            if newest_ts is None or ts > newest_ts:
                newest_ts = ts

    return total_funding, count, newest_ts


# ==================== Order Helpers ====================

async def place_aggressive_order(
    client: EdgeXClient,
    contract_id: str,
    tick_size: float,
    step_size: float,
    side: str,
    size_base: float,
    ref_price: float,
    cross_pct: float = 3.0,  # Percentage from mid price (default 3%, well under 5% exchange limit)
) -> Dict:
    """
    Place an aggressive limit order that crosses the spread to emulate a market order.
    Uses percentage-based pricing from mid price for consistent execution across all assets.
    """
    # Get current bid/ask for mid price calculation
    bid, ask = await get_edgex_best_bid_ask(client, contract_id)

    # Calculate mid for logging
    mid = (bid + ask) / 2.0 if (bid and ask) else (bid or ask or ref_price)

    price = cross_price(
        side,
        bid,
        ask,
        tick=tick_size,
        cross_pct=cross_pct,
    )
    size = _round_to_tick(size_base, step_size)

    params = CreateOrderParams(
        contract_id=str(contract_id),  # Convert to string for EdgeX SDK
        size=str(size),
        price=f"{price:.10f}",
        type=EdgeXType.LIMIT,
        side=EdgeXSide.BUY if side == "buy" else EdgeXSide.SELL,
        time_in_force=EdgeXTIF.GOOD_TIL_CANCEL,
    )
    metadata = await client.get_metadata()
    logger.info(
        "EdgeX order: side=%s size=%s price=%s (mid=%.2f cross_pct=%.1f%%)",
        side,
        size,
        price,
        mid,
        cross_pct,
    )
    return await client.order.create_order(params, metadata.get("data", {}))


async def close_position(
    client: EdgeXClient,
    contract_id: str,
    tick_size: float,
    step_size: float,
    cross_pct: float = 3.0,  # Percentage from mid price (default 3%, well under 5% exchange limit)
) -> Optional[str]:
    """
    Close an open EdgeX position by placing an offsetting aggressive order.
    Returns the order ID if one was submitted, otherwise None.
    """
    size = await get_edgex_open_size(client, contract_id)
    if abs(size) < 1e-12:
        logger.info("EdgeX: already flat for %s", contract_id)
        return None

    bid, ask = await get_edgex_best_bid_ask(client, contract_id)
    if not bid and not ask:
        raise RuntimeError("EdgeX: no market data to close position.")

    if size > 0:
        ref_price = bid if bid else ask
        side = "sell"
    else:
        ref_price = ask if ask else bid
        side = "buy"

    resp = await place_aggressive_order(
        client,
        contract_id,
        tick_size,
        step_size,
        side,
        abs(size),
        ref_price,
        cross_pct=cross_pct,
    )
    if resp.get("code") == "SUCCESS":
        return resp.get("data", {}).get("orderId")
    raise RuntimeError(f"EdgeX close failed: {resp.get('msg', 'Unknown error')}")
