#!/usr/bin/env python3
"""
hedge_cli.py
-------------
CLI to open a **hedged pair** (one LONG, one SHORT) across Lighter and EdgeX
at (nearly) the same time, plus a command to close both positions.

CHANGES (per request)
- Symbol is configured **once** (in JSON) and reused for both exchanges.
- All exchange credentials and URLs come from **.env** (dotenv).
- Optional `quote` in JSON (default "USD"); EdgeX contract name derived as SYMBOL+QUOTE.
- `open` supports --size-base or --size-quote (auto-converts quote→base using mids).
- `close` sends reduce-only closers on Lighter and an offsetting close on EdgeX.

ENV EXPECTATIONS (loaded via python-dotenv)
-------------------------------------------
EdgeX (from your market_maker.py):
  EDGEX_BASE_URL, EDGEX_WS_URL, EDGEX_ACCOUNT_ID, EDGEX_STARK_PRIVATE_KEY

Lighter (from your market_maker_v2.py):
  API_KEY_PRIVATE_KEY, ACCOUNT_INDEX, API_KEY_INDEX, MARGIN_MODE
  LIGHTER_BASE_URL or BASE_URL (fallback)
  LIGHTER_WS_URL  or WEBSOCKET_URL (fallback)

USAGE
-----
python hedge_cli.py open  --size-base 0.05 --config hedge_config.json
python hedge_cli.py open  --size-quote 100  --config hedge_config.json
python hedge_cli.py close --config hedge_config.json
"""
import asyncio
import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple
import time
from datetime import datetime

# --- Third-party SDKs ---
from edgex_sdk import Client as EdgeXClient
import lighter

# --- Local modules ---
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    load_env,
    _round_to_tick, _ceil_to_tick, _floor_to_tick,
    compute_base_size_from_quote as _compute_base_size_from_quote,
    get_avg_mid as _get_avg_mid,
)
import lighter_client
import edgex_client

CRYPTO_LIST = ["BTC", "ETH", "SOL", "BNB", "ASTER", "PAXG","DOGE","XRP","LINK","HYPE","XPL","TRUMP","LTC","PUMP","FARTCOIN"]

# ---------- Logging ----------
# Configure root logger first to capture all loggers (including websockets, asyncio, etc.)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Clear any existing handlers to avoid duplicates
root_logger.handlers.clear()

# File handler: captures DEBUG and above
log_file_handler = logging.FileHandler("hedge_cli.log")
log_file_handler.setLevel(logging.DEBUG)
log_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
root_logger.addHandler(log_file_handler)

# Console handler: only WARNING and above (to keep console clean)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
root_logger.addHandler(console_handler)

# Our logger
logger = logging.getLogger("hedge_cli")

# ---------- Data model for minimal JSON config ----------
@dataclass
class AppConfig:
    symbol: str                 # e.g., "PAXG"
    long_exchange: str          # "edgex" or "lighter"
    short_exchange: str         # "edgex" or "lighter"
    leverage: float             # leverage to set on both venues (best effort)
    quote: str = "USD"          # used to derive EdgeX contract, e.g., PAXG+USD -> PAXGUSD
    notional: float = 40.0      # Default notional size in quote currency (e.g., USD)

# Note: load_env(), _round_to_tick, _ceil_to_tick, _floor_to_tick imported from utils.py

# ---------- Aliases for client module functions ----------
# These provide backwards compatibility with local function names

# Lighter aliases (data fetching)
lighter_get_market_details = lighter_client.get_lighter_market_details
lighter_best_bid_ask = lighter_client.get_lighter_best_bid_ask
lighter_set_leverage = lighter_client.lighter_set_leverage
lighter_get_open_size = lighter_client.get_lighter_open_size
lighter_get_position_details = lighter_client.get_lighter_position_details
lighter_available_capital_ws = lighter_client.get_lighter_balance
LighterOrderBookFetcher = lighter_client.LighterOrderBookFetcher

# EdgeX aliases (data fetching)
edgex_find_contract_id = edgex_client.get_edgex_contract_details
edgex_best_bid_ask = edgex_client.get_edgex_best_bid_ask
edgex_set_leverage = edgex_client.set_edgex_leverage
edgex_get_leverage = edgex_client.get_edgex_leverage
edgex_get_open_size = edgex_client.get_edgex_open_size
edgex_get_position_details = edgex_client.get_edgex_position_details
edgex_available_usd = edgex_client.get_edgex_balance
edgex_get_funding_payments = edgex_client.get_edgex_funding_payments

# ---------- Crossing helpers ----------
# Note: This cross_price uses tick-based approach (different from lighter_client.cross_price
# which uses percentage-based approach). Keep local for backwards compatibility.
def cross_price(side: str, ref_bid: float, ref_ask: float, tick: float, cross_ticks: int = 10) -> float:
    """
    Return an aggressive price that *crosses the spread* by at least `cross_ticks`.
    - For BUY: price >= best_ask + cross_ticks*tick
    - For SELL: price <= best_bid - cross_ticks*tick
    Falls back to using the other side if bid/ask is missing.
    """
    cross_ticks = max(1, int(cross_ticks))
    if side == "buy":
        if ref_ask:
            return _ceil_to_tick(ref_ask + cross_ticks * tick, tick)
        elif ref_bid:
            # If only bid available, still step above it
            return _ceil_to_tick(ref_bid + cross_ticks * tick, tick)
    else:  # sell
        if ref_bid:
            return _floor_to_tick(ref_bid - cross_ticks * tick, tick)
        elif ref_ask:
            # If only ask available, still step below it
            return _floor_to_tick(ref_ask - cross_ticks * tick, tick)
    # As a last resort just return the reference rounded, though this shouldn't happen
    ref = ref_ask if side == "buy" else ref_bid
    return _ceil_to_tick(ref, tick) if side == "buy" else _floor_to_tick(ref, tick)

# ---------- Lighter Order Functions (using local tick-based cross_price) ----------
# Note: lighter_get_market_details, LighterOrderBookFetcher, lighter_best_bid_ask,
# and lighter_set_leverage are aliased from lighter_client module above.

async def lighter_place_aggressive_order(
    signer: lighter.SignerClient,
    market_id: int,
    price_tick: float,
    amount_tick: float,
    side: str,                  # "buy" or "sell"
    size_base: float,
    ref_price: float, cross_ticks: int = 10
) -> str:
    """Aggressive limit (crossing) to emulate a market order (reduce_only=False).
    Note: size_base should already be rounded to tick before calling this."""

    if ref_price is None:
        raise RuntimeError(f"Lighter: ref_price is None for {side} order")

    px = cross_price(side, ref_bid=None if side=='buy' else ref_price, ref_ask=ref_price if side=='buy' else None, tick=price_tick, cross_ticks=cross_ticks)

    # Size should already be rounded - just scale to integer units
    base_scaled = int(round(size_base / amount_tick))
    price_scaled = int(px / price_tick)

    logger.info(f"Lighter order: {side} {base_scaled} units @ {price_scaled} scaled ({px:.4f} actual)")

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

async def lighter_close_position(
    signer: lighter.SignerClient,
    market_id: int,
    price_tick: float,
    amount_tick: float,
    side: str,  # "buy" to close short, "sell" to close long
    size_base: float,
    ref_price: float,
    cross_ticks: int = 10
) -> None:
    """Close a position with a reduce-only aggressive limit order."""
    logger.info("--- lighter_close_position called ---")
    logger.info(f"Inputs: market_id={market_id}, price_tick={price_tick}, amount_tick={amount_tick}, side='{side}', size_base={size_base}, ref_price={ref_price}, cross_ticks={cross_ticks}")

    if ref_price is None:
        logger.error("Lighter close failed: ref_price is None.")
        raise RuntimeError(f"Lighter: ref_price is None for {side} reduce-only order")

    # Use a large size so reduce-only will close whatever position exists
    base_scaled = int(round(size_base / amount_tick))

    # For a closing order, we want to cross the spread to get filled.
    px = cross_price(side, ref_bid=None if side == 'buy' else ref_price, ref_ask=ref_price if side == 'buy' else None, tick=price_tick, cross_ticks=cross_ticks)
    price_scaled = int(px / price_tick)

    logger.info(f"Calculated values: base_scaled={base_scaled}, aggressive_price={px}, price_scaled={price_scaled}")

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
    logger.info(f"Calling signer.create_order with params: {json.dumps(order_params, indent=2)}")

    tx, tx_hash, err = await signer.create_order(**order_params)

    logger.info(f"signer.create_order response: tx={tx}, tx_hash={tx_hash}, err={err}")
    logger.info(f"Response types: tx type={type(tx)}, tx_hash type={type(tx_hash)}, err type={type(err)}")


    if err is not None:
        logger.warning(f"Lighter reduce-only limit {side.upper()} returned error: {err}")
        err_str = str(err).lower()
        if 'no open position' not in err_str and 'no position' not in err_str:
            raise RuntimeError(f"Lighter {side.upper()} reduce-only limit order failed: {err}")

    logger.info(f"Lighter reduce-only limit order sent successfully: tx={tx}, tx_hash={tx_hash}")
    logger.info("--- lighter_close_position finished ---")

# ---------- EdgeX (short/long) ----------
# Note: edgex_find_contract_id, edgex_best_bid_ask, edgex_set_leverage, edgex_get_leverage
# are aliased from edgex_client module above.

async def lighter_get_leverage(signer: lighter.SignerClient, market_id: int) -> Optional[float]:
    """Get current leverage setting for a market on Lighter.
    Note: Lighter doesn't expose current leverage via API easily, returns None."""
    # Lighter doesn't provide an easy way to query current leverage
    # It's set per-order via the update_leverage call
    return None

async def setup_leverage_both_exchanges(
    cfg: AppConfig,
    env: dict,
    edgex: EdgeXClient,
    signer: lighter.SignerClient,
    e_contract_id: str,
    l_market_id: int,
    verify: bool = True
) -> Tuple[bool, bool]:
    """
    Set leverage on both exchanges to match config.
    Returns (edgex_success, lighter_success).
    If verify=True, will check that leverage was set correctly.
    """
    edgex_success = False
    lighter_success = False

    print(f"\nSetting leverage to {cfg.leverage}x on both exchanges...")

    # Set Lighter leverage
    try:
        await lighter_set_leverage(signer, l_market_id, int(cfg.leverage), env["MARGIN_MODE"])
        lighter_success = True
        print(f"  ✓ Lighter: Set to {cfg.leverage}x ({env['MARGIN_MODE']} margin)")
    except Exception as e:
        print(f"  ✗ Lighter: Failed to set leverage - {e}")
        logger.error(f"Lighter leverage set failed: {e}")

    # Set EdgeX leverage
    try:
        await edgex_set_leverage(edgex, int(env["EDGEX_ACCOUNT_ID"]), e_contract_id, cfg.leverage)
        edgex_success = True
        print(f"  ✓ EdgeX: Set to {cfg.leverage}x")
    except Exception as e:
        print(f"  ✗ EdgeX: Failed to set leverage - {e}")
        logger.error(f"EdgeX leverage set failed: {e}")

    # Verify if requested
    if verify and edgex_success:
        try:
            current_edgex_lev = await edgex_get_leverage(edgex, e_contract_id)
            if current_edgex_lev:
                if abs(current_edgex_lev - cfg.leverage) < 0.1:
                    print(f"  ✓ EdgeX: Verified at {current_edgex_lev}x")
                else:
                    print(f"  ⚠ EdgeX: Set to {cfg.leverage}x but reads as {current_edgex_lev}x")
            else:
                print(f"  ⚠ EdgeX: Could not verify (no open position)")
        except Exception as e:
            logger.debug(f"Could not verify EdgeX leverage: {e}")

    # Lighter verification note
    if lighter_success and verify:
        print(f"  ℹ Lighter: Verification not available (will apply on next order)")

    if not (edgex_success and lighter_success):
        print(f"\n⚠️  WARNING: Leverage setting failed on one or more exchanges!")
        print(f"  This may result in unexpected margin usage.")
        return edgex_success, lighter_success

    print(f"✓ Leverage configured on both exchanges\n")
    return edgex_success, lighter_success

# Note: lighter_get_open_size, lighter_get_position_details, edgex_get_open_size,
# edgex_get_position_details are aliased from client modules above.

async def edgex_place_aggressive_order(
    client: EdgeXClient,
    contract_id: str,
    tick_size: float,
    step_size: float,
    side: str,                # "buy" or "sell"
    size_base: float,
    ref_price: float, cross_ticks: int = 10
) -> dict:
    """Aggressive LIMIT (crossing). Note: size_base should already be rounded to tick before calling this."""
    px = cross_price(side, ref_bid=ref_price if side=='sell' else None, ref_ask=ref_price if side=='buy' else None, tick=tick_size, cross_ticks=cross_ticks)
    # Size should already be rounded - use as-is
    size = size_base

    params = CreateOrderParams(
        contract_id=contract_id,
        size=str(size),
        price=f"{px:.10f}",
        type=EdgeXType.LIMIT,
        side=EdgeXSide.BUY if side == "buy" else EdgeXSide.SELL,
        time_in_force=EdgeXTIF.GOOD_TIL_CANCEL  # marketable (not POST_ONLY)
    )
    meta = await client.get_metadata()
    return await client.order.create_order(params, meta.get("data", {}))

async def edgex_close_position(client: EdgeXClient, contract_id: str, tick_size: float, step_size: float, cross_ticks: int = 10) -> Optional[str]:
    """Detect current size and send an offsetting aggressive order to close it."""
    size = await edgex_get_open_size(client, contract_id)
    if abs(size) < 1e-12:
        logger.info("EdgeX: already flat.")
        return None
    bid, ask = await edgex_best_bid_ask(client, contract_id)
    if not bid and not ask:
        raise RuntimeError("EdgeX: no market data to close.")
    if size > 0:
        ref_px = bid if bid else ask
        side = "sell"
        size_abs = abs(size)
    else:
        ref_px = ask if ask else bid
        side = "buy"
        size_abs = abs(size)
    resp = await edgex_place_aggressive_order(client, contract_id, tick_size, step_size, side, size_abs, ref_px, cross_ticks=cross_ticks)
    if resp.get("code") == "SUCCESS":
        return resp.get("data", {}).get("orderId")
    raise RuntimeError(f"EdgeX close failed: {resp.get('msg','Unknown error')}")

# ---------- Coordination / Sizing ----------
# Async wrappers for utils functions (used with await in existing code)
async def compute_base_size_from_quote(avg_mid: float, size_quote: float) -> float:
    """Async wrapper for utils.compute_base_size_from_quote."""
    return _compute_base_size_from_quote(avg_mid, size_quote)

async def get_avg_mid(l_bid: Optional[float], l_ask: Optional[float], e_bid: Optional[float], e_ask: Optional[float]) -> float:
    """Async wrapper for utils.get_avg_mid."""
    return _get_avg_mid(l_bid, l_ask, e_bid, e_ask)

# Note: edgex_available_usd and lighter_available_capital_ws are aliased from client modules above.

async def compute_max_delta_neutral_size(cfg: AppConfig, env: dict) -> dict:
    """
    Returns a dict with per-venue capital, mids, and the max base size you can place delta-neutral.
    Assumptions:
      - Initial margin ≈ notional/leverage
      - Use *available USD* at each venue
      - Safety and fee buffers applied
    """
    # Build clients
    api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
    order_api = lighter.OrderApi(api_client)
    # GitHub SDK signature: SignerClient(url, account_index, api_private_keys, api_key_index)
    signer = lighter.SignerClient(
        env["LIGHTER_BASE_URL"],
        int(env["ACCOUNT_INDEX"]),
        int(env["API_KEY_INDEX"]),
        env["API_KEY_PRIVATE_KEY"]
    )
    # EdgeX client
    edgex = EdgeXClient(
        base_url=env["EDGEX_BASE_URL"],
        account_id=int(env["EDGEX_ACCOUNT_ID"]) if env.get("EDGEX_ACCOUNT_ID") else 0,
        stark_private_key=env.get("EDGEX_STARK_PRIVATE_KEY", ""),
    )

    # Market details & prices
    l_market_id, l_price_tick, l_amount_tick = await lighter_get_market_details(order_api, cfg.symbol)
    l_bid, l_ask = await lighter_best_bid_ask(order_api, cfg.symbol, l_market_id)
    contract_name = f"{cfg.symbol.upper()}{cfg.quote.upper()}"
    e_contract_id, e_tick, e_step = await edgex_find_contract_id(edgex, contract_name)
    e_bid, e_ask = await edgex_best_bid_ask(edgex, e_contract_id)

    # Compute a common mid
    mid = await get_avg_mid(l_bid, l_ask, e_bid, e_ask)

    # Capital on EdgeX (REST)
    e_total, e_avail = await edgex_available_usd(edgex)

    # Capital on Lighter (prefer WS user_stats; if not available, fall back to 0)
    try:
        l_avail, l_portv = await lighter_available_capital_ws(env["LIGHTER_WS_URL"], env["ACCOUNT_INDEX"], timeout=8.0)
    except Exception:
        l_avail = 0.0
        l_portv = 0.0

    # Buffers
    safety_margin = 0.01    # 1% safety
    fee_buffer = 0.001      # 0.1% fees/slippage

    # Capacity per venue in base units = available_usd * (1 - buffers) * leverage / mid
    def venue_capacity(av_usd: float) -> float:
        usable = max(av_usd, 0.0) * (1.0 - safety_margin - fee_buffer)
        return (usable * float(cfg.leverage)) / mid if mid and mid > 0 else 0.0

    cap_lighter = venue_capacity(l_avail)
    cap_edgex = venue_capacity(e_avail)

    # Role-specific capacities
    long_cap = cap_lighter if cfg.long_exchange == "lighter" else cap_edgex
    short_cap = cap_edgex if cfg.short_exchange == "edgex" else cap_lighter

    # Max delta-neutral base size
    raw_max_base = max(min(long_cap, short_cap), 0.0)
    # Round conservatively by both step sizes
    max_base = max(_round_to_tick(raw_max_base, l_amount_tick), _round_to_tick(raw_max_base, e_step))

    result = {
        "symbol": cfg.symbol,
        "quote": cfg.quote,
        "mid_price": mid,
        "lighter_available_usd": l_avail,
        "edgex_available_usd": e_avail,
        "lighter_capacity_base": cap_lighter,
        "edgex_capacity_base": cap_edgex,
        "long_exchange": cfg.long_exchange,
        "short_exchange": cfg.short_exchange,
        "max_delta_neutral_base": max_base,
        "ticks": {"lighter_amount_tick": l_amount_tick, "edgex_step": e_step},
    }

    await signer.close()
    await api_client.close()
    await edgex.close()
    return result

# ---------- High-level ops ----------
async def open_hedge(cfg: AppConfig, env: dict, size_base: Optional[float], size_quote: Optional[float], cross_ticks: int = 10) -> None:
    # Lighter clients
    api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
    order_api = lighter.OrderApi(api_client)
    # GitHub SDK signature: SignerClient(url, account_index, api_private_keys, api_key_index)
    signer = lighter.SignerClient(
        env["LIGHTER_BASE_URL"],
        int(env["ACCOUNT_INDEX"]),
        int(env["API_KEY_INDEX"]),
        env["API_KEY_PRIVATE_KEY"]
    )
    err = signer.check_client()
    if err:
        raise RuntimeError(f"Lighter check_client error: {err}")

    # EdgeX client
    edgex = EdgeXClient(
        base_url=env["EDGEX_BASE_URL"],
        account_id=int(env["EDGEX_ACCOUNT_ID"]) if env.get("EDGEX_ACCOUNT_ID") else 0,
        stark_private_key=env.get("EDGEX_STARK_PRIVATE_KEY", ""),
    )

    # Market details & prices
    try:
        l_market_id, l_price_tick, l_amount_tick = await lighter_get_market_details(order_api, cfg.symbol)
        l_bid, l_ask = await lighter_best_bid_ask(order_api, cfg.symbol, l_market_id)

        contract_name = f"{cfg.symbol.upper()}{cfg.quote.upper()}"
        e_contract_id, e_tick, e_step = await edgex_find_contract_id(edgex, contract_name)
        e_bid, e_ask = await edgex_best_bid_ask(edgex, e_contract_id)

        if not any([l_bid, l_ask, e_bid, e_ask]):
            raise RuntimeError("Could not fetch quotes from either venue.")

        # Ensure we have at least one price from each exchange for order placement
        if not (l_bid or l_ask):
            raise RuntimeError(
                f"Could not fetch any prices from Lighter for {cfg.symbol}. "
                f"The order book may be empty or the market may be inactive. "
                f"Check hedge_cli.log for details. Try a more liquid market like BTC or ETH."
            )
        if not (e_bid or e_ask):
            raise RuntimeError(f"Could not fetch any prices from EdgeX for {contract_name}")
    except Exception as e:
        # Clean up clients before re-raising
        await signer.close()
        await api_client.close()
        await edgex.close()
        raise

    # Set leverage on both exchanges with verification
    edgex_lev_ok, lighter_lev_ok = await setup_leverage_both_exchanges(
        cfg, env, edgex, signer, e_contract_id, l_market_id, verify=True
    )

    if not (edgex_lev_ok and lighter_lev_ok):
        print("⚠️  Proceeding despite leverage setup issues. Monitor margin carefully.")
        # Don't abort - proceed with warning

    # Determine size in BASE
    if size_base is None:
        avg_mid = await get_avg_mid(l_bid, l_ask, e_bid, e_ask)
        size_base = await compute_base_size_from_quote(avg_mid, float(size_quote))
        print(f"Converting {size_quote} {cfg.quote} to {size_base:.6f} {cfg.symbol} at mid price ${avg_mid:.2f}")

    # Round to ensure SAME size on both exchanges
    # Use the coarser tick size (larger value) and round down to ensure both can handle it
    coarser_tick = max(l_amount_tick, e_step)
    size_base = _floor_to_tick(size_base, coarser_tick)

    # Verify both exchanges can handle this size after their individual rounding
    lighter_rounded = _round_to_tick(size_base, l_amount_tick)
    edgex_rounded = _round_to_tick(size_base, e_step)

    if abs(lighter_rounded - edgex_rounded) > min(l_amount_tick, e_step):
        # If rounded sizes differ significantly, floor to coarser tick again
        size_base = _floor_to_tick(size_base, coarser_tick)
        logger.warning(f"Adjusted size to {size_base} to ensure same size on both exchanges")

    if size_base <= 0:
        raise SystemExit("Computed size rounds to zero. Increase size.")

    # Get minimum order sizes from contract metadata
    # EdgeX minimum is typically in the stepSize field, but let's check contract metadata
    edgex_metadata = await edgex.get_metadata()
    edgex_contracts = edgex_metadata.get("data", {}).get("contractList", [])
    edgex_min_size = None
    for c in edgex_contracts:
        if c.get("contractId") == e_contract_id:
            edgex_min_size = float(c.get("minOrderSize", "0"))
            break

    # Check minimum order sizes
    avg_mid = await get_avg_mid(l_bid, l_ask, e_bid, e_ask)
    size_usd = size_base * avg_mid

    errors = []
    if edgex_min_size and size_base < edgex_min_size:
        edgex_min_usd = edgex_min_size * avg_mid
        errors.append(f"EdgeX minimum: {edgex_min_size} {cfg.symbol} (${edgex_min_usd:.2f} USD)")

    # Lighter minimum is typically the amount_tick, but let's be conservative
    lighter_min_size = l_amount_tick * 10  # Assume 10x the tick size as minimum
    if size_base < lighter_min_size:
        lighter_min_usd = lighter_min_size * avg_mid
        errors.append(f"Lighter estimated minimum: {lighter_min_size} {cfg.symbol} (${lighter_min_usd:.2f} USD)")

    if errors:
        print(f"\n❌ ERROR: Order size too small!")
        print(f"  Requested: {size_base} {cfg.symbol} (${size_usd:.2f} USD)")
        print(f"\n  Minimum requirements:")
        for err in errors:
            print(f"    • {err}")
        if edgex_min_size:
            recommended_usd = edgex_min_size * avg_mid * 1.1  # 10% buffer
            print(f"\n  💡 Recommendation: Use at least --notional {recommended_usd:.0f}")
        await signer.close()
        await api_client.close()
        await edgex.close()
        raise SystemExit(1)

    # Who is long / short (define before printing)
    long_leg = cfg.long_exchange.lower()
    short_leg = cfg.short_exchange.lower()
    if set([long_leg, short_leg]) != {"edgex", "lighter"}:
        if long_leg == short_leg:
            raise SystemExit("Config error: long_exchange and short_exchange cannot be the same.")
        raise SystemExit("Config error: exchanges must be 'edgex' and 'lighter' in some order.")

    print(f"\n┌{'─' * 66}┐")
    print(f"│{'Opening Delta-Neutral Hedge':^66}│")
    print(f"├{'─' * 66}┤")
    print(f"│  Symbol:           {cfg.symbol}/{cfg.quote:<4}                                      │")
    print(f"│  Size:             {size_base:.6f} {cfg.symbol:<4}                              │")
    print(f"│  Leverage:         {cfg.leverage}x                                             │")
    print(f"│                                                                  │")
    print(f"│  Tick Sizes:                                                     │")
    print(f"│    Lighter:        {l_amount_tick:.8f}                                │")
    print(f"│    EdgeX:          {e_step:.8f}                                │")
    print(f"│    Using coarser:  {coarser_tick:.8f}                                │")
    print(f"│                                                                  │")
    print(f"│  Long Position:    {long_leg.capitalize():<8}                                       │")
    print(f"│  Short Position:   {short_leg.capitalize():<8}                                       │")
    print(f"└{'─' * 66}┘\n")

    # Crossing refs
    l_ref_buy = l_ask if l_ask else l_bid
    l_ref_sell = l_bid if l_bid else l_ask
    e_ref_buy = e_ask if e_ask else e_bid
    e_ref_sell = e_bid if e_bid else e_ask

    # Verify final sizes that will be sent (for logging)
    lighter_final_size = size_base
    edgex_final_size = size_base
    lighter_scaled = int(round(lighter_final_size / l_amount_tick))
    edgex_scaled = int(round(edgex_final_size / e_step))

    logger.info(f"Final sizes - Lighter: {lighter_final_size} ({lighter_scaled} scaled units), EdgeX: {edgex_final_size}")

    # Place both legs concurrently
    tasks = []
    if long_leg == "lighter":
        tasks.append(lighter_place_aggressive_order(signer, l_market_id, l_price_tick, l_amount_tick, "buy", size_base, l_ref_buy, cross_ticks=cross_ticks))
        tasks.append(edgex_place_aggressive_order(edgex, e_contract_id, e_tick, e_step, "sell", size_base, e_ref_sell, cross_ticks=cross_ticks))
    else:
        tasks.append(lighter_place_aggressive_order(signer, l_market_id, l_price_tick, l_amount_tick, "sell", size_base, l_ref_sell, cross_ticks=cross_ticks))
        tasks.append(edgex_place_aggressive_order(edgex, e_contract_id, e_tick, e_step, "buy", size_base, e_ref_buy, cross_ticks=cross_ticks))

    print("Placing orders on both exchanges...")
    print(f"  Size per exchange: {size_base:.8f} {cfg.symbol}")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for errors
    errors = [(i, r) for i, r in enumerate(results) if isinstance(r, Exception)]
    if errors:
        print(f"\n❌ ERROR: One or more orders failed!")

        # Determine which legs succeeded
        leg_names = []
        if long_leg == "lighter":
            leg_names = ["Lighter (LONG)", "EdgeX (SHORT)"]
        else:
            leg_names = ["Lighter (SHORT)", "EdgeX (LONG)"]

        for i, err in errors:
            err_str = str(err)
            print(f"   {leg_names[i]}: {err_str}")

            # Check if it's a minimum size error
            if "minOrderSize" in err_str or "minimum" in err_str.lower():
                avg_mid = await get_avg_mid(l_bid, l_ask, e_bid, e_ask)
                size_usd = size_base * avg_mid
                print(f"   → Order size was {size_base} {cfg.symbol} (${size_usd:.2f} USD)")

                # Try to extract minimum from error
                if "minOrderSize" in err_str:
                    import re
                    match = re.search(r"'minOrderSize':\s*'([0-9.]+)'", err_str)
                    if match:
                        min_size = float(match.group(1))
                        min_usd = min_size * avg_mid
                        print(f"   → Minimum required: {min_size} {cfg.symbol} (${min_usd:.2f} USD)")
                        recommended = min_usd * 1.1
                        print(f"   💡 Try: --notional {recommended:.0f}")

        # Check which legs succeeded
        successful_indices = [i for i in range(len(results)) if not isinstance(results[i], Exception)]
        if successful_indices:
            print(f"\n⚠️  CRITICAL: Partial fill detected!")
            print(f"   Successfully opened on: {', '.join([leg_names[i] for i in successful_indices])}")
            print(f"   Failed on: {', '.join([leg_names[i] for i, _ in errors])}")
            print(f"\n   YOU HAVE AN UNHEDGED POSITION!")
            print(f"   Please immediately close the open position manually or run:")
            print(f"   python hedge_cli.py close")
        else:
            print(f"\n✓ Both orders failed - no positions opened.")

        await signer.close()
        await api_client.close()
        await edgex.close()
        raise SystemExit(1)

    print("✓ Both orders placed successfully")
    logger.info(f"Opened hedge: size_base={size_base} {cfg.symbol}. Legs placed concurrently.")

    # Give exchanges time to process
    await asyncio.sleep(2)

    # Verify positions
    print("\nVerifying positions...")
    edgex_size = await edgex_get_open_size(edgex, e_contract_id)
    print(f"  EdgeX position:  {edgex_size:+.6f} {cfg.symbol}")
    print(f"  Lighter position: (verification not available via API)")

    print(f"\n✓ Hedge opened successfully!")
    print(f"  Total exposure: {size_base:.6f} {cfg.symbol} on each exchange")
    print(f"  Delta-neutral: LONG {long_leg.capitalize()}, SHORT {short_leg.capitalize()}\n")

    await signer.close()
    await api_client.close()
    await edgex.close()

async def close_both(cfg: AppConfig, env: dict, cross_ticks: int = 10) -> None:
    api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
    order_api = lighter.OrderApi(api_client)
    account_api = lighter.AccountApi(api_client)
    # GitHub SDK signature: SignerClient(url, account_index, api_private_keys, api_key_index)
    signer = lighter.SignerClient(
        env["LIGHTER_BASE_URL"],
        int(env["ACCOUNT_INDEX"]),
        int(env["API_KEY_INDEX"]),
        env["API_KEY_PRIVATE_KEY"]
    )
    err = signer.check_client()
    if err:
        raise RuntimeError(f"Lighter check_client error: {err}")

    edgex = EdgeXClient(
        base_url=env["EDGEX_BASE_URL"],
        account_id=int(env["EDGEX_ACCOUNT_ID"]) if env.get("EDGEX_ACCOUNT_ID") else 0,
        stark_private_key=env.get("EDGEX_STARK_PRIVATE_KEY", ""),
    )

    l_market_id, l_price_tick, l_amount_tick = await lighter_get_market_details(order_api, cfg.symbol)
    l_bid, l_ask = await lighter_best_bid_ask(order_api, cfg.symbol, l_market_id)

    contract_name = f"{cfg.symbol.upper()}{cfg.quote.upper()}"
    e_contract_id, e_tick, e_step = await edgex_find_contract_id(edgex, contract_name)

    # Check current positions
    print(f"\n┌{'─' * 66}┐")
    print(f"│{'Closing Delta-Neutral Hedge':^66}│")
    print(f"├{'─' * 66}┤")
    print(f"│  Symbol: {cfg.symbol}/{cfg.quote:<4}                                              │")
    print(f"└{'─' * 66}┘\n")

    print("Checking current positions...")
    edgex_size = await edgex_get_open_size(edgex, e_contract_id)
    print(f"  EdgeX position:  {edgex_size:+.6f} {cfg.symbol}")

    lighter_size = await lighter_get_open_size(account_api, env["ACCOUNT_INDEX"], l_market_id)
    print(f"  Lighter position: {lighter_size:+.6f} {cfg.symbol}")

    print(f"\nClosing positions on both exchanges...")

    tasks = []

    # Lighter close task
    if abs(lighter_size) > l_amount_tick:  # Only close if position is larger than min tick
        lighter_close_side = "sell" if lighter_size > 0 else "buy"
        lighter_ref_price = l_bid if lighter_close_side == "sell" else l_ask
        if not lighter_ref_price:
            logger.warning("Lighter: No ref_price to close position, skipping.")
            print("  Lighter: No reference price available, cannot send close order.")
        else:
            tasks.append(
                lighter_close_position(signer, l_market_id, l_price_tick, l_amount_tick, lighter_close_side, abs(lighter_size), lighter_ref_price, cross_ticks=cross_ticks)
            )
    else:
        print("  Lighter: No position found to close.")
        logger.info("Lighter: Position is zero or negligible, skipping close order.")

    # EdgeX close task (it has its own internal size check)
    tasks.append(edgex_close_position(edgex, e_contract_id, e_tick, e_step, cross_ticks=cross_ticks))

    if not tasks:
        print("\n✓ No positions found on either exchange. Nothing to do.")
        await signer.close()
        await api_client.close()
        await edgex.close()
        return

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for errors
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"\n❌ ERROR: One or more close orders failed!")
        # This part is tricky because we don't know which task failed easily by index if one was skipped
        # For now, just print all errors.
        for err in errors:
            print(f"   - {err}")
        print(f"\n⚠️  WARNING: Please check both exchanges manually to verify positions were closed.")
        await signer.close()
        await api_client.close()
        await edgex.close()
        raise SystemExit(1)

    print("✓ Close orders sent to both exchanges")
    logger.info("Close command dispatched for both venues.")

    # Give exchanges time to process
    await asyncio.sleep(2)

    # Verify positions are closed
    print("\nVerifying closure...")
    edgex_size_after = await edgex_get_open_size(edgex, e_contract_id)
    print(f"  EdgeX position:  {edgex_size_after:+.6f} {cfg.symbol}")
    lighter_size_after = await lighter_get_open_size(account_api, env["ACCOUNT_INDEX"], l_market_id)
    print(f"  Lighter position: {lighter_size_after:+.6f} {cfg.symbol}")

    edgex_closed = abs(edgex_size_after) < e_step
    lighter_closed = abs(lighter_size_after) < l_amount_tick

    if edgex_closed and lighter_closed:
        print(f"\n✓ Hedge closed successfully on both exchanges!")
    else:
        print(f"\n⚠️  WARNING: One or more positions not fully closed.")
        if not edgex_closed:
            print(f"  EdgeX position remaining: {edgex_size_after:+.6f} {cfg.symbol}")
        if not lighter_closed:
            print(f"  Lighter position remaining: {lighter_size_after:+.6f} {cfg.symbol}")
        print(f"  Please check both exchanges manually.\n")


    await signer.close()
    await api_client.close()
    await edgex.close()

async def test_leverage_setup(cfg: AppConfig, env: dict) -> None:
    """Test function to verify leverage can be set correctly on both exchanges."""

    print(f"\n┌{'─' * 66}┐")
    print(f"│{'Test Leverage Setup':^66}│")
    print(f"├{'─' * 66}┤")
    print(f"│  Market:           {cfg.symbol}/{cfg.quote:<4}                                      │")
    print(f"│  Target Leverage:  {cfg.leverage}x                                             │")
    print(f"│                                                                  │")
    print(f"│  This will attempt to set leverage on both exchanges and        │")
    print(f"│  verify the settings (EdgeX only - Lighter verification         │")
    print(f"│  requires an open position).                                    │")
    print(f"└{'─' * 66}┘\n")

    # Build clients
    api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
    order_api = lighter.OrderApi(api_client)
    # GitHub SDK signature: SignerClient(url, account_index, api_private_keys, api_key_index)
    signer = lighter.SignerClient(
        env["LIGHTER_BASE_URL"],
        int(env["ACCOUNT_INDEX"]),
        int(env["API_KEY_INDEX"]),
        env["API_KEY_PRIVATE_KEY"]
    )
    err = signer.check_client()
    if err:
        raise RuntimeError(f"Lighter check_client error: {err}")

    edgex = EdgeXClient(
        base_url=env["EDGEX_BASE_URL"],
        account_id=int(env["EDGEX_ACCOUNT_ID"]) if env.get("EDGEX_ACCOUNT_ID") else 0,
        stark_private_key=env.get("EDGEX_STARK_PRIVATE_KEY", ""),
    )

    # Get market details
    try:
        l_market_id, _, _ = await lighter_get_market_details(order_api, cfg.symbol)
        contract_name = f"{cfg.symbol.upper()}{cfg.quote.upper()}"
        e_contract_id, _, _ = await edgex_find_contract_id(edgex, contract_name)
    except Exception as e:
        print(f"❌ Failed to get market details: {e}")
        await signer.close()
        await api_client.close()
        await edgex.close()
        raise SystemExit(1)

    # Test leverage setup
    edgex_ok, lighter_ok = await setup_leverage_both_exchanges(
        cfg, env, edgex, signer, e_contract_id, l_market_id, verify=True
    )

    # Summary
    print("\n" + "═" * 68)
    if edgex_ok and lighter_ok:
        print("✓ Leverage test PASSED")
        print("  Both exchanges accepted the leverage setting")
    elif edgex_ok or lighter_ok:
        print("⚠ Leverage test PARTIALLY PASSED")
        print(f"  EdgeX: {'✓ OK' if edgex_ok else '✗ FAILED'}")
        print(f"  Lighter: {'✓ OK' if lighter_ok else '✗ FAILED'}")
    else:
        print("✗ Leverage test FAILED")
        print("  Could not set leverage on either exchange")
    print("═" * 68 + "\n")

    await signer.close()
    await api_client.close()
    await edgex.close()

async def test_hedge(cfg: AppConfig, env: dict, notional_usd: float = 20.0, cross_ticks: int = 10) -> None:
    """Test function: opens a small delta-neutral position, waits, then closes it.
    Asks for user confirmation between each step."""

    print(f"\n┌{'─' * 66}┐")
    print(f"│{'Test Delta-Neutral Hedge':^66}│")
    print(f"├{'─' * 66}┤")
    print(f"│  Market:           {cfg.symbol}/{cfg.quote:<4}                                      │")
    print(f"│  Test Size:        ${notional_usd:.2f} USD per exchange                     │")
    print(f"│  Leverage:         {cfg.leverage}x                                             │")
    print(f"│                                                                  │")
    print(f"│  This will:                                                      │")
    print(f"│    1. Open delta-neutral positions (~${notional_usd:.0f} each side)          │")
    print(f"│    2. Wait for your confirmation to proceed                      │")
    print(f"│    3. Close both positions                                       │")
    print(f"└{'─' * 66}┘\n")

    # Initial confirmation
    response = input("Press ENTER to start the test, or 'q' to quit: ").strip().lower()
    if response == 'q':
        print("Test aborted by user.\n")
        raise SystemExit(0)

    # Step 1: Open
    print("\n" + "═" * 68)
    print("STEP 1: Opening test positions")
    print("═" * 68)
    try:
        await open_hedge(cfg, env, size_base=None, size_quote=notional_usd, cross_ticks=cross_ticks)
    except Exception as e:
        print(f"\n❌ Failed to open positions: {e}")
        print(f"⚠️  Test aborted. Please check positions manually.\n")
        raise SystemExit(1)

    # Confirmation before closing
    print("\n" + "═" * 68)
    print("STEP 2: Verify positions on exchanges")
    print("═" * 68)
    print("✓ Positions opened successfully!")
    print("\nPlease verify positions on both exchanges:")
    print(f"  - Check {cfg.long_exchange.capitalize()} for LONG position")
    print(f"  - Check {cfg.short_exchange.capitalize()} for SHORT position")
    print(f"  - Both should be ~${notional_usd:.2f} notional")
    print(f"\nYou can leave positions open to collect funding, or close them now.")
    print(f"⚠️  NOTE: Minimum 2 minute wait required before closing (self-trade protection)")

    import time
    start_time = time.time()
    min_wait = 120  # 2 minutes minimum

    while True:
        response = input("\nType 'close' to close positions, 'wait' to wait 30s, or 'skip' to exit without closing: ").strip().lower()
        elapsed = int(time.time() - start_time)

        if response == 'close':
            if elapsed < min_wait:
                remaining = min_wait - elapsed
                print(f"⚠️  Must wait at least 2 minutes before closing (self-trade protection)")
                print(f"   Time elapsed: {elapsed}s / {min_wait}s")
                print(f"   Still need to wait {remaining}s. Use 'wait' to continue or just wait.")
                continue
            break
        elif response == 'wait':
            print("Waiting 30 seconds...")
            for i in range(30, 0, -5):
                print(f"  {i}s remaining...", flush=True)
                await asyncio.sleep(5)
            elapsed = int(time.time() - start_time)
            print(f"Total elapsed time: {elapsed}s / {min_wait}s")
        elif response == 'skip':
            print("\n⚠️  Positions left open. Close manually when ready.\n")
            raise SystemExit(0)
        else:
            print("Invalid input. Please type 'close', 'wait', or 'skip'.")

    # Step 3: Close
    print("\n" + "═" * 68)
    print("STEP 3: Closing test positions")
    print("═" * 68)
    try:
        await close_both(cfg, env, cross_ticks=cross_ticks)
    except Exception as e:
        print(f"\n❌ Failed to close positions: {e}")
        print(f"⚠️  Please close positions manually on both exchanges.\n")
        raise SystemExit(1)

    # Summary
    print("\n" + "═" * 68)
    print(f"✓ Test completed successfully!")
    print("═" * 68)
    print(f"  Opened and closed ${notional_usd:.2f} positions on both exchanges")
    print(f"  Check your exchange accounts for fees/slippage details\n")


async def test_auto_hedge(cfg: AppConfig, env: dict, notional_usd: float = 20.0, cross_ticks: int = 10) -> None:
    """Automated test: opens a small delta-neutral position, waits 2 mins, then closes it."""

    print(f"\n┌{'─' * 66}┐")
    print(f"│{'Automated Delta-Neutral Hedge Test':^66}│")
    print(f"├{'─' * 66}┤")
    print(f"│  Market:           {cfg.symbol}/{cfg.quote:<4}                                      │")
    print(f"│  Test Size:        ${notional_usd:.2f} USD per exchange                     │")
    print(f"│  Leverage:         {cfg.leverage}x                                             │")
    print(f"│                                                                  │")
    print(f"│  This will automatically:                                        │")
    print(f"│    1. Open delta-neutral positions (~${notional_usd:.0f} each side)          │")
    print(f"│    2. Wait for 2 minutes                                         │")
    print(f"│    3. Close both positions                                       │")
    print(f"└{'─' * 66}┘\n")

    # Step 1: Open
    print("\n" + "═" * 68)
    print("STEP 1: Opening test positions")
    print("═" * 68)
    try:
        await open_hedge(cfg, env, size_base=None, size_quote=notional_usd, cross_ticks=cross_ticks)
    except Exception as e:
        print(f"\n❌ Failed to open positions: {e}")
        print(f"⚠️  Test aborted. Please check positions manually.\n")
        raise SystemExit(1)

    # Step 2: Wait
    print("\n" + "═" * 68)
    print("STEP 2: Waiting for 2 minutes before closing")
    print("═" * 68)
    print("✓ Positions opened successfully!")
    print("Waiting for 120 seconds (self-trade protection)...")
    for i in range(120, 0, -10):
        print(f"  {i}s remaining...", flush=True)
        await asyncio.sleep(10)
    print("Wait complete.")


    # Step 3: Close
    print("\n" + "═" * 68)
    print("STEP 3: Closing test positions")
    print("═" * 68)
    try:
        await close_both(cfg, env, cross_ticks=cross_ticks)
    except Exception as e:
        print(f"\n❌ Failed to close positions: {e}")
        print(f"⚠️  Please close positions manually on both exchanges.\n")
        raise SystemExit(1)

    # Summary
    print("\n" + "═" * 68)
    print(f"✓ Automated test completed successfully!")
    print("═" * 68)
    print(f"  Opened and closed ${notional_usd:.2f} positions on both exchanges")
    print(f"  Check your exchange accounts for fees/slippage details\n")


# ---------- Funding Rate ----------
async def edgex_get_funding_info(client: EdgeXClient, contract_id: str, base_url: str, symbol: str, quote: str) -> Optional[float]:
    """Fetches and displays current and historical funding rates for EdgeX. Returns current rate as percentage."""
    logger.info(f"Fetching funding rates for EdgeX contract: {contract_id}")

    current_rate = None

    # 1. Get current rate from the ticker/quote
    try:
        quote_resp = await client.quote.get_24_hour_quote(contract_id)
        if quote_resp.get("code") == "SUCCESS" and quote_resp.get("data"):
            quote_data = quote_resp["data"][0]
            current_rate = float(quote_data.get("fundingRate", "0")) * 100
            next_funding_time_ms = int(quote_data.get("nextFundingTime", "0"))
            next_funding_dt = datetime.fromtimestamp(next_funding_time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')

            # Calculate APR: EdgeX pays 6 times per day (every 4 hours)
            edgex_apr = current_rate * 6 * 365

            print(f"\n┌──────────────────────── EdgeX Funding ───────────────────────┐")
            print(f"│  Market: {symbol}/{quote:>4}                                               │")
            print(f"│                                                              │")
            print(f"│  Current Rate:          {current_rate:>8.6f}% per period                  │")
            print(f"│  Annualized (APR):      {edgex_apr:>8.2f}%                           │")
            print(f"│  Frequency:             Every 4 hours (6x/day)              │")
            print(f"│  Next Funding:          {next_funding_dt} UTC              │")
        else:
            print(f"\n┌──────────────────────── EdgeX Funding ───────────────────────┐")
            print(f"│  Market: {symbol}/{quote:>4}                                               │")
            print(f"│                                                              │")
            print(f"│  Could not retrieve current funding rate.                   │")
    except Exception as e:
        logger.error(f"Error fetching current EdgeX funding rate: {e}")
        print(f"\n┌──────────────────────── EdgeX Funding ───────────────────────┐")
        print(f"│  Market: {symbol}/{quote:>4}                                               │")
        print(f"│                                                              │")
        print(f"│  Failed to retrieve current funding rate.                   │")

    # 2. Get historical rates
    try:
        import aiohttp
        params = {"contractId": contract_id, "size": "3"}
        # Direct HTTP request since SDK doesn't expose this endpoint
        url = f"{base_url}/api/v1/public/funding/getFundingRatePage"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                hist_resp = await response.json()

        if hist_resp.get("code") == "SUCCESS" and hist_resp.get("data", {}).get("dataList"):
            print(f"│                                                              │")
            print(f"│  Recent Historical Rates:                                    │")
            for rate_info in hist_resp["data"]["dataList"]:
                rate = float(rate_info.get("fundingRate", "0")) * 100
                rate_time_ms = int(rate_info.get("fundingTime", "0"))
                rate_dt = datetime.fromtimestamp(rate_time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')
                print(f"│    {rate_dt} UTC    {rate:>8.6f}%                     │")
            print(f"└──────────────────────────────────────────────────────────────┘")
        else:
            print(f"│  Could not retrieve historical funding rates.               │")
            print(f"└──────────────────────────────────────────────────────────────┘")
            logger.debug(f"Historical funding response: {hist_resp}")
    except Exception as e:
        logger.error(f"Error fetching historical EdgeX funding rates: {e}")
        print(f"│  Failed to retrieve historical funding rates.               │")
        print(f"└──────────────────────────────────────────────────────────────┘")

    return current_rate

async def lighter_get_funding_info(api_client: lighter.ApiClient, market_id: int, symbol: str, quote: str) -> Optional[float]:
    """Fetches and displays recent funding rates for Lighter. Returns current rate as percentage."""
    logger.info(f"Fetching funding rates for Lighter market: {market_id}")
    print(f"\n┌─────────────────────── Lighter Funding ──────────────────────┐")
    print(f"│  Market: {symbol}/{quote:>4}                                               │")
    print(f"│                                                              │")

    latest_rate = None

    try:
        candlestick_api = lighter.CandlestickApi(api_client)
        end_ts = int(time.time() * 1000)
        # Go back 3 days to be safe, as funding can be every 8 hours
        start_ts = end_ts - (3 * 24 * 3600 * 1000)
        funding_resp = await candlestick_api.fundings(market_id=market_id, resolution="1h", start_timestamp=start_ts, end_timestamp=end_ts, count_back=10)

        if funding_resp.fundings:
            # Get latest rate for "Current Rate" display
            latest_funding = funding_resp.fundings[-1] if funding_resp.fundings else None
            if latest_funding:
                raw_rate = float(latest_funding.rate)
                logger.debug(f"Raw Lighter funding rate: {raw_rate}")

                # CORRECTED: the Lighter rate is a DECIMAL, not a percentage.
                #
                # The old comment here claimed "already as percentage" and passed the
                # raw value through, which understated every Lighter APR by 100x.
                #
                # Verified against live data: /api/v1/funding-rates returns Lighter and
                # Hyperliquid side by side, and across 98 same-sign common symbols the
                # median lighter/hyperliquid ratio is 0.9600 (e.g. 0.000096 vs
                # 0.00010). Hyperliquid's rate is unambiguously an hourly DECIMAL, so
                # Lighter's is too.
                latest_rate = raw_rate * 100.0

                timestamp_value = int(latest_funding.timestamp)
                if timestamp_value > 1e12:
                    rate_time_s = timestamp_value / 1000
                else:
                    rate_time_s = timestamp_value
                latest_dt = datetime.fromtimestamp(rate_time_s).strftime('%Y-%m-%d %H:%M:%S')

                # Calculate APR: Lighter pays 24 times per day (every hour)
                lighter_apr = latest_rate * 24 * 365

                print(f"│  Current Rate:          {latest_rate:>8.6f}% per period                  │")
                print(f"│  Annualized (APR):      {lighter_apr:>8.2f}%                           │")
                print(f"│  Frequency:             Every hour (24x/day)                │")
                print(f"│  Last Updated:          {latest_dt} UTC              │")

            print(f"│                                                              │")
            print(f"│  Recent Historical Rates:                                    │")
            # Show only last 3 entries for consistency with EdgeX
            recent_fundings = list(reversed(funding_resp.fundings))[-3:]
            for rate_info in reversed(recent_fundings):
                rate = float(rate_info.rate)  # Already in percentage
                # Lighter timestamp - check if it's already in seconds or needs conversion
                timestamp_value = int(rate_info.timestamp)
                logger.debug(f"Raw timestamp from Lighter: {timestamp_value}")

                # If timestamp > 1e12, it's in milliseconds; otherwise it's in seconds
                if timestamp_value > 1e12:
                    rate_time_s = timestamp_value / 1000
                else:
                    rate_time_s = timestamp_value

                rate_dt = datetime.fromtimestamp(rate_time_s).strftime('%Y-%m-%d %H:%M:%S')
                print(f"│    {rate_dt} UTC    {rate:>8.6f}%                     │")
            print(f"└──────────────────────────────────────────────────────────────┘")
        else:
            print(f"│  No funding rate data found for this market.                 │")
            print(f"└──────────────────────────────────────────────────────────────┘")

    except Exception as e:
        logger.error(f"Error fetching Lighter funding rates: {e}")
        print(f"│  Failed to retrieve funding rates.                           │")
        print(f"└──────────────────────────────────────────────────────────────┘")

    return latest_rate


async def show_funding_rates(cfg: AppConfig, env: dict) -> Tuple[Optional[str], Optional[str]]:
    """Connects to exchanges and prints funding rate information. Returns (optimal_long_exch, optimal_short_exch) if config is suboptimal."""

    edgex_rate = None
    lighter_rate = None

    # EdgeX
    edgex = EdgeXClient(
        base_url=env["EDGEX_BASE_URL"],
        account_id=int(env["EDGEX_ACCOUNT_ID"]) if env.get("EDGEX_ACCOUNT_ID") else 0,
        stark_private_key=env.get("EDGEX_STARK_PRIVATE_KEY", ""),
    )
    try:
        contract_name = f"{cfg.symbol.upper()}{cfg.quote.upper()}"
        e_contract_id, _, _ = await edgex_find_contract_id(edgex, contract_name)
        edgex_rate = await edgex_get_funding_info(edgex, e_contract_id, env["EDGEX_BASE_URL"], cfg.symbol, cfg.quote)
    except Exception as e:
        logger.error(f"Could not process EdgeX funding rates: {e}")
    finally:
        await edgex.close()

    # Lighter
    lighter_api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
    order_api = lighter.OrderApi(lighter_api_client)
    try:
        l_market_id, _, _ = await lighter_get_market_details(order_api, cfg.symbol)
        lighter_rate = await lighter_get_funding_info(lighter_api_client, l_market_id, cfg.symbol, cfg.quote)
    except Exception as e:
        logger.error(f"Could not process Lighter funding rates: {e}")
    finally:
        await lighter_api_client.close()

    # Analysis and recommendation
    if edgex_rate is not None and lighter_rate is not None:
        print(f"\n┌────────────────────── Funding Analysis ──────────────────────┐")
        print(f"│                                                              │")

        # Calculate spread
        spread = abs(edgex_rate - lighter_rate)
        spread_bps = spread * 100  # Convert to basis points

        print(f"│  EdgeX Rate:            {edgex_rate:>8.6f}%                           │")
        print(f"│  Lighter Rate:          {lighter_rate:>8.6f}%                           │")
        print(f"│  Spread:                {spread:>8.6f}% ({spread_bps:>6.2f} bps)              │")
        print(f"│                                                              │")

        # Determine optimal configuration
        # If both positive: short on higher rate (collect more funding), long on lower rate (pay less)
        # If both negative: long on more negative (collect more), short on less negative (pay less)
        # If mixed: long on negative (collect), short on positive (collect)

        if edgex_rate > 0 and lighter_rate > 0:
            # Both positive - short the higher, long the lower
            if edgex_rate > lighter_rate:
                long_exch = "Lighter"
                short_exch = "EdgeX"
                reason = "EdgeX has higher positive funding (collect more)"
            else:
                long_exch = "EdgeX"
                short_exch = "Lighter"
                reason = "Lighter has higher positive funding (collect more)"
        elif edgex_rate < 0 and lighter_rate < 0:
            # Both negative - long the more negative, short the less negative
            if edgex_rate < lighter_rate:
                long_exch = "EdgeX"
                short_exch = "Lighter"
                reason = "EdgeX more negative (collect more funding)"
            else:
                long_exch = "Lighter"
                short_exch = "EdgeX"
                reason = "Lighter more negative (collect more funding)"
        else:
            # Mixed signs - long negative, short positive
            if edgex_rate < 0:
                long_exch = "EdgeX"
                short_exch = "Lighter"
                reason = "EdgeX negative (collect), Lighter positive (collect)"
            else:
                long_exch = "Lighter"
                short_exch = "EdgeX"
                reason = "Lighter negative (collect), EdgeX positive (collect)"

        # Calculate estimated APR
        # EdgeX: pays every 4 hours = 6 times per day
        # Lighter: pays every hour = 24 times per day
        edgex_periods_per_day = 6  # Every 4 hours
        lighter_periods_per_day = 24  # Every hour
        days_per_year = 365

        # For delta-neutral strategy:
        # - Collect funding on SHORT position (pay funding on LONG position)
        # - Net profit = (short_rate * short_periods) - (long_rate * long_periods)

        if short_exch.lower() == "edgex":
            short_periods = edgex_periods_per_day
            long_periods = lighter_periods_per_day
        else:
            short_periods = lighter_periods_per_day
            long_periods = edgex_periods_per_day

        # Daily profit from funding
        daily_profit = (abs(lighter_rate if short_exch.lower() == "lighter" else edgex_rate) * short_periods) - \
                       (abs(edgex_rate if short_exch.lower() == "lighter" else lighter_rate) * long_periods)

        estimated_apr = daily_profit * days_per_year

        print(f"│  RECOMMENDATION:                                             │")
        print(f"│    LONG  on {long_exch:>7}                                         │")
        print(f"│    SHORT on {short_exch:>7}                                         │")
        print(f"│                                                              │")
        print(f"│  Rationale: {reason:48} │")
        print(f"│                                                              │")
        print(f"│  Estimated APR from funding: {estimated_apr:>7.2f}%                    │")
        print(f"│  (EdgeX: 6 periods/day, Lighter: 24 periods/day)            │")

        # Check if current config matches recommendation
        current_matches = (cfg.long_exchange.lower() == long_exch.lower() and
                          cfg.short_exchange.lower() == short_exch.lower())

        print(f"│                                                              │")
        if current_matches:
            print(f"│  ✓ Current config matches optimal setup                      │")
            print(f"│    (No changes needed to hedge_config.json)                  │")
        else:
            print(f"│  ⚠ Current config is SUBOPTIMAL                              │")
            print(f"│    (Currently: LONG={cfg.long_exchange.capitalize()}, SHORT={cfg.short_exchange.capitalize()})               │")
            print(f"│    Will update hedge_config.json to optimal setup           │")

        print(f"└──────────────────────────────────────────────────────────────┘")

        # Update config if suboptimal
        if not current_matches:
            return long_exch, short_exch
        return None, None

    return None, None


async def check_symbol_availability(symbol: str, quote: str, env: dict) -> Tuple[bool, bool]:
    """Check if a symbol is available on both EdgeX and Lighter. Returns (edgex_available, lighter_available)."""
    edgex_available = False
    lighter_available = False

    # Check EdgeX
    try:
        edgex = EdgeXClient(
            base_url=env["EDGEX_BASE_URL"],
            account_id=int(env["EDGEX_ACCOUNT_ID"]) if env.get("EDGEX_ACCOUNT_ID") else 0,
            stark_private_key=env.get("EDGEX_STARK_PRIVATE_KEY", ""),
        )
        contract_name = f"{symbol.upper()}{quote.upper()}"
        try:
            await edgex_find_contract_id(edgex, contract_name)
            edgex_available = True
        except RuntimeError:
            pass
        await edgex.close()
    except Exception:
        pass

    # Check Lighter
    try:
        lighter_api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
        order_api = lighter.OrderApi(lighter_api_client)
        try:
            await lighter_get_market_details(order_api, symbol)
            lighter_available = True
        except RuntimeError:
            pass
        await lighter_api_client.close()
    except Exception:
        pass

    return edgex_available, lighter_available


async def fetch_symbol_funding(symbol: str, quote: str, env: dict) -> dict:
    """Fetch funding rates for a single symbol."""
    logger.info(f"Checking {symbol}...")

    # Check availability
    edgex_avail, lighter_avail = await check_symbol_availability(symbol, quote, env)

    if not (edgex_avail and lighter_avail):
        status = []
        if not edgex_avail:
            status.append("EdgeX")
        if not lighter_avail:
            status.append("Lighter")
        logger.warning(f"{symbol}: Not available on {', '.join(status)}")
        return {
            "symbol": symbol,
            "available": False,
            "missing_on": status
        }

    # Get funding rates
    edgex_rate = None
    lighter_rate = None

    # EdgeX
    try:
        edgex = EdgeXClient(
            base_url=env["EDGEX_BASE_URL"],
            account_id=int(env["EDGEX_ACCOUNT_ID"]) if env.get("EDGEX_ACCOUNT_ID") else 0,
            stark_private_key=env.get("EDGEX_STARK_PRIVATE_KEY", ""),
        )
        contract_name = f"{symbol.upper()}{quote.upper()}"
        e_contract_id, _, _ = await edgex_find_contract_id(edgex, contract_name)

        # Get rate without printing
        quote_resp = await edgex.quote.get_24_hour_quote(e_contract_id)
        if quote_resp.get("code") == "SUCCESS" and quote_resp.get("data"):
            quote_data = quote_resp["data"][0]
            edgex_rate = float(quote_data.get("fundingRate", "0")) * 100

        await edgex.close()
    except Exception as e:
        logger.error(f"Error fetching EdgeX funding for {symbol}: {e}")

    # Lighter
    try:
        lighter_api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
        order_api = lighter.OrderApi(lighter_api_client)
        l_market_id, _, _ = await lighter_get_market_details(order_api, symbol)

        # Get rate without printing
        candlestick_api = lighter.CandlestickApi(lighter_api_client)
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (3 * 24 * 3600 * 1000)
        funding_resp = await candlestick_api.fundings(market_id=l_market_id, resolution="1h",
                                                      start_timestamp=start_ts, end_timestamp=end_ts, count_back=1)

        if funding_resp.fundings:
            latest_funding = funding_resp.fundings[-1]
            lighter_rate = float(latest_funding.rate)

        await lighter_api_client.close()
    except Exception as e:
        logger.error(f"Error fetching Lighter funding for {symbol}: {e}")

    if edgex_rate is not None and lighter_rate is not None:
        # Calculate APRs
        edgex_apr = edgex_rate * 6 * 365
        lighter_apr = lighter_rate * 24 * 365

        # Determine optimal setup (compare APRs, not period rates)
        if edgex_apr > lighter_apr:
            long_exch = "Lighter"
            short_exch = "EdgeX"
            short_rate = edgex_rate
            short_periods = 6
            long_rate = lighter_rate
            long_periods = 24
        else:
            long_exch = "EdgeX"
            short_exch = "Lighter"
            short_rate = lighter_rate
            short_periods = 24
            long_rate = edgex_rate
            long_periods = 6

        # Calculate net APR
        # When going long, negative funding means you receive funding (profit)
        # So we ADD abs() when long_rate is negative, SUBTRACT when positive
        if long_rate < 0:
            daily_profit = (abs(short_rate) * short_periods) + (abs(long_rate) * long_periods)
        else:
            daily_profit = (abs(short_rate) * short_periods) - (abs(long_rate) * long_periods)
        net_apr = daily_profit * 365

        return {
            "symbol": symbol,
            "available": True,
            "edgex_rate": edgex_rate,
            "edgex_apr": edgex_apr,
            "lighter_rate": lighter_rate,
            "lighter_apr": lighter_apr,
            "long_exch": long_exch,
            "short_exch": short_exch,
            "net_apr": net_apr
        }

    return {
        "symbol": symbol,
        "available": False,
        "missing_on": ["Data unavailable"]
    }


async def show_funding_rates_multi(symbols: list, quote: str, env: dict):
    """Show funding rates for multiple symbols in a summary table."""

    print(f"\nChecking funding rates for {len(symbols)} symbols...")
    print(f"(Symbols available on both exchanges will be analyzed)\n")

    # Fetch all symbols concurrently
    results = await asyncio.gather(*[fetch_symbol_funding(symbol, quote, env) for symbol in symbols])

    # Print summary table
    print(f"\n┌{'─' * 86}┐")
    print(f"│{'Funding Rate Summary':^86}│")
    print(f"├{'─' * 86}┤")
    print(f"│ {'Symbol':<8} │ {'EdgeX APR':>12} │ {'Lighter APR':>12} │ {'Optimal Setup':<18} │ {'Net APR':>12} │")
    print(f"├{'─' * 86}┤")

    for result in results:
        if not result["available"]:
            missing = ", ".join(result["missing_on"])
            print(f"│ {result['symbol']:<8} │ {'N/A':>12} │ {'N/A':>12} │ {'Not on ' + missing:<18} │ {'N/A':>12} │")
        else:
            setup = f"L:{result['long_exch'][:1]} S:{result['short_exch'][:1]}"
            print(f"│ {result['symbol']:<8} │ {result['edgex_apr']:>11.2f}% │ {result['lighter_apr']:>11.2f}% │ {setup:<18} │ {result['net_apr']:>11.2f}% │")

    print(f"└{'─' * 86}┘")
    print(f"\nLegend: L:E = Long EdgeX, L:L = Long Lighter, S:E = Short EdgeX, S:L = Short Lighter")


async def show_leverage_info(symbols: list, quote: str, env: dict):
    """Show maximum leverage for multiple symbols in a summary table."""

    results = []

    print(f"\nChecking maximum leverage for {len(symbols)} symbols...")
    print(f"(Symbols available on both exchanges will be analyzed)\n")

    for symbol in symbols:
        logger.info(f"Checking leverage for {symbol}...")

        # Check availability
        edgex_avail, lighter_avail = await check_symbol_availability(symbol, quote, env)

        if not (edgex_avail and lighter_avail):
            status = []
            if not edgex_avail:
                status.append("EdgeX")
            if not lighter_avail:
                status.append("Lighter")
            logger.warning(f"{symbol}: Not available on {', '.join(status)}")
            results.append({
                "symbol": symbol,
                "available": False,
                "missing_on": status
            })
            continue

        # Get leverage info
        edgex_max_leverage = None
        edgex_default_leverage = None
        lighter_max_leverage = None
        lighter_default_leverage = None

        # EdgeX - check contract metadata for max leverage
        try:
            edgex = EdgeXClient(
                base_url=env["EDGEX_BASE_URL"],
                account_id=int(env["EDGEX_ACCOUNT_ID"]) if env.get("EDGEX_ACCOUNT_ID") else 0,
                stark_private_key=env.get("EDGEX_STARK_PRIVATE_KEY", ""),
            )
            contract_name = f"{symbol.upper()}{quote.upper()}"

            metadata = await edgex.get_metadata()
            contracts = metadata.get("data", {}).get("contractList", [])
            for c in contracts:
                if c.get("contractName") == contract_name:
                    # EdgeX uses displayMaxLeverage for max leverage
                    edgex_max_leverage = float(c.get("displayMaxLeverage", "0"))
                    edgex_default_leverage = float(c.get("defaultLeverage", "0"))

                    if edgex_max_leverage == 0:
                        edgex_max_leverage = None
                    if edgex_default_leverage == 0:
                        edgex_default_leverage = None

                    logger.debug(f"EdgeX {symbol} - max: {edgex_max_leverage}, default: {edgex_default_leverage}")
                    break

            await edgex.close()
        except Exception as e:
            logger.error(f"Error fetching EdgeX leverage for {symbol}: {e}")

        # Lighter - Lighter's API doesn't expose max leverage per market in order books
        # Using known defaults: Lighter typically offers 20x max leverage
        try:
            lighter_api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
            order_api = lighter.OrderApi(lighter_api_client)

            # Just verify the market exists
            await lighter_get_market_details(order_api, symbol)

            # Lighter default values (these are typical for Lighter)
            lighter_max_leverage = 20.0  # Lighter typically supports up to 20x
            lighter_default_leverage = 3.0  # Default leverage

            logger.debug(f"Lighter {symbol} - max: {lighter_max_leverage} (default), default: {lighter_default_leverage}")

            await lighter_api_client.close()
        except Exception as e:
            logger.error(f"Error fetching Lighter leverage for {symbol}: {e}")

        results.append({
            "symbol": symbol,
            "available": True,
            "edgex_max_leverage": edgex_max_leverage,
            "edgex_default_leverage": edgex_default_leverage,
            "lighter_max_leverage": lighter_max_leverage,
            "lighter_default_leverage": lighter_default_leverage,
        })

    # Print summary table
    print(f"\n┌{'─' * 94}┐")
    print(f"│{'Leverage Summary':^94}│")
    print(f"├{'─' * 94}┤")
    print(f"│ {'Symbol':<8} │ {'EdgeX Default':>15} │ {'EdgeX Max':>13} │ {'Lighter Default':>17} │ {'Lighter Max':>15} │ {'Min Max':>10} │")
    print(f"├{'─' * 94}┤")

    for result in results:
        if not result["available"]:
            missing = ", ".join(result["missing_on"])
            print(f"│ {result['symbol']:<8} │ {'N/A':>15} │ {'N/A':>13} │ {'N/A':>17} │ {'N/A':>15} │ {'N/A':>10} │")
        else:
            edgex_def = result['edgex_default_leverage']
            edgex_max = result['edgex_max_leverage']
            lighter_def = result['lighter_default_leverage']
            lighter_max = result['lighter_max_leverage']

            edgex_def_str = f"{edgex_def:.0f}x" if edgex_def and edgex_def > 0 else "N/A"
            edgex_max_str = f"{edgex_max:.0f}x" if edgex_max and edgex_max > 0 else "N/A"
            lighter_def_str = f"{lighter_def:.0f}x" if lighter_def and lighter_def > 0 else "N/A"
            lighter_max_str = f"{lighter_max:.0f}x" if lighter_max and lighter_max > 0 else "N/A"

            # Calculate min max for delta-neutral (both must have max leverage)
            if edgex_max and lighter_max and edgex_max > 0 and lighter_max > 0:
                min_max = min(edgex_max, lighter_max)
                min_max_str = f"{min_max:.0f}x"
            else:
                min_max_str = "N/A"

            print(f"│ {result['symbol']:<8} │ {edgex_def_str:>15} │ {edgex_max_str:>13} │ {lighter_def_str:>17} │ {lighter_max_str:>15} │ {min_max_str:>10} │")

    print(f"└{'─' * 94}┘")
    print(f"\nNote: 'Min Max' shows the maximum usable leverage for delta-neutral hedging")
    print(f"      (limited by the exchange with lower max leverage)")
    print(f"      Lighter max leverage value is typical default (may vary by market)")


async def edgex_get_funding_payments(client: EdgeXClient, contract_id: str, start_time_ms: Optional[int] = None) -> Tuple[float, int, Optional[int]]:
    """
    Fetch funding payments for a specific contract from EdgeX.

    Args:
        client: EdgeX client
        contract_id: Contract ID to query
        start_time_ms: Optional start time in milliseconds (epoch)

    Returns:
        Tuple of (total_funding_usd, num_payments, position_open_time_ms)
    """
    from edgex_sdk import GetPositionTransactionPageParams

    params = GetPositionTransactionPageParams(
        size="100",
        filter_contract_id_list=[contract_id]
    )

    # Add start time filter if provided
    if start_time_ms:
        params.filter_start_created_time_inclusive = start_time_ms

    # Add funding fee filter by manually adding to query params
    # Note: SDK may not support filterTypeList directly, so we'll use the account client method
    account_client = client.account

    total_funding = 0.0
    num_payments = 0
    position_open_time_ms = None

    try:
        response = await account_client.get_position_transaction_page(params)

        logger.debug(f"EdgeX transaction response code: {response.get('code')}")

        # EdgeX returns code as 'SUCCESS' string or 0 integer
        code = response.get("code")
        is_success = code == "SUCCESS" or code == 0

        if is_success and response.get("data"):
            transactions = response["data"].get("dataList", [])
            logger.info(f"EdgeX: Found {len(transactions)} total transactions for contract {contract_id}")

            # Log transaction types to see what we're getting
            tx_types = {}
            for tx in transactions:
                tx_type = tx.get("type", "UNKNOWN")
                tx_types[tx_type] = tx_types.get(tx_type, 0) + 1
            logger.info(f"EdgeX transaction types: {tx_types}")

            # Find position open time (first BUY_POSITION or SELL_POSITION with deltaOpenSize > 0)
            for tx in reversed(transactions):  # Start from oldest
                tx_type = tx.get("type", "")
                if tx_type in ["BUY_POSITION", "SELL_POSITION"]:
                    delta_open = float(tx.get("deltaOpenSize", 0))
                    if abs(delta_open) > 0:
                        position_open_time_ms = int(tx.get("createdTime", 0))
                        logger.info(f"EdgeX position opened at {position_open_time_ms}")
                        break

            # Filter for SETTLE_FUNDING_FEE type transactions
            for tx in transactions:
                if tx.get("type") == "SETTLE_FUNDING_FEE":
                    # deltaFundingFee is the funding payment (negative = paid, positive = received)
                    delta_funding_fee = float(tx.get("deltaFundingFee", 0))
                    total_funding += delta_funding_fee
                    num_payments += 1

                    logger.debug(f"EdgeX funding payment: {delta_funding_fee} USD at {tx.get('createdTime')}")
        else:
            logger.warning(f"EdgeX transaction response code: {code}")

    except Exception as e:
        logger.error(f"Error fetching EdgeX funding payments: {e}", exc_info=True)
        return 0.0, 0, None

    return total_funding, num_payments, position_open_time_ms


async def lighter_get_funding_payments(account_api: lighter.AccountApi, signer: lighter.SignerClient, account_index: int, market_id: int, start_time_ms: Optional[int] = None) -> Tuple[float, int, Optional[int]]:
    """
    Fetch funding payments for a specific market from Lighter.

    Args:
        account_api: Lighter Account API
        signer: Lighter Signer Client (for authentication)
        account_index: Account index
        market_id: Market ID to query
        start_time_ms: Optional start time in milliseconds (epoch)

    Returns:
        Tuple of (total_funding_usd, num_payments, earliest_timestamp_ms)
    """
    total_funding = 0.0
    num_payments = 0
    earliest_timestamp = None

    try:
        # Generate auth token for the request
        auth_token_tuple = signer.create_auth_token_with_expiry()
        # Extract the token string from the tuple (first element)
        auth_token = auth_token_tuple[0] if isinstance(auth_token_tuple, tuple) else auth_token_tuple

        # Fetch position funding history
        response = await account_api.position_funding(
            account_index=account_index,
            market_id=market_id,
            limit=100,  # Get last 100 funding payments
            auth=auth_token
        )

        if hasattr(response, 'position_fundings'):
            for funding in response.position_fundings:
                # Check if we should filter by start time
                if start_time_ms and funding.timestamp < start_time_ms:
                    continue

                # Track earliest timestamp (position open time)
                if earliest_timestamp is None or funding.timestamp < earliest_timestamp:
                    earliest_timestamp = funding.timestamp

                # 'change' field is the USD value of funding payment
                # Negative = paid, positive = received
                change_usd = float(funding.change)
                total_funding += change_usd
                num_payments += 1

                logger.debug(f"Lighter funding payment: {change_usd} USD at {funding.timestamp}")

    except Exception as e:
        logger.error(f"Error fetching Lighter funding payments: {e}")
        return 0.0, 0, None

    return total_funding, num_payments, earliest_timestamp


async def check_position_status(cfg: AppConfig, env: dict):
    """Check current delta-neutral position status across both exchanges."""
    contract_name = cfg.symbol + cfg.quote

    # Initialize clients
    ex_client = EdgeXClient(
        base_url=env["EDGEX_BASE_URL"],
        account_id=int(env["EDGEX_ACCOUNT_ID"]),
        stark_private_key=env["EDGEX_STARK_PRIVATE_KEY"]
    )

    api_client = lighter.ApiClient(configuration=lighter.Configuration(host=env["LIGHTER_BASE_URL"]))
    order_api = lighter.OrderApi(api_client)
    # GitHub SDK signature: SignerClient(url, account_index, api_private_keys, api_key_index)
    signer = lighter.SignerClient(
        env["LIGHTER_BASE_URL"],
        int(env["ACCOUNT_INDEX"]),
        int(env["API_KEY_INDEX"]),
        env["API_KEY_PRIVATE_KEY"]
    )

    try:
        # Get contract/market info in parallel
        t_start = time.time()
        print("⏳ Fetching contract/market info...", flush=True)
        t0 = time.time()
        (edgex_contract_id, edgex_tick_size, edgex_step_size), \
        (lighter_market_id, lighter_tick_size, lighter_step_size) = await asyncio.gather(
            edgex_find_contract_id(ex_client, contract_name),
            lighter_get_market_details(order_api, cfg.symbol)
        )
        print(f"✓ Contract/market lookup: {time.time() - t0:.2f}s", flush=True)

        # Fetch prices first
        print("⏳ Fetching market prices...", flush=True)
        t0 = time.time()
        (edgex_bid, edgex_ask), (lighter_bid, lighter_ask) = await asyncio.gather(
            edgex_best_bid_ask(ex_client, edgex_contract_id),
            lighter_best_bid_ask(order_api, cfg.symbol, lighter_market_id)
        )

        # Calculate mid prices (needed for position details)
        edgex_mid = (edgex_bid + edgex_ask) / 2.0 if edgex_bid and edgex_ask else None
        lighter_mid = (lighter_bid + lighter_ask) / 2.0 if lighter_bid and lighter_ask else None

        # Fetch position details in parallel (requires mid prices for PnL calculation)
        lighter_account_api = lighter.AccountApi(api_client)
        edgex_pos_details, lighter_pos_details = await asyncio.gather(
            edgex_get_position_details(ex_client, edgex_contract_id, edgex_mid if edgex_mid else 0.0),
            lighter_get_position_details(lighter_account_api, env["ACCOUNT_INDEX"], lighter_market_id)
        )

        # Extract sizes for backward compatibility
        edgex_size = edgex_pos_details['size']
        lighter_size = lighter_pos_details['size']

        print(f"✓ Positions and prices: {time.time() - t0:.2f}s", flush=True)

        # Use average mid or fallback
        if edgex_mid and lighter_mid:
            avg_mid = (edgex_mid + lighter_mid) / 2.0
        elif edgex_mid:
            avg_mid = edgex_mid
        elif lighter_mid:
            avg_mid = lighter_mid
        else:
            avg_mid = None

        # Calculate position values
        edgex_value = abs(edgex_size) * edgex_mid if edgex_mid else None
        lighter_value = abs(lighter_size) * lighter_mid if lighter_mid else None

        # Calculate PnL percentages
        edgex_pnl_pct = None
        if edgex_value and edgex_value > 0:
            edgex_pnl_pct = (edgex_pos_details['unrealized_pnl'] / edgex_value) * 100

        lighter_pnl_pct = None
        if lighter_value and lighter_value > 0:
            lighter_pnl_pct = (lighter_pos_details['unrealized_pnl'] / lighter_value) * 100

        # Determine position direction
        edgex_direction = "LONG" if edgex_size > 0 else ("SHORT" if edgex_size < 0 else "FLAT")
        lighter_direction = "LONG" if lighter_size > 0 else ("SHORT" if lighter_size < 0 else "FLAT")

        # Calculate net exposure (should be near 0 for delta-neutral)
        net_size = edgex_size + lighter_size
        net_value = None
        if avg_mid:
            net_value = abs(net_size) * avg_mid

        # Determine if hedged
        is_hedged = False
        hedge_quality = "N/A"
        if abs(edgex_size) > 1e-9 and abs(lighter_size) > 1e-9:
            if (edgex_size > 0 and lighter_size < 0) or (edgex_size < 0 and lighter_size > 0):
                is_hedged = True
                # Calculate hedge balance (how well matched the sizes are)
                size_ratio = min(abs(edgex_size), abs(lighter_size)) / max(abs(edgex_size), abs(lighter_size))
                if size_ratio >= 0.99:
                    hedge_quality = "Excellent"
                elif size_ratio >= 0.95:
                    hedge_quality = "Good"
                elif size_ratio >= 0.90:
                    hedge_quality = "Fair"
                else:
                    hedge_quality = "Poor"

        # Fetch funding payments if we have positions
        edgex_funding = 0.0
        edgex_funding_count = 0
        edgex_open_time_ms = None
        lighter_funding = 0.0
        lighter_funding_count = 0
        lighter_open_time_ms = None
        total_funding = 0.0
        position_open_time_ms = None
        position_duration_str = "N/A"

        if abs(edgex_size) > 1e-9 or abs(lighter_size) > 1e-9:
            try:
                # Fetch funding payments in parallel (no time filter - get all available)
                print("⏳ Fetching funding payment history...", flush=True)
                t0 = time.time()
                (edgex_funding, edgex_funding_count, edgex_open_time_ms), \
                (lighter_funding, lighter_funding_count, lighter_open_time_ms) = await asyncio.gather(
                    edgex_get_funding_payments(ex_client, edgex_contract_id),
                    lighter_get_funding_payments(lighter_account_api, signer, env["ACCOUNT_INDEX"], lighter_market_id)
                )
                print(f"✓ Funding payments: {time.time() - t0:.2f}s", flush=True)
                total_funding = edgex_funding + lighter_funding

                # Calculate position duration if we have open time (use earliest)
                if edgex_open_time_ms:
                    position_open_time_ms = edgex_open_time_ms
                    current_time_ms = int(time.time() * 1000)
                    duration_ms = current_time_ms - position_open_time_ms
                    duration_hours = duration_ms / (1000 * 60 * 60)
                    duration_days = duration_hours / 24

                    if duration_days >= 1:
                        position_duration_str = f"{duration_days:.1f} days"
                    else:
                        position_duration_str = f"{duration_hours:.1f} hours"

            except Exception as e:
                logger.warning(f"Failed to fetch funding payment history: {e}")

        # Format position open date/time if available
        from datetime import datetime
        edgex_open_date_str = "N/A"
        if edgex_open_time_ms:
            edgex_open_dt = datetime.fromtimestamp(edgex_open_time_ms / 1000)
            edgex_open_date_str = edgex_open_dt.strftime('%Y-%m-%d %H:%M:%S UTC')

        lighter_open_date_str = "N/A"
        # Try to use timestamp from position object first (more accurate)
        lighter_pos_timestamp = lighter_pos_details.get('open_timestamp')
        if lighter_pos_timestamp:
            lighter_open_dt = datetime.fromtimestamp(lighter_pos_timestamp)
            lighter_open_date_str = lighter_open_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        elif lighter_open_time_ms:
            # Fallback: Use EdgeX time if available (positions opened together)
            # Otherwise use funding payment timestamp (less accurate - rounded to hour)
            if edgex_open_time_ms and abs(edgex_open_time_ms - lighter_open_time_ms * 1000) < 3600000:  # Within 1 hour
                lighter_open_dt = datetime.fromtimestamp(edgex_open_time_ms / 1000)
                lighter_open_date_str = lighter_open_dt.strftime('%Y-%m-%d %H:%M:%S UTC') + " (~same as EdgeX)"
            else:
                lighter_open_dt = datetime.fromtimestamp(lighter_open_time_ms)
                lighter_open_date_str = lighter_open_dt.strftime('%Y-%m-%d %H:%M:%S UTC') + " (estimated)"

        # Keep the overall position open time for duration calculation
        position_open_date_str = "N/A"
        if position_open_time_ms:
            position_open_dt = datetime.fromtimestamp(position_open_time_ms / 1000)
            position_open_date_str = position_open_dt.strftime('%Y-%m-%d %H:%M:%S UTC')

        # Calculate trading fees (open + estimated close)
        # Fee rate: 0.036% per trade
        fee_rate = 0.00036
        trading_fees_open = 0.0
        trading_fees_total = 0.0  # Open + Close

        if abs(edgex_size) > 1e-9 or abs(lighter_size) > 1e-9:
            # Calculate fees based on notional values
            edgex_fee = 0.0
            lighter_fee = 0.0

            if edgex_value is not None:
                edgex_fee = edgex_value * fee_rate
            if lighter_value is not None:
                lighter_fee = lighter_value * fee_rate

            trading_fees_open = edgex_fee + lighter_fee
            trading_fees_total = trading_fees_open * 2  # Double for open + close

        # ANSI Color codes (subtle colors)
        RESET = "\033[0m"
        BOLD = "\033[1m"
        CYAN = "\033[36m"
        GREEN = "\033[32m"
        RED = "\033[31m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        DIM = "\033[2m"

        # Formatted output with colors
        print(f"\n✓ Total time: {time.time() - t_start:.2f}s\n", flush=True)
        print(f"{CYAN}┌{'─' * 78}┐{RESET}")
        print(f"{CYAN}│{BOLD}{'Delta-Neutral Position Status':^78}{RESET}{CYAN}│{RESET}")
        print(f"{CYAN}├{'─' * 78}┤{RESET}")

        # Market Info
        print(f"{CYAN}│{RESET}  {BOLD}Market:{RESET} {YELLOW}{cfg.symbol}/{cfg.quote}{RESET}{'':65}  {CYAN}│{RESET}")
        if avg_mid:
            print(f"{CYAN}│{RESET}  Mid Price: ${avg_mid:>12.2f}{'':48}  {CYAN}│{RESET}")
        else:
            print(f"{CYAN}│{RESET}  Mid Price: {'N/A':>12}{'':48}  {CYAN}│{RESET}")

        # Display position open time and duration if available
        if position_open_date_str != "N/A":
            print(f"{CYAN}│{RESET}  {DIM}Opened:{RESET} {position_open_date_str}{'':38}  {CYAN}│{RESET}")
        if position_duration_str != "N/A":
            print(f"{CYAN}│{RESET}  {DIM}Duration:{RESET} {position_duration_str}{'':57}  {CYAN}│{RESET}")

        print(f"{CYAN}│{'':78}│{RESET}")
        print(f"{CYAN}├{'─' * 78}┤{RESET}")

        # EdgeX Position
        edgex_color = GREEN if edgex_direction == "LONG" else (RED if edgex_direction == "SHORT" else RESET)
        # Use configured leverage (API doesn't reliably return leverage for both exchanges)
        leverage = cfg.leverage
        print(f"{CYAN}│{RESET}  {BOLD}EdgeX Position{RESET} ({leverage}x leverage){'':44}  {CYAN}│{RESET}")
        print(f"{CYAN}│{RESET}    Size: {edgex_color}{edgex_size:>12.6f} {cfg.symbol}{RESET}{'':46}  {CYAN}│{RESET}")
        print(f"{CYAN}│{RESET}    Direction: {edgex_color}{edgex_direction:>12}{RESET}{'':46}  {CYAN}│{RESET}")
        if edgex_value is not None:
            print(f"{CYAN}│{RESET}    Notional: ${edgex_value:>12.2f}{'':48}  {CYAN}│{RESET}")
        if edgex_open_date_str != "N/A":
            print(f"{CYAN}│{RESET}    {DIM}Opened:{RESET} {edgex_open_date_str}{'':44}  {CYAN}│{RESET}")

        # Display PnL info for EdgeX
        if abs(edgex_size) > 1e-9:
            edgex_upnl = edgex_pos_details['unrealized_pnl']
            edgex_pnl_color = GREEN if edgex_upnl >= 0 else RED
            pnl_sign = "+" if edgex_upnl >= 0 else ""
            print(f"{CYAN}│{RESET}    Unrealized PnL: {edgex_pnl_color}{pnl_sign}${edgex_upnl:>10.2f}{RESET}{'':44}  {CYAN}│{RESET}")
            if edgex_pnl_pct is not None:
                pnl_pct_color = GREEN if edgex_pnl_pct >= 0 else RED
                pnl_pct_sign = "+" if edgex_pnl_pct >= 0 else ""
                print(f"{CYAN}│{RESET}    PnL%: {pnl_pct_color}{pnl_pct_sign}{edgex_pnl_pct:>10.2f}%{RESET}{'':51}  {CYAN}│{RESET}")

        print(f"{CYAN}│{'':78}│{RESET}")

        # Lighter Position
        lighter_color = GREEN if lighter_direction == "LONG" else (RED if lighter_direction == "SHORT" else RESET)
        print(f"{CYAN}│{RESET}  {BOLD}Lighter Position{RESET} ({leverage}x leverage){'':42}  {CYAN}│{RESET}")
        print(f"{CYAN}│{RESET}    Size: {lighter_color}{lighter_size:>12.6f} {cfg.symbol}{RESET}{'':46}  {CYAN}│{RESET}")
        print(f"{CYAN}│{RESET}    Direction: {lighter_color}{lighter_direction:>12}{RESET}{'':46}  {CYAN}│{RESET}")
        if lighter_value is not None:
            print(f"{CYAN}│{RESET}    Notional: ${lighter_value:>12.2f}{'':48}  {CYAN}│{RESET}")
        if lighter_open_date_str != "N/A":
            print(f"{CYAN}│{RESET}    {DIM}Opened:{RESET} {lighter_open_date_str}{'':44}  {CYAN}│{RESET}")

        # Display PnL info for Lighter
        if abs(lighter_size) > 1e-9:
            lighter_upnl = lighter_pos_details['unrealized_pnl']
            lighter_pnl_color = GREEN if lighter_upnl >= 0 else RED
            pnl_sign = "+" if lighter_upnl >= 0 else ""
            print(f"{CYAN}│{RESET}    Unrealized PnL: {lighter_pnl_color}{pnl_sign}${lighter_upnl:>10.2f}{RESET}{'':44}  {CYAN}│{RESET}")
            if lighter_pnl_pct is not None:
                pnl_pct_color = GREEN if lighter_pnl_pct >= 0 else RED
                pnl_pct_sign = "+" if lighter_pnl_pct >= 0 else ""
                print(f"{CYAN}│{RESET}    PnL%: {pnl_pct_color}{pnl_pct_sign}{lighter_pnl_pct:>10.2f}%{RESET}{'':51}  {CYAN}│{RESET}")

        print(f"{CYAN}│{'':78}│{RESET}")
        print(f"{CYAN}├{'─' * 78}┤{RESET}")

        # Hedge Status
        hedge_status_color = GREEN if is_hedged else RED
        quality_color = GREEN if hedge_quality in ["Excellent", "Good"] else (YELLOW if hedge_quality == "Fair" else RED)
        print(f"{CYAN}│{RESET}  {BOLD}Hedge Status{RESET}{'':64}  {CYAN}│{RESET}")
        print(f"{CYAN}│{RESET}    Hedged: {hedge_status_color}{('YES' if is_hedged else 'NO'):>12}{RESET}{'':50}  {CYAN}│{RESET}")
        print(f"{CYAN}│{RESET}    Quality: {quality_color}{hedge_quality:>12}{RESET}{'':49}  {CYAN}│{RESET}")
        print(f"{CYAN}│{RESET}    Net Exposure: {net_size:>12.6f} {cfg.symbol}{'':42}  {CYAN}│{RESET}")
        if net_value is not None:
            print(f"{CYAN}│{RESET}    Net Value: ${net_value:>12.2f}{'':48}  {CYAN}│{RESET}")

        # Display funding payments section if there are any payments
        if edgex_funding_count > 0 or lighter_funding_count > 0:
            print(f"{CYAN}│{'':78}│{RESET}")
            print(f"{CYAN}├{'─' * 78}┤{RESET}")
            print(f"{CYAN}│{RESET}  {BOLD}Funding Payments{RESET}{'':60}  {CYAN}│{RESET}")

            # EdgeX funding
            if edgex_funding_count > 0:
                edgex_funding_color = GREEN if edgex_funding >= 0 else RED
                funding_sign = "+" if edgex_funding >= 0 else ""
                print(f"{CYAN}│{RESET}    EdgeX: {edgex_funding_color}{funding_sign}${edgex_funding:>12.2f}{RESET} {DIM}({edgex_funding_count} payments){RESET}{'':36}  {CYAN}│{RESET}")
            else:
                print(f"{CYAN}│{RESET}    EdgeX: {DIM}No payments{RESET}{'':54}  {CYAN}│{RESET}")

            # Lighter funding
            if lighter_funding_count > 0:
                lighter_funding_color = GREEN if lighter_funding >= 0 else RED
                funding_sign = "+" if lighter_funding >= 0 else ""
                print(f"{CYAN}│{RESET}    Lighter: {lighter_funding_color}{funding_sign}${lighter_funding:>12.2f}{RESET} {DIM}({lighter_funding_count} payments){RESET}{'':34}  {CYAN}│{RESET}")
            else:
                print(f"{CYAN}│{RESET}    Lighter: {DIM}No payments{RESET}{'':52}  {CYAN}│{RESET}")

            print(f"{CYAN}│{'':78}│{RESET}")

            # Net funding
            total_funding_color = GREEN if total_funding > 0 else (RED if total_funding < 0 else RESET)
            total_sign = "+" if total_funding >= 0 else ""
            status_text = "Profit" if total_funding > 0 else ("Loss" if total_funding < 0 else "Break-even")
            status_color = GREEN if total_funding > 0 else (RED if total_funding < 0 else YELLOW)

            print(f"{CYAN}│{RESET}    {BOLD}Net Funding:{RESET} {total_funding_color}{total_sign}${total_funding:>12.2f}{RESET} {status_color}({status_text}){RESET}{'':36}  {CYAN}│{RESET}")

        # Display trading fees section
        if trading_fees_total > 0:
            print(f"{CYAN}│{'':78}│{RESET}")
            print(f"{CYAN}├{'─' * 78}┤{RESET}")
            print(f"{CYAN}│{RESET}  {BOLD}Trading Fees{RESET} {DIM}(0.036% per trade){RESET}{'':46}  {CYAN}│{RESET}")
            print(f"{CYAN}│{RESET}    Paid to Open: {RED}-${trading_fees_open:>12.2f}{RESET}{'':44}  {CYAN}│{RESET}")
            print(f"{CYAN}│{RESET}    Est. to Close: {DIM}-${trading_fees_open:>12.2f}{RESET}{'':43}  {CYAN}│{RESET}")
            print(f"{CYAN}│{RESET}    {BOLD}Total (Open+Close):{RESET} {RED}-${trading_fees_total:>12.2f}{RESET}{'':36}  {CYAN}│{RESET}")

        # Display net PnL section if we have both funding and fees
        if (edgex_funding_count > 0 or lighter_funding_count > 0) and trading_fees_total > 0:
            # Calculate net unrealized PnL from positions
            total_unrealized_pnl = 0.0
            if abs(edgex_size) > 1e-9:
                total_unrealized_pnl += edgex_pos_details['unrealized_pnl']
            if abs(lighter_size) > 1e-9:
                total_unrealized_pnl += lighter_pos_details['unrealized_pnl']

            # Net PnL = Funding + Unrealized PnL - Fees
            net_pnl_funding_only = total_funding - trading_fees_total
            net_pnl_total = total_funding + total_unrealized_pnl - trading_fees_total

            net_pnl_color = GREEN if net_pnl_funding_only > 0 else (RED if net_pnl_funding_only < 0 else YELLOW)
            net_pnl_sign = "+" if net_pnl_funding_only > 0 else ("-" if net_pnl_funding_only < 0 else "")
            pnl_status = "Profit" if net_pnl_funding_only > 0 else ("Loss" if net_pnl_funding_only < 0 else "Break-even")
            pnl_status_color = GREEN if net_pnl_funding_only > 0 else (RED if net_pnl_funding_only < 0 else YELLOW)

            net_pnl_total_color = GREEN if net_pnl_total > 0 else (RED if net_pnl_total < 0 else YELLOW)
            net_pnl_total_sign = "+" if net_pnl_total > 0 else ("-" if net_pnl_total < 0 else "")
            pnl_total_status = "Profit" if net_pnl_total > 0 else ("Loss" if net_pnl_total < 0 else "Break-even")
            pnl_total_status_color = GREEN if net_pnl_total > 0 else (RED if net_pnl_total < 0 else YELLOW)

            print(f"{CYAN}│{'':78}│{RESET}")
            print(f"{CYAN}├{'─' * 78}┤{RESET}")
            print(f"{CYAN}│{RESET}  {BOLD}Net P&L{RESET}{'':69}  {CYAN}│{RESET}")
            print(f"{CYAN}│{RESET}    {DIM}Funding - Fees:{RESET} {net_pnl_color}{net_pnl_sign}${abs(net_pnl_funding_only):>12.2f}{RESET} {pnl_status_color}({pnl_status}){RESET}{'':30}  {CYAN}│{RESET}")
            print(f"{CYAN}│{RESET}    {BOLD}Total (incl. Unrealized PnL):{RESET} {net_pnl_total_color}{net_pnl_total_sign}${abs(net_pnl_total):>12.2f}{RESET} {pnl_total_status_color}({pnl_total_status}){RESET}{'':16}  {CYAN}│{RESET}")

        print(f"{CYAN}└{'─' * 78}┘{RESET}", flush=True)

        if not is_hedged and (abs(edgex_size) > 1e-9 or abs(lighter_size) > 1e-9):
            print(f"\n⚠️  WARNING: Positions are not properly hedged!")
            if abs(edgex_size) > 1e-9 and abs(lighter_size) < 1e-9:
                print(f"    You have a {edgex_direction} position on EdgeX with no offsetting Lighter position.")
            elif abs(lighter_size) > 1e-9 and abs(edgex_size) < 1e-9:
                print(f"    You have a {lighter_direction} position on Lighter with no offsetting EdgeX position.")
            elif (edgex_size > 0 and lighter_size > 0) or (edgex_size < 0 and lighter_size < 0):
                print(f"    Both positions are in the same direction ({edgex_direction})!")

        if is_hedged and hedge_quality in ["Fair", "Poor"]:
            size_diff = abs(abs(edgex_size) - abs(lighter_size))
            print(f"\n⚠️  WARNING: Hedge is imbalanced by {size_diff:.6f} {cfg.symbol}")
            print(f"    Consider rebalancing your positions.")

    finally:
        # Close clients (this may take a few seconds for WebSocket cleanup)
        print("\n\033[2m⏳ Closing connections...\033[0m", flush=True)
        t_cleanup = time.time()
        await ex_client.close()
        await api_client.close()
        await signer.close()
        cleanup_time = time.time() - t_cleanup
        print(f"\033[2m✓ Cleanup complete ({cleanup_time:.2f}s)\033[0m", flush=True)


# ---------- Config parsing ----------
def load_config(path: str) -> AppConfig:
    with open(path, "r") as f:
        raw = json.load(f)

    def must(k, obj=raw):
        if k not in obj:
            raise SystemExit(f"Missing required config key: {k}")
        return obj[k]

    return AppConfig(
        symbol=str(must("symbol")).upper(),
        long_exchange=str(must("long_exchange")).lower(),
        short_exchange=str(must("short_exchange")).lower(),
        leverage=float(must("leverage")),
        quote=str(raw.get("quote", "USD")).upper(),
        notional=float(raw.get("notional", 40.0))
    )

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Open/close a cross-venue hedge (Lighter + EdgeX).")
    parser.add_argument("--config", default="hedge_config.json", help="Path to JSON config.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_cap = sub.add_parser("capacity", help="Show available capital on both exchanges and max delta-neutral base size.")

    p_open = sub.add_parser("open", help="Open a hedge (one long, one short). Uses notional from config by default.")
    p_open.add_argument("--cross-ticks", type=int, default=100, help="How many ticks beyond the spread to cross for immediate fill.")
    g = p_open.add_mutually_exclusive_group(required=False)
    g.add_argument("--size-base", type=float, help="Order size in BASE units (e.g., PAXG). Overrides config notional.")
    g.add_argument("--size-quote", type=float, help="Order notional in QUOTE (e.g., USD). Overrides config notional.")

    p_close = sub.add_parser("close", help="Close both positions.")
    p_close.add_argument("--cross-ticks", type=int, default=100, help="How many ticks beyond the spread to cross for immediate fill.")

    p_fund = sub.add_parser("funding", help="Show current and historical funding rates.")

    p_fund_all = sub.add_parser("funding_all", help="Show funding rates for multiple markets.")
    p_fund_all.add_argument("--symbols", nargs="+", default=CRYPTO_LIST,
                           help="List of symbols to check")

    p_leverage = sub.add_parser("check_leverage", help="Show current and maximum leverage for multiple markets.")
    p_leverage.add_argument("--symbols", nargs="+", default=CRYPTO_LIST,
                           help="List of symbols to check")

    p_test = sub.add_parser("test", help="Test delta-neutral hedge with small position (opens then closes).")
    p_test.add_argument("--notional", type=float, default=20.0, help="Notional size in USD for the test (default: 20)")
    p_test.add_argument("--cross-ticks", type=int, default=100, help="How many ticks beyond the spread to cross for immediate fill.")

    p_test_auto = sub.add_parser("test_auto", help="Run a fully automated test (open, wait 2m, close).")
    p_test_auto.add_argument("--notional", type=float, default=20.0, help="Notional size in USD for the test (default: 20)")
    p_test_auto.add_argument("--cross-ticks", type=int, default=100, help="How many ticks beyond the spread to cross for immediate fill.")

    p_test_lev = sub.add_parser("test_leverage", help="Test leverage setup on both exchanges (no trading).")

    p_status = sub.add_parser("status", help="Check current delta-neutral position status across both exchanges.")

    args = parser.parse_args()
    cfg = load_config(args.config)
    env = load_env()

    if args.cmd == "open":
        size_base = args.size_base
        size_quote = args.size_quote
        if size_base is None and size_quote is None:
            size_quote = cfg.notional  # Use notional from config as default
            print(f"Using default notional size from config: {size_quote} {cfg.quote}")
        asyncio.run(open_hedge(cfg, env, size_base, size_quote, cross_ticks=args.cross_ticks))

    elif args.cmd == "capacity":
        info = asyncio.run(compute_max_delta_neutral_size(cfg, env))

        # Formatted output
        print(f"\n┌───────────────────── Delta-Neutral Capacity ─────────────────────┐")
        print(f"│  Market: {info['symbol']}/{info['quote']:>4}                                              │")
        print(f"│  Mid Price:                              ${info['mid_price']:>12.2f}       │")
        print(f"│                                                                      │")
        print(f"│  Exchange Capacities:                                                │")
        print(f"│    Lighter Available USD:                ${info['lighter_available_usd']:>12.2f}       │")
        print(f"│    Lighter Capacity (base):              {info['lighter_capacity_base']:>12.6f}       │")
        print(f"│                                                                      │")
        print(f"│    EdgeX Available USD:                  ${info['edgex_available_usd']:>12.2f}       │")
        print(f"│    EdgeX Capacity (base):                {info['edgex_capacity_base']:>12.6f}       │")
        print(f"│                                                                      │")
        print(f"│  Hedge Configuration:                                                │")
        print(f"│    Long Exchange:                        {info['long_exchange'].capitalize():>12}       │")
        print(f"│    Short Exchange:                       {info['short_exchange'].capitalize():>12}       │")
        print(f"│                                                                      │")
        print(f"│  Max Delta-Neutral Size:                 {info['max_delta_neutral_base']:>12.6f}       │")
        print(f"└──────────────────────────────────────────────────────────────────────┘")
        return
    elif args.cmd == "close":
        asyncio.run(close_both(cfg, env, cross_ticks=args.cross_ticks))
    elif args.cmd == "funding":
        optimal_long, optimal_short = asyncio.run(show_funding_rates(cfg, env))

        # Update config file if recommendation differs from current config
        if optimal_long and optimal_short:
            print(f"\nUpdating {args.config} with optimal configuration...")

            # Read current config
            with open(args.config, 'r') as f:
                config_data = json.load(f)

            # Update exchanges
            config_data['long_exchange'] = optimal_long.lower()
            config_data['short_exchange'] = optimal_short.lower()

            # Write back to file
            with open(args.config, 'w') as f:
                json.dump(config_data, f, indent=2)

            print(f"✓ Updated config: LONG={optimal_long}, SHORT={optimal_short}")
    elif args.cmd == "funding_all":
        asyncio.run(show_funding_rates_multi(args.symbols, cfg.quote, env))
    elif args.cmd == "check_leverage":
        asyncio.run(show_leverage_info(args.symbols, cfg.quote, env))
    elif args.cmd == "test":
        asyncio.run(test_hedge(cfg, env, notional_usd=args.notional, cross_ticks=args.cross_ticks))
    elif args.cmd == "test_auto":
        asyncio.run(test_auto_hedge(cfg, env, notional_usd=args.notional, cross_ticks=args.cross_ticks))
    elif args.cmd == "test_leverage":
        asyncio.run(test_leverage_setup(cfg, env))
    elif args.cmd == "status":
        try:
            asyncio.run(check_position_status(cfg, env))
        finally:
            # Cancel all remaining tasks to prevent hanging
            try:
                loop = asyncio.get_event_loop()
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
            except RuntimeError:
                pass  # Loop already closed
            # Force immediate exit to avoid lingering network connections
            import sys
            sys.exit(0)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
