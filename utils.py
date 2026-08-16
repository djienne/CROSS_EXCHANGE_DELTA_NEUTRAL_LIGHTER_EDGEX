"""
utils.py
--------
Common utilities shared across all modules in the delta-neutral hedging system.

Contains:
- Rounding functions (tick-aware decimal arithmetic)
- Environment loading (EdgeX + Lighter credentials)
- DateTime utilities (timezone-aware UTC)
- Math utilities (APR calculations, mid price averaging)
- ANSI color codes for console output
"""

import os
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from datetime import datetime, timezone
from typing import Optional, Dict, List

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ==================== ANSI Color Codes ====================

class Colors:
    """ANSI color codes for console output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'


# ==================== Rounding Functions ====================

def _round_to_tick(value: float, tick: float) -> float:
    """Round `value` to the nearest multiple of `tick` (banker's rounding)."""
    if not tick or tick <= 0:
        return value
    d_value = Decimal(str(value))
    d_tick = Decimal(str(tick))
    return float((d_value / d_tick).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * d_tick)


def _ceil_to_tick(value: float, tick: float) -> float:
    """Round `value` up to the nearest multiple of `tick`."""
    if not tick or tick <= 0:
        return value
    d_value = Decimal(str(value))
    d_tick = Decimal(str(tick))
    return float((d_value / d_tick).quantize(Decimal('1'), rounding=ROUND_UP) * d_tick)


def _floor_to_tick(value: float, tick: float) -> float:
    """Round `value` down to the nearest multiple of `tick`."""
    if not tick or tick <= 0:
        return value
    d_value = Decimal(str(value))
    d_tick = Decimal(str(tick))
    return float((d_value / d_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * d_tick)


# ==================== Environment Loading ====================

def load_env() -> dict:
    """
    Load required environment variables for both exchanges.

    Returns dict with keys:
        EdgeX: EDGEX_BASE_URL, EDGEX_WS_URL, EDGEX_ACCOUNT_ID, EDGEX_STARK_PRIVATE_KEY
        Lighter: LIGHTER_BASE_URL, LIGHTER_WS_URL, API_KEY_PRIVATE_KEY, ACCOUNT_INDEX, API_KEY_INDEX, MARGIN_MODE

    Note: EDGEX_ACCOUNT_ID is returned as string; callers must convert to int() for EdgeX SDK.
    """
    load_dotenv()
    env: Dict[str, object] = {}

    # EdgeX
    env["EDGEX_BASE_URL"] = os.getenv("EDGEX_BASE_URL", "https://pro.edgex.exchange")
    env["EDGEX_WS_URL"] = os.getenv("EDGEX_WS_URL", "wss://quote.edgex.exchange")
    env["EDGEX_ACCOUNT_ID"] = os.getenv("EDGEX_ACCOUNT_ID")
    env["EDGEX_STARK_PRIVATE_KEY"] = os.getenv("EDGEX_STARK_PRIVATE_KEY")

    # Lighter (support both LIGHTER_* and legacy names)
    env["LIGHTER_BASE_URL"] = os.getenv("LIGHTER_BASE_URL", os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai"))
    env["LIGHTER_WS_URL"] = os.getenv("LIGHTER_WS_URL", os.getenv("WEBSOCKET_URL", "wss://mainnet.zklighter.elliot.ai/stream"))
    env["API_KEY_PRIVATE_KEY"] = os.getenv("API_KEY_PRIVATE_KEY") or os.getenv("LIGHTER_PRIVATE_KEY")
    env["ACCOUNT_INDEX"] = int(os.getenv("ACCOUNT_INDEX", os.getenv("LIGHTER_ACCOUNT_INDEX", "0")))
    env["API_KEY_INDEX"] = int(os.getenv("API_KEY_INDEX", os.getenv("LIGHTER_API_KEY_INDEX", "0")))
    env["MARGIN_MODE"] = "cross"  # Always cross margin for delta-neutral hedging

    missing = [key for key in ("EDGEX_ACCOUNT_ID", "EDGEX_STARK_PRIVATE_KEY", "API_KEY_PRIVATE_KEY") if not env.get(key)]
    if missing:
        logger.warning("Missing env vars: %s. Trading may fail.", missing)

    return env


# ==================== DateTime Utilities ====================

def utc_now() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return an ISO 8601 timestamp suffixed with Z for UTC."""
    return utc_now().isoformat().replace("+00:00", "Z")


def to_iso_z(dt_obj: datetime) -> str:
    """Convert datetime to ISO string with Z suffix, adding UTC if naive."""
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    return dt_obj.isoformat().replace("+00:00", "Z")


def from_iso_z(iso_string: str) -> datetime:
    """
    Parse ISO timestamp with Z or +00:00 suffix.

    Handles malformed formats like '...+00:00Z' or '...+00:00+00:00'.
    """
    # Clean up by removing 'Z' first, then any duplicate timezone info
    cleaned = iso_string.rstrip('Z')
    # If we have duplicate +00:00, keep only the first one
    if cleaned.count('+00:00') > 1:
        parts = cleaned.split('+00:00')
        cleaned = parts[0] + '+00:00'
    # If we don't have timezone info, add it
    elif not ('+' in cleaned or '-' in cleaned[-6:]):
        cleaned = cleaned + '+00:00'

    return datetime.fromisoformat(cleaned)


# ==================== Math Utilities ====================

def compute_base_size_from_quote(avg_mid: float, size_quote: float) -> float:
    """Convert quote notional into base size using the average mid price."""
    if avg_mid <= 0:
        raise ValueError("Invalid mid price to compute base size.")
    return size_quote / avg_mid


def get_avg_mid(
    lighter_bid: Optional[float],
    lighter_ask: Optional[float],
    edgex_bid: Optional[float],
    edgex_ask: Optional[float],
) -> float:
    """
    Average mid price between both exchanges, falling back gracefully.

    Tries multiple combinations to ensure a valid mid price is returned.
    Raises RuntimeError if no usable prices are available.
    """
    mids: List[float] = []
    if lighter_bid and lighter_ask:
        mids.append((lighter_bid + lighter_ask) / 2.0)
    if edgex_bid and edgex_ask:
        mids.append((edgex_bid + edgex_ask) / 2.0)

    if mids:
        return sum(mids) / len(mids)

    # Fallback combinations
    if lighter_bid and lighter_ask:
        return (lighter_bid + lighter_ask) / 2.0
    if edgex_bid and edgex_ask:
        return (edgex_bid + edgex_ask) / 2.0
    if lighter_bid and edgex_ask:
        return (lighter_bid + edgex_ask) / 2.0
    if edgex_bid and lighter_ask:
        return (edgex_bid + lighter_ask) / 2.0

    raise RuntimeError("No usable prices from either venue.")


def _calculate_apr(rate: float, periods_per_day: int) -> float:
    """
    Convert a per-period funding rate (decimal form) into annualized percentage.

    Args:
        rate: Funding rate as decimal (e.g., 0.0001 for 0.01%)
        periods_per_day: Number of funding periods per day. Do NOT hardcode this —
            resolve it per venue/contract at runtime.

            - EdgeX: 6. Verified: the metadata API reports
              `fundingRateIntervalMin: "240"` (4h) for BTCUSDT and ETHUSDT, and a
              recorded sample shows fundingTime -> nextFundingTime deltas of exactly 4h.
              It is configurable per contract ("must be an integer multiple of 60
              minutes"), so read `fundingRateIntervalMin` rather than assuming 6.
              NOTE: EdgeX's prose docs claim hourly funding. They contradict its own
              API and the recorded data. Trust the API field.

            - Lighter: 24. RESOLVED EMPIRICALLY against live data.

              Lighter's /api/v1/funding-rates is a CROSS-VENUE endpoint: one response
              carries binance, bybit, hyperliquid and lighter rows together. Comparing
              Lighter to Hyperliquid — whose convention is established beyond doubt as
              hourly settlement with the rate as a DECIMAL — over 98 same-sign common
              symbols gave a median ratio of 0.9600, with dozens of pairs sitting at
              exactly 0.000096 (Lighter) vs 0.00010 (Hyperliquid).

              Identical order of magnitude across a hundred symbols is only possible
              if both the unit and the period match. So Lighter is HOURLY and DECIMAL,
              and the *100 below is correct.

              This settles two previously contradictory claims in this codebase:
                * lighter_edgex_hedge.py passed periods_per_day=3, understating every
                  Lighter APR by exactly 8x and mis-ranking every cross-venue spread.
                * examples/hedge_cli.py asserted "Lighter API returns rate already as
                  percentage" and omitted the *100, understating by 100x. That comment
                  is wrong; the rate is a decimal.

    Returns:
        Annualized percentage rate (e.g., 36.5 for 36.5% APR)
    """
    return rate * periods_per_day * 365 * 100.0
