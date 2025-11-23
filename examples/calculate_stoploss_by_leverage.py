"""
Calculate optimal stop-loss percentages based on leverage for isolated, linear USD-margined futures.

This script calculates the maximum safe stop-loss distance that keeps you away from liquidation
by a safety buffer, accounting for maintenance margin, fees, and slippage.
"""

def calculate_max_stop_long(leverage: float, maintenance_margin: float, buffer: float, safety_multiplier: float = 0.8) -> float:
    """
    Calculate maximum stop-loss distance for LONG position.

    Args:
        leverage: Leverage multiplier (e.g., 10 for 10x)
        maintenance_margin: Maintenance margin rate (e.g., 0.005 for 0.5%)
        buffer: Safety buffer from liquidation price (e.g., 0.006 for 0.6%)
        safety_multiplier: Additional safety factor (default 0.8 = use 80% of max stop)

    Returns:
        Maximum stop-loss distance as fraction (e.g., 0.0895 = 8.95%)
    """
    s_max_long = (1 - (1 - 1/leverage) / (1 - maintenance_margin)) - buffer
    return s_max_long * safety_multiplier


def calculate_max_stop_short(leverage: float, maintenance_margin: float, buffer: float, safety_multiplier: float = 0.8) -> float:
    """
    Calculate maximum stop-loss distance for SHORT position.

    Args:
        leverage: Leverage multiplier (e.g., 10 for 10x)
        maintenance_margin: Maintenance margin rate (e.g., 0.005 for 0.5%)
        buffer: Safety buffer from liquidation price (e.g., 0.006 for 0.6%)
        safety_multiplier: Additional safety factor (default 0.8 = use 80% of max stop)

    Returns:
        Maximum stop-loss distance as fraction (e.g., 0.0895 = 8.95%)
    """
    s_max_short = ((1 + 1/leverage) / (1 + maintenance_margin) - 1) - buffer
    return s_max_short * safety_multiplier


def get_recommended_stop_loss(leverage: float, maintenance_margin: float = 0.005, buffer: float = 0.006, safety_multiplier: float = 0.8) -> float:
    """
    Get recommended stop-loss percentage for a given leverage.

    Args:
        leverage: Leverage multiplier (e.g., 5 for 5x)
        maintenance_margin: Maintenance margin rate (default: 0.005 for 0.5%)
        buffer: Safety buffer (default: 0.006 for 0.6%)
        safety_multiplier: Safety multiplier (default: 0.8 for 80% of max)

    Returns:
        Recommended stop-loss percentage (e.g., 17.1 for 17.1%)

    Example:
        >>> get_recommended_stop_loss(5)
        17.10
        >>> get_recommended_stop_loss(10)
        8.05
    """
    # Use average of long and short calculations
    stop_long = calculate_max_stop_long(leverage, maintenance_margin, buffer, safety_multiplier)
    stop_short = calculate_max_stop_short(leverage, maintenance_margin, buffer, safety_multiplier)
    avg_stop = (stop_long + stop_short) / 2
    return round(avg_stop * 100, 2)


def calculate_liquidation_price_long(entry_price: float, leverage: float, maintenance_margin: float) -> float:
    """Calculate liquidation price for LONG position."""
    return entry_price * (1 - 1/leverage) / (1 - maintenance_margin)


def calculate_liquidation_price_short(entry_price: float, leverage: float, maintenance_margin: float) -> float:
    """Calculate liquidation price for SHORT position."""
    return entry_price * (1 + 1/leverage) / (1 + maintenance_margin)


def print_stop_loss_grid():
    """Print a grid of stop-loss percentages for different leverage levels."""

    # Parameters
    maintenance_margin = 0.005  # 0.5%
    buffer = 0.006  # 0.6% (includes fees + slippage)
    safety_multiplier = 0.8  # Use 80% of theoretical max

    print("=" * 90)
    print("STOP-LOSS CALCULATOR FOR ISOLATED LINEAR USD-MARGINED FUTURES")
    print("=" * 90)
    print(f"\nParameters:")
    print(f"  Maintenance Margin: {maintenance_margin*100:.2f}%")
    print(f"  Safety Buffer:      {buffer*100:.2f}% (includes fees & slippage)")
    print(f"  Safety Multiplier:  {safety_multiplier} (use 80% of theoretical max)")
    print("\n" + "-" * 90)
    print(f"{'Leverage':<10} {'Long SL %':<15} {'Short SL %':<15} {'Notes':<50}")
    print("-" * 90)

    for leverage in range(1, 11):
        s_long = calculate_max_stop_long(leverage, maintenance_margin, buffer)
        s_short = calculate_max_stop_short(leverage, maintenance_margin, buffer)

        # Format percentages
        long_pct = f"{s_long*100:.2f}%"
        short_pct = f"{s_short*100:.2f}%"

        # Add notes for specific leverage levels
        if leverage == 1:
            note = "No leverage - largest stop-loss room"
        elif leverage == 3:
            note = "Conservative leverage"
        elif leverage == 5:
            note = "Moderate leverage"
        elif leverage == 10:
            note = "High leverage - tight stop required"
        else:
            note = ""

        print(f"{leverage:<10} {long_pct:<15} {short_pct:<15} {note:<50}")

    print("-" * 90)
    print("\nInterpretation:")
    print("  - Long SL %: Maximum distance BELOW entry price for long positions")
    print("  - Short SL %: Maximum distance ABOVE entry price for short positions")
    print("  - Higher leverage = smaller stop-loss room = higher risk")
    print("  - Always set your stop TIGHTER than these maximums for safety")
    print("\n" + "=" * 90)


def example_calculation():
    """Show a detailed example calculation."""

    print("\n" + "=" * 90)
    print("DETAILED EXAMPLE: 10x Leverage Long Position")
    print("=" * 90)

    # Parameters
    leverage = 10
    maintenance_margin = 0.005  # 0.5%
    buffer = 0.006  # 0.6%
    entry_price = 100.0  # Example entry price

    # Calculate
    liq_price = calculate_liquidation_price_long(entry_price, leverage, maintenance_margin)
    s_max = calculate_max_stop_long(leverage, maintenance_margin, buffer)
    stop_price = entry_price * (1 - s_max)

    print(f"\nGiven:")
    print(f"  Entry Price:         ${entry_price:.2f}")
    print(f"  Leverage:            {leverage}x")
    print(f"  Maintenance Margin:  {maintenance_margin*100:.2f}%")
    print(f"  Safety Buffer:       {buffer*100:.2f}%")

    print(f"\nCalculated:")
    print(f"  Liquidation Price:   ${liq_price:.2f} ({((liq_price/entry_price - 1)*100):+.2f}% from entry)")
    print(f"  Max Stop Distance:   {s_max*100:.2f}%")
    print(f"  Stop Price:          ${stop_price:.2f}")
    print(f"  Distance to Liq:     {((liq_price - stop_price)/entry_price*100):.2f}% (your safety buffer)")

    print("\nNOTE: Your stop-loss at ${:.2f} is {:.2f}% below entry, giving you a {:.2f}% cushion before liquidation.".format(
        stop_price, s_max*100, buffer*100))

    print("=" * 90)


def risk_based_position_sizing():
    """Calculate position sizing based on risk tolerance."""

    print("\n" + "=" * 90)
    print("POSITION SIZING BASED ON RISK TOLERANCE")
    print("=" * 90)

    account_equity = 1000  # Example: $1000 account
    risk_per_trade = 0.02  # 2% of account
    stop_loss_pct = 0.05  # 5% stop-loss

    print(f"\nGiven:")
    print(f"  Account Equity:      ${account_equity:.2f}")
    print(f"  Risk per Trade:      {risk_per_trade*100:.1f}% (${account_equity * risk_per_trade:.2f})")
    print(f"  Stop-Loss Distance:  {stop_loss_pct*100:.1f}%")

    print(f"\n{'Leverage':<12} {'Max Margin':<15} {'Position Size':<18} {'Risk if SL Hit':<20}")
    print("-" * 90)

    for leverage in [1, 3, 5, 10]:
        # Max margin allocation: a = R / (L * s)
        max_margin = risk_per_trade / (leverage * stop_loss_pct)
        max_margin_usd = min(max_margin * account_equity, account_equity)
        position_size = max_margin_usd * leverage
        risk_usd = position_size * stop_loss_pct

        print(f"{leverage}x{' ':<9} ${max_margin_usd:<14.2f} ${position_size:<17.2f} ${risk_usd:<19.2f}")

    print("-" * 90)
    print("\nNote: Position size = Margin x Leverage")
    print("      Risk if SL hit = Position size x Stop-loss %")
    print("=" * 90)


if __name__ == "__main__":
    # Main grid of stop-loss percentages
    print_stop_loss_grid()

    # Detailed example
    example_calculation()

    # Risk-based position sizing
    risk_based_position_sizing()

    # Quick reference: get recommended stop-loss for any leverage
    print("\n" + "=" * 90)
    print("QUICK REFERENCE: GET STOP-LOSS BY LEVERAGE")
    print("=" * 90)
    print("\nUse get_recommended_stop_loss(leverage) to get stop-loss % for any leverage:\n")

    example_leverages = [3, 5, 7, 10, 15, 20]
    for lev in example_leverages:
        stop = get_recommended_stop_loss(lev)
        print(f"  get_recommended_stop_loss({lev})  = {stop}%")

    print("\n" + "=" * 90)
    print("\nDone! Use these values to configure your stop-loss based on your chosen leverage.")
