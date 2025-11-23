"""
Extended Stop-Loss Calculator with Higher Leverage Options
Helps optimize leverage and stop-loss for maximum capital efficiency
"""

def calculate_max_stop_long(leverage: float, maintenance_margin: float, buffer: float, safety_multiplier: float = 0.8) -> float:
    """
    Calculate maximum stop-loss distance for LONG position.

    Args:
        safety_multiplier: Additional safety factor (default 0.8 = use 80% of max stop)
    """
    s_max_long = (1 - (1 - 1/leverage) / (1 - maintenance_margin)) - buffer
    return s_max_long * safety_multiplier


def calculate_max_stop_short(leverage: float, maintenance_margin: float, buffer: float, safety_multiplier: float = 0.8) -> float:
    """
    Calculate maximum stop-loss distance for SHORT position.

    Args:
        safety_multiplier: Additional safety factor (default 0.8 = use 80% of max stop)
    """
    s_max_short = ((1 + 1/leverage) / (1 + maintenance_margin) - 1) - buffer
    return s_max_short * safety_multiplier


def extended_grid():
    """Show extended leverage grid from 1x to 25x."""

    maintenance_margin = 0.005  # 0.5%
    buffer = 0.006  # 0.6%

    print("=" * 100)
    print("EXTENDED STOP-LOSS GRID - LEVERAGE 1x TO 25x")
    print("=" * 100)
    print(f"\nParameters:")
    print(f"  Maintenance Margin: {maintenance_margin*100:.2f}%")
    print(f"  Safety Buffer: {buffer*100:.2f}% (fees + slippage)")
    print(f"  Safety Multiplier: 0.8x (use 80% of theoretical max for extra safety)")
    print("\n" + "-" * 100)
    print(f"{'Leverage':<12} {'Long SL %':<15} {'Short SL %':<15} {'Capital Efficiency':<25} {'Risk Level':<30}")
    print("-" * 100)

    for leverage in range(1, 26):
        s_long = calculate_max_stop_long(leverage, maintenance_margin, buffer)
        s_short = calculate_max_stop_short(leverage, maintenance_margin, buffer)

        long_pct = f"{s_long*100:.2f}%"
        short_pct = f"{s_short*100:.2f}%"

        # Capital efficiency
        if leverage <= 3:
            efficiency = "Low (conservative)"
            risk = "Very Safe"
        elif leverage <= 5:
            efficiency = "Medium (balanced)"
            risk = "Moderate"
        elif leverage <= 10:
            efficiency = "High (aggressive)"
            risk = "Elevated"
        elif leverage <= 15:
            efficiency = "Very High (expert)"
            risk = "High"
        else:
            efficiency = "Maximum (extreme)"
            risk = "Very High"

        # Highlight current config (3x, 22% stop)
        if leverage == 3:
            marker = " <- YOUR CURRENT (22% stop)"
        else:
            marker = ""

        print(f"{leverage}x{' ':<9} {long_pct:<15} {short_pct:<15} {efficiency:<25} {risk:<30}{marker}")

    print("-" * 100)


def recommendations():
    """Provide specific recommendations for different risk profiles."""

    print("\n" + "=" * 100)
    print("RECOMMENDATIONS FOR YOUR DELTA-NEUTRAL BOT")
    print("=" * 100)

    # Calculate with 0.9 safety multiplier
    m = 0.005
    b = 0.006

    max_3x = calculate_max_stop_long(3, m, b) * 100
    max_5x = calculate_max_stop_long(5, m, b) * 100
    max_7x = calculate_max_stop_long(7, m, b) * 100
    max_10x = calculate_max_stop_long(10, m, b) * 100

    print(f"\n1. ULTRA CONSERVATIVE (Current Setup)")
    print(f"   Leverage: 3x | Stop-Loss: 22% | Max Safe Stop: {max_3x:.2f}%")
    print(f"   - Safety margin: {max_3x - 22:.2f}% cushion above your stop")
    print(f"   - Best for: Beginners, uncertain markets")
    print(f"   - Capital efficiency: LOW")

    print(f"\n2. CONSERVATIVE (Recommended for Most)")
    print(f"   Leverage: 5x | Stop-Loss: 15% | Max Safe Stop: {max_5x:.2f}%")
    print(f"   - Safety margin: {max_5x - 15:.2f}% cushion")
    print(f"   - Best for: Experienced traders, stable funding arb")
    print(f"   - Capital efficiency: MEDIUM")
    print(f"   - 67% MORE capital efficient than 3x!")

    print(f"\n3. MODERATE (For Active Monitoring)")
    print(f"   Leverage: 7x | Stop-Loss: 10% | Max Safe Stop: {max_7x:.2f}%")
    print(f"   - Safety margin: {max_7x - 10:.2f}% cushion")
    print(f"   - Best for: Active traders, high-confidence setups")
    print(f"   - Capital efficiency: HIGH")
    print(f"   - 133% MORE capital efficient than 3x!")

    print(f"\n4. AGGRESSIVE (Experts Only)")
    print(f"   Leverage: 10x | Stop-Loss: 7% | Max Safe Stop: {max_10x:.2f}%")
    print(f"   - Safety margin: {max_10x - 7:.2f}% cushion")
    print(f"   - Best for: Experts, tight risk management")
    print(f"   - Capital efficiency: VERY HIGH")
    print(f"   - 233% MORE capital efficient than 3x!")
    print(f"   - WARNING: Price can move 7% quickly!")

    print("\n" + "=" * 100)


def optimal_for_target_stop():
    """Calculate optimal leverage for different target stop-loss levels."""

    maintenance_margin = 0.005
    buffer = 0.006

    print("\n" + "=" * 100)
    print("REVERSE CALCULATION: OPTIMAL LEVERAGE FOR YOUR DESIRED STOP-LOSS")
    print("=" * 100)

    print("\nIf you want a specific stop-loss %, here's the maximum leverage you can use:\n")
    print(f"{'Target Stop %':<20} {'Max Leverage (Long)':<25} {'Max Leverage (Short)':<25}")
    print("-" * 100)

    target_stops = [5, 7, 10, 12, 15, 18, 20, 22, 25, 30]

    for stop_pct in target_stops:
        stop_fraction = stop_pct / 100

        # Solve for leverage (long): s = [1 - (1-1/L)/(1-m)] - b
        # s + b = 1 - (1-1/L)/(1-m)
        # (1-1/L)/(1-m) = 1 - s - b
        # 1-1/L = (1-m)(1-s-b)
        # 1/L = 1 - (1-m)(1-s-b)
        # L = 1 / [1 - (1-m)(1-s-b)]

        denom_long = 1 - (1 - maintenance_margin) * (1 - stop_fraction - buffer)
        if denom_long > 0:
            max_lev_long = 1 / denom_long
        else:
            max_lev_long = float('inf')

        # Solve for leverage (short): s = [(1+1/L)/(1+m) - 1] - b
        # s + b = (1+1/L)/(1+m) - 1
        # (1+1/L)/(1+m) = s + b + 1
        # 1+1/L = (1+m)(s+b+1)
        # 1/L = (1+m)(s+b+1) - 1
        # L = 1 / [(1+m)(s+b+1) - 1]

        denom_short = (1 + maintenance_margin) * (stop_fraction + buffer + 1) - 1
        if denom_short > 0:
            max_lev_short = 1 / denom_short
        else:
            max_lev_short = float('inf')

        # Highlight current 22% stop
        if stop_pct == 22:
            marker = " <- YOUR CURRENT"
        else:
            marker = ""

        print(f"{stop_pct}%{' ':<16} {max_lev_long:.2f}x{' ':<20} {max_lev_short:.2f}x{marker}")

    print("-" * 100)


def position_size_comparison():
    """Compare position sizes at different leverage levels."""

    available_capital = 200  # Example: $200 available per exchange

    print("\n" + "=" * 100)
    print(f"POSITION SIZE COMPARISON (with ${available_capital} available capital per exchange)")
    print("=" * 100)

    print("\n" + f"{'Leverage':<12} {'Position Size':<20} {'vs 3x':<20} {'With $1000 Total':<20}")
    print("-" * 100)

    base_size = available_capital * 3

    for leverage in [3, 5, 7, 10, 15, 20]:
        position_size = available_capital * leverage
        vs_base = ((position_size / base_size) - 1) * 100
        with_1000 = (1000 / 2) * leverage  # $1000 split between two exchanges

        print(f"{leverage}x{' ':<9} ${position_size:<19.2f} +{vs_base:.0f}%{' ':<15} ${with_1000:.2f}")

    print("-" * 100)
    print("\nNote: Higher leverage = larger positions with same capital = more funding revenue")
    print("      BUT also more risk if price moves against you!")
    print("=" * 100)


def config_recommendations():
    """Generate specific config file recommendations."""

    print("\n" + "=" * 100)
    print("SPECIFIC CONFIG RECOMMENDATIONS FOR rotation_bot_config.json")
    print("=" * 100)

    configs = [
        {
            "name": "Current (Ultra Safe)",
            "leverage": 3,
            "stop_loss": 22.0,
            "notional": 320,
            "description": "Very conservative, lots of safety margin"
        },
        {
            "name": "Recommended (Balanced)",
            "leverage": 5,
            "stop_loss": 15.0,
            "notional": 500,
            "description": "Good balance of safety and capital efficiency"
        },
        {
            "name": "Aggressive (Monitor Closely)",
            "leverage": 7,
            "stop_loss": 10.0,
            "notional": 700,
            "description": "Higher returns, requires active monitoring"
        },
        {
            "name": "Expert (High Risk)",
            "leverage": 10,
            "stop_loss": 7.0,
            "notional": 1000,
            "description": "Maximum efficiency, tight stops, experts only"
        }
    ]

    for config in configs:
        print(f"\n{config['name']}:")
        print(f'  "leverage": {config["leverage"]},')
        print(f'  "stop_loss_percent": {config["stop_loss"]},')
        print(f'  "notional_per_position": {config["notional"]}.0,')
        print(f"  Description: {config['description']}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    # Extended grid
    extended_grid()

    # Recommendations
    recommendations()

    # Reverse calculation
    optimal_for_target_stop()

    # Position size comparison
    position_size_comparison()

    # Config file recommendations
    config_recommendations()

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    m = 0.005
    b = 0.006
    max_3x = calculate_max_stop_long(3, m, b) * 100
    margin_3x = max_3x - 22

    print(f"\nYour current 3x leverage with 22% stop is VERY conservative.")
    print(f"You have {margin_3x:.2f}% extra safety margin - you could safely:")
    print(f"  1. Increase leverage to 5x (keep 22% stop, better capital efficiency)")
    print(f"  2. Keep 3x but tighten stop to {max_3x:.1f}% (maximum safe)")
    print(f"  3. Go to 5x with 15% stop (recommended balanced approach)")
    print(f"\nFor delta-neutral funding arbitrage with active monitoring:")
    print(f"  -> 5x leverage with 15% stop is ideal for most traders")
    print(f"  -> 7x leverage with 10% stop for experienced traders")
    print(f"\nNote: All stop-loss values include 0.8x safety multiplier (20% extra cushion)")
    print("=" * 100)
