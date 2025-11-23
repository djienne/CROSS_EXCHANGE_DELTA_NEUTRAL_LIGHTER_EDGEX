# Cross-Exchange Delta-Neutral Arbitrage Bot

Automated 24/7 funding rate arbitrage bot for cryptocurrency perpetual futures. Opens simultaneous long and short positions across EdgeX and Lighter exchanges to capture funding rate differentials while maintaining market-neutral exposure.

## 🎯 Features

- **Fully Automated**: 24/7 operation for scanning, opening, monitoring, and closing delta-neutral positions
- **Cross-Exchange Arbitrage**: Simultaneously trades on EdgeX and Lighter to capture funding rate spreads
- **Delta-Neutral**: Minimizes directional risk with balanced long/short positions on the same asset
- **Smart Position Selection**: Automatically selects the best opportunity based on:
  - Net APR (funding rate spread between exchanges)
  - 24h trading volume (minimum $250M combined)
  - Price spread between exchanges (maximum 0.15%)
- **Automatic Rollback**: If one leg fails during position opening, automatically closes the successful leg to prevent orphaned exposure
- **Percentage-Based Execution**: Uses 3% from mid price for aggressive fills (well under 5% exchange limit)
- **State Persistence**: Seamlessly resumes from `logs/bot_state.json` after restarts
- **Position Recovery**: Automatically verifies and recovers positions on restart
- **Configurable Leverage**: Supports 1x-5x leverage with automatic stop-loss calculation
- **Risk Management**:
  - Automatic stop-loss: `(100/leverage) × 0.7`
  - Capital safety margin: 5% buffer on position sizing
  - Real-time PnL tracking for both exchanges

## 🏗️ Architecture

### Core Files

- **`lighter_edgex_hedge.py`**: Main production bot with automated 24/7 rotation
- **`examples/hedge_cli.py`**: Manual trading CLI for testing and analysis
- **`lighter_client.py`**: Lighter exchange helper functions
- **`edgex_client.py`**: EdgeX exchange helper functions
- **`emergency_close.py`**: Emergency position closer (Linux/macOS only)

### Configuration

- **`bot_config.json`**: Automated bot configuration
- **`hedge_config.json`**: Manual CLI configuration (examples/)
- **`.env`**: Exchange API credentials (see `.env.example`)

## 📋 Prerequisites

- Docker & Docker Compose (recommended for 24/7 operation)
- Python 3.8+ (for local development)
- EdgeX API credentials
- Lighter API credentials

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository_url>
cd CROSS_EXCHANGE_DELTA_NEUTRAL_LIGHTER_EDGEX
```

### 2. Configure API Credentials

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# EdgeX Credentials
EDGEX_BASE_URL=https://pro.edgex.exchange
EDGEX_WS_URL=wss://quote.edgex.exchange
EDGEX_ACCOUNT_ID=123456  # Must be integer
EDGEX_STARK_PRIVATE_KEY=0x...

# Lighter Credentials
LIGHTER_BASE_URL=https://mainnet.zklighter.elliot.ai
LIGHTER_WS_URL=wss://mainnet.zklighter.elliot.ai/stream
LIGHTER_PRIVATE_KEY=0x...
LIGHTER_ACCOUNT_INDEX=0
LIGHTER_API_KEY_INDEX=0
```

⚠️ **CRITICAL:** `EDGEX_ACCOUNT_ID` must be an integer, not a string (SDK uses bitwise operations).

### 3. Configure Bot Strategy

Edit `bot_config.json`:

```json
{
  "symbols_to_monitor": ["BTC", "ETH", "SOL", "BNB", "LINK"],
  "leverage": 3,
  "notional_per_position": 1000,
  "hold_duration_hours": 12,
  "min_net_apr_threshold": 5.0,
  "min_volume_usd": 250000000,
  "max_spread_pct": 0.15,
  "enable_stop_loss": true
}
```

## 🚀 Usage

### Production Bot (24/7 Automated)

**With Docker (Recommended):**

```bash
# Start the bot
docker-compose up -d lighter_edgex_hedge

# View logs
docker-compose logs -f lighter_edgex_hedge

# Stop the bot
docker-compose down
```

**Without Docker:**

```bash
python lighter_edgex_hedge.py
```

### Manual CLI Tool

```bash
# Check funding rates (auto-updates config)
python examples/hedge_cli.py funding

# Compare multiple markets
python examples/hedge_cli.py funding_all

# Check available capital
python examples/hedge_cli.py capacity

# Check current positions
python examples/hedge_cli.py status

# Open position manually
python examples/hedge_cli.py open --size-quote 100

# Close position manually
python examples/hedge_cli.py close

# Test full cycle
python examples/hedge_cli.py test --notional 20
```

### Utility Scripts

**Check all spreads:**
```bash
python check_all_spreads.py
```

**Check trading volume:**
```bash
python check_volume.py
```

**Compare funding rates:**
```bash
python test_funding_comparison.py --symbols BTC ETH SOL
```

**Emergency close (Linux/macOS only):**
```bash
# On Linux/macOS
python emergency_close.py --dry-run  # Check positions
python emergency_close.py             # Close all

# On Windows (MUST use Docker)
docker-compose run emergency_close --dry-run
docker-compose run emergency_close
```

## ⚙️ How It Works

### State Machine (Production Bot)

1. **IDLE** → Waiting to start analysis
2. **ANALYZING** → Fetching funding rates, volumes, spreads
3. **OPENING** → Executing delta-neutral position entry
4. **HOLDING** → Monitoring position health, collecting funding
5. **CLOSING** → Exiting both positions
6. **WAITING** → Cooldown before next cycle
7. **ERROR** → Manual intervention required

### Position Selection

The bot selects opportunities based on:

1. ✅ **Volume Filter**: Combined 24h volume ≥ $250M
2. ✅ **Spread Filter**: Cross-exchange mid price spread ≤ 0.15%
3. ✅ **APR Filter**: Net APR ≥ configured threshold (default 5%)
4. ✅ **Best APR**: Highest net APR from remaining candidates

### Order Execution Strategy

All orders use **percentage-based aggressive limit orders**:

- **BUY orders**: `price = mid × 1.03` (3% above mid)
- **SELL orders**: `price = mid × 0.97` (3% below mid)

This ensures:
- ✅ Near-instant fills across all assets
- ✅ No price bound violations (well under 5% exchange limit)
- ✅ Predictable slippage (~3%)

### Automatic Rollback (New!)

If one leg fails during position opening:

1. ✅ Detects which leg succeeded
2. ✅ Immediately closes the successful position
3. ✅ Prevents orphaned unhedged exposure
4. ✅ Tries next candidate symbol

**Example:**
```
SOL attempt: Lighter LONG ✓ | EdgeX SHORT ✗
  → Rollback: Close Lighter LONG immediately
  → Try next candidate (BNB)
```

### Capital Management

Position sizing algorithm:

1. Fetch available USD on both exchanges
2. Apply safety margin (5%) and fee buffer
3. Calculate per-exchange capacity: `available × leverage / mid_price`
4. Max size = **minimum** of both exchanges (delta-neutral requirement)
5. Round conservatively using coarser tick size

## 📊 Monitoring

### State File (`logs/bot_state.json`)

Tracks:
- Current state and cycle count
- Active position details (symbol, sizing, entry prices)
- Capital status on both exchanges
- Completed cycle history
- Cumulative performance stats

### Logs

- `logs/lighter_edgex_hedge.log`: Full bot debug output
- `hedge_cli.log`: Manual CLI operations

### Performance Tracking

The bot displays:
- **EdgeX PnL**: Unrealized profit/loss from EdgeX position
- **Lighter PnL**: Unrealized profit/loss from Lighter position
- **Total PnL**: Combined unrealized PnL
- **Long-term PnL**: Total profit since bot started
- **Cycle Stats**: Success rate, average PnL per cycle

## 🔧 Configuration Reference

### Bot Config (`bot_config.json`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `symbols_to_monitor` | List of symbols to analyze | `["BTC", "ETH", ...]` |
| `leverage` | Leverage for both exchanges | `3` |
| `notional_per_position` | Max position size in USD | `1000` |
| `hold_duration_hours` | How long to hold each position | `12` |
| `min_net_apr_threshold` | Minimum net APR to open (%) | `5.0` |
| `min_volume_usd` | Minimum 24h volume filter | `250000000` |
| `max_spread_pct` | Max price spread filter (%) | `0.15` |
| `enable_stop_loss` | Auto stop-loss enabled | `true` |

**Note:** Stop-loss is auto-calculated as `(100/leverage) × 0.7`.

## ⚠️ Known Limitations

### Windows Compatibility

The Lighter SDK only supports Linux/macOS. On Windows:
- ✅ **Bot works**: Via Docker
- ❌ **emergency_close.py**: Does NOT work directly
- ✅ **Emergency close**: Use `docker-compose run emergency_close`

### Exchange-Specific Notes

**EdgeX:**
- Contract name format: `{SYMBOL}{QUOTE}` (e.g., "BTCUSD")
- `account_id` MUST be integer (SDK uses bitwise operations)
- Capital from `totalEquity` field

**Lighter:**
- Market identification by symbol only
- Capital via WebSocket `user_stats` channel
- Position close uses dual reduce-only orders

## 🐛 Troubleshooting

### "Position size mismatch" error
- Uses coarser tick size and floors for consistency
- Check logs for detailed size calculations

### "Leverage setup failed" warning
- EdgeX can't verify leverage until position exists
- Warning is informational, bot proceeds safely

### Unhedged position detected
- One leg failed/closed manually
- Bot enters ERROR state
- **Fix:** Run emergency close, then restart

### API rate limits (Lighter)
- Bot uses global semaphore (max 2 concurrent calls)
- Automatic retry with exponential backoff
- Staggered delays (1s between symbols)

## 📈 Recent Improvements (January 2025)

### 1. Funding Rate Cache (January 2025)
- **Smart caching system**: Funding rates are cached for 5 minutes by default to avoid redundant API calls
- **Automatic cache management**: Cache entries expire automatically after TTL (300 seconds)
- **Per-exchange caching**: Separate cache entries for EdgeX and Lighter funding rates
- **Transparent operation**: Cache hits logged at DEBUG level for visibility
- **Why it matters**: Prevents duplicate API calls when bot restarts or analyzes multiple times within short period
- **Configuration**: Adjust `FUNDING_CACHE_TTL_SECONDS` in code if needed (default: 300s)
- **Cache key format**: `(symbol, quote, exchange)` - e.g., `("BTC", "USD", "edgex")`

### 2. Automatic Rollback on Partial Fills
- Detects partial fills and closes orphaned positions
- Prevents unhedged directional exposure
- Tries next best opportunity automatically

### 3. Percentage-Based Pricing
- Changed from tick-based (500 ticks) to percentage-based (3%)
- Works consistently across all assets
- Well under 5% exchange price limits (Lighter rejects orders at exactly 5%)

### 4. Volume & Spread Filtering
- $250M minimum combined volume
- 0.15% maximum price spread
- Ensures high-quality pairs only

## 📝 License

[Your License Here]

## ⚠️ Disclaimer

**Trading cryptocurrencies involves significant risk.** This bot is provided as-is, without warranty or guarantee of profitability. The authors are not responsible for financial losses. Use at your own risk and only with capital you can afford to lose.

Funding rate arbitrage carries risks including:
- Exchange downtime/maintenance
- Extreme market volatility
- Liquidation risk (especially with leverage)
- Smart contract risks (especially on DEXs)
- API rate limiting or restrictions

Always test with small positions first and monitor performance closely.
