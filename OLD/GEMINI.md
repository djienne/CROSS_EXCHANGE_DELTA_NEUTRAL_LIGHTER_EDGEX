# Gemini Context: Cross-Exchange Delta-Neutral Hedging Bot

## Project Overview

This project is a Python-based command-line application (`hedge_cli.py`) designed to execute delta-neutral trading strategies across the Lighter and EdgeX perpetual futures exchanges. Its primary goal is to capture funding rate arbitrage opportunities by simultaneously opening a LONG position on one exchange and a SHORT position on the other for the same market, thus maintaining a market-neutral stance.

**Key Technologies:**
- **Backend:** Python 3.10
- **Core Logic:** Asynchronous operations using `asyncio`.
- **Dependencies:** `edgex-python-sdk`, `lighter-python`, `python-dotenv`, `websockets`, `aiohttp`.
- **Containerization:** Docker and Docker Compose.

**Architecture:**
- **CLI Application:** A single entrypoint `hedge_cli.py` provides several commands for analysis, trading, and testing.
- **Configuration:**
    - `hedge_config.json`: Defines the core trading strategy parameters (symbol, leverage, exchanges, default notional size).
    - `.env`: Stores sensitive API credentials and exchange URLs, loaded via `python-dotenv`.
- **Logging:** All operations are logged to `hedge_cli.log` at the `DEBUG` level, while the console only shows `WARNING` and above to maintain clarity.

## Building and Running

The application can be run directly with Python or via the provided Docker Compose services.

### Local Python Execution

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Commands:**
    ```bash
    # Example: Check trading capacity
    python hedge_cli.py capacity --config hedge_config.json

    # Example: Open a position with a $100 notional size
    python hedge_cli.py open --size-quote 100 --config hedge_config.json
    ```

### Docker Execution

The `docker-compose.yml` file defines services for all major commands, making them easy to execute.

1.  **Build the Image:**
    ```bash
    docker-compose build
    ```

2.  **Run Services:**
    ```bash
    # Example: Check trading capacity
    docker-compose run capacity

    # Example: Open a position with a $100 notional size
    docker-compose run open --size-quote 100

    # Example: Close positions
    docker-compose run close

    # Example: Run an automated test trade
    docker-compose run test_auto --notional 25
    ```

## Development Conventions

- **Configuration:** All strategy parameters are centralized in `hedge_config.json`. Sensitive data is kept separate in a `.env` file.
- **Asynchronous Code:** The entire application is built on `asyncio` to handle concurrent API calls to both exchanges efficiently.
- **Error Handling:** The application includes checks for partial fills and provides clear warnings if one leg of the hedge fails, instructing the user to take manual action.
- **Order Types:** The bot uses aggressive limit orders that cross the spread by a configurable number of ticks (`--cross-ticks`) to ensure immediate execution, simulating market orders while retaining price control.
- **Sizing Logic:** The system intelligently handles different tick and step sizes between exchanges by rounding down to the coarser of the two, ensuring identical position sizes are placed on both venues.
- **Testing:** The CLI includes built-in `test` and `test_auto` commands to perform end-to-end checks with small amounts before committing significant capital. The `test_leverage` command allows for pre-flight checks of exchange settings.
