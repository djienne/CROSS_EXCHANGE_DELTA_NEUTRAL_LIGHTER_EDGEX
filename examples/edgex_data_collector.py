#!/usr/bin/env python3
"""
EdgeX Market Data Collector - FINAL STABLE VERSION
Focuses on reliably collecting Ticker, Trades, and Order Book (depth) data streams.
Processes order book snapshots and changes to create a full snapshot for each update.
Debug logging is on by default.
"""

import asyncio
import json
import time
import os
import csv
import logging
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import websockets

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- Data Models ---
@dataclass
class PriceData:
    timestamp: float; contract_id: str; price: float
    size: float = 0; side: Optional[str] = None; exchange_timestamp: Optional[int] = None

@dataclass
class TradeData:
    timestamp: float; contract_id: str; price: float; size: float; side: str
    ticketId: Optional[str] = None; exchange_timestamp: Optional[int] = None

# --- Statistics Tracker ---
class DataStats:
    def __init__(self):
        self.start_time = time.time()
        self.counters = defaultdict(int)
    
    def update(self, data_type: str, count: int = 1):
        self.counters[data_type] += count
    
    def get_summary(self) -> Dict[str, Any]:
        runtime = time.time() - self.start_time
        return {
            'runtime_formatted': f"{runtime//3600:.0f}h {(runtime%3600)//60:.0f}m {runtime%60:.0f}s",
            'counters': dict(self.counters),
            'rates_per_minute': {k: v / (runtime / 60) for k, v in self.counters.items() if runtime > 0},
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# --- Main Collector Class ---
class EdgeXDataCollector:
    """Main data collector class for EdgeX - FINAL STABLE VERSION"""
    
    def __init__(self, contract_map: Dict[str, str], output_dir: str, ws_url: str):
        self.contract_map = contract_map
        self.contract_ids = list(contract_map.keys())
        self.output_dir = output_dir
        self.ws_url = ws_url
        self.stats = DataStats()
        self.depth_level = 15 # The depth level to subscribe to
        
        self.running = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self.executor = ThreadPoolExecutor(max_workers=len(self.contract_ids) * 3)
        
        self.data_buffers = {cid: {'tickers': deque(), 'trades': deque(), 'depth': deque()} for cid in self.contract_ids}
        self.contract_files = {}
        self.order_books = {cid: {'bids': {}, 'asks': {}} for cid in self.contract_ids}

        self.last_ticker_state: Dict[str, Dict] = {}
        self.last_data_time = {cid: {'ticker': 0, 'trades': 0, 'depth': 0} for cid in self.contract_ids}
        self.connection_health_threshold = 60  # seconds without data before considering unhealthy
        self._force_reconnect = False

        logger.info("Initializing collector with debug logging enabled.")
        os.makedirs(output_dir, exist_ok=True)
        self._init_csv_files()

    def _init_csv_files(self):
        for contract_id in self.contract_ids:
            ticker_name = self.contract_map.get(contract_id, contract_id)
            self.contract_files[contract_id] = {}
            
            # Create dynamic headers for the depth file
            depth_headers = ['timestamp']
            for i in range(1, self.depth_level + 1):
                depth_headers.append(f'bid_{i}_price')
                depth_headers.append(f'bid_{i}_size')
            for i in range(1, self.depth_level + 1):
                depth_headers.append(f'ask_{i}_price')
                depth_headers.append(f'ask_{i}_size')

            headers = {
                'tickers': ['timestamp', 'contract_id', 'price', 'size', 'side', 'exchange_timestamp'],
                'trades': ['timestamp', 'contract_id', 'price', 'size', 'side', 'ticketId', 'exchange_timestamp'],
                'depth': depth_headers,
            }

            for data_type, header_list in headers.items():
                path = os.path.join(self.output_dir, f"{data_type}_{ticker_name}.csv")
                self.contract_files[contract_id][data_type] = path
                logger.info(f"Initialized CSV file: {path}")
                if not os.path.exists(path):
                    with open(path, 'w', newline='') as f:
                        csv.writer(f).writerow(header_list)
    
    def _write_to_csv_sync(self, filename: str, data: List[Dict]):
        if not data: return
        logger.debug(f"WRITING {len(data)} rows to {os.path.basename(filename)}")
        try:
            headers = list(data[0].keys())
            file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
            with open(filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(data)
        except Exception:
            logger.error(f"Error writing to CSV {filename}", exc_info=True)

    async def _flush_buffers(self):
        loop = asyncio.get_running_loop()
        logger.debug("Starting periodic buffer flush...")
        for contract_id, buffers in self.data_buffers.items():
            for data_type, buffer in buffers.items():
                if buffer:
                    data_to_write = list(buffer)
                    buffer.clear()
                    filename = self.contract_files[contract_id][data_type]
                    await loop.run_in_executor(self.executor, self._write_to_csv_sync, filename, data_to_write)
        logger.debug("Buffer flush complete.")
    
    def _handle_message(self, message: str):
        try:
            data = json.loads(message)
            pretty_payload = json.dumps(data, indent=2)
            logger.debug(f"RAW MESSAGE RECEIVED:\n{pretty_payload}")
            
            if data.get('type') == 'subscribed':
                logger.info(f"SUCCESSFULLY SUBSCRIBED to channel: {data.get('channel')}")
                return
            elif data.get('type') == 'error':
                logger.error(f"SERVER ERROR on subscription: {data.get('content', {}).get('msg')} for request: {data.get('request')}")
                return
            
            channel = data.get('channel', '')
            if 'ticker' in channel: self._process_ticker(data)
            elif 'trades' in channel: self._process_trade(data)
            elif 'depth' in channel: self._process_depth(data)
            else: logger.debug(f"Ignoring message with type '{data.get('type')}' and channel: '{channel}'")
        except json.JSONDecodeError:
            logger.warning(f"RAW MESSAGE (NOT JSON): {message[:300]}")
        except Exception:
            logger.error(f"Error processing message: {message[:300]}", exc_info=True)
            
    def _process_ticker(self, msg: Dict):
        contract_id = msg['channel'].split('.')[1]
        if contract_id not in self.contract_ids: return
        
        update_data = msg.get('content', {}).get('data', [{}])[0]
        
        last_state = self.last_ticker_state.get(contract_id, {})
        last_state.update(update_data)
        self.last_ticker_state[contract_id] = last_state
        current_data = last_state

        now = time.time()
        exchange_ts = current_data.get('timestamp')
        records_to_add = []

        last_price = next((float(current_data[k]) for k in ['lastPrice', 'close'] if current_data.get(k) is not None), None)
        bid_price = next((float(current_data[k]) for k in ['bestBidPrice', 'bid'] if current_data.get(k) is not None), None)
        ask_price = next((float(current_data[k]) for k in ['bestAskPrice', 'ask'] if current_data.get(k) is not None), None)
        
        if last_price:
            records_to_add.append(asdict(PriceData(now, contract_id, last_price, 0, 'last', exchange_ts)))
        if bid_price:
            records_to_add.append(asdict(PriceData(now, contract_id, bid_price, 0, 'bid', exchange_ts)))
        if ask_price:
            records_to_add.append(asdict(PriceData(now, contract_id, ask_price, 0, 'ask', exchange_ts)))
        
        if records_to_add:
            logger.debug(f"PROCESSED TICKER GROUP for {contract_id}: last={last_price}, bid={bid_price}, ask={ask_price}")
            self.data_buffers[contract_id]['tickers'].extend(records_to_add)
            self.stats.update('ticker_datapoints', len(records_to_add))
            self.last_data_time[contract_id]['ticker'] = time.time()

    def _process_trade(self, msg: Dict):
        contract_id = msg['channel'].split('.')[1]
        if contract_id not in self.contract_ids: return
        for data in msg.get('content', {}).get('data', []):
            price_val = data.get('price')
            size_val = data.get('size')
            
            side_val = 'unknown'
            if isinstance(data.get('isBuyerMaker'), bool):
                side_val = 'sell' if data['isBuyerMaker'] else 'buy'

            if price_val is None or size_val is None: continue
            
            ticket_id = str(data.get('ticketId', ''))
            exchange_ts = data.get('time')

            self.data_buffers[contract_id]['trades'].append(asdict(TradeData(
                time.time(),
                contract_id,
                float(price_val),
                float(size_val),
                side_val,
                ticket_id,
                exchange_ts
            )))
            self.stats.update('trade_updates')
            self.last_data_time[contract_id]['trades'] = time.time()

    def _format_book_to_row(self, contract_id: str) -> Dict:
        book = self.order_books[contract_id]
        row = {'timestamp': time.time()}

        try:
            # Filter out invalid prices and sort
            valid_bids = [(p, s) for p, s in book['bids'].items() if p and s]
            valid_asks = [(p, s) for p, s in book['asks'].items() if p and s]

            sorted_bids = sorted(valid_bids, key=lambda item: float(item[0]), reverse=True)
            sorted_asks = sorted(valid_asks, key=lambda item: float(item[0]))

            for i in range(self.depth_level):
                if i < len(sorted_bids):
                    price, size = sorted_bids[i]
                    row[f'bid_{i+1}_price'] = price
                    row[f'bid_{i+1}_size'] = size
                else:
                    row[f'bid_{i+1}_price'] = None
                    row[f'bid_{i+1}_size'] = None

                if i < len(sorted_asks):
                    price, size = sorted_asks[i]
                    row[f'ask_{i+1}_price'] = price
                    row[f'ask_{i+1}_size'] = size
                else:
                    row[f'ask_{i+1}_price'] = None
                    row[f'ask_{i+1}_size'] = None

        except Exception as e:
            logger.error(f"Error formatting order book for {contract_id}: {e}", exc_info=True)
            # Return empty row with None values if formatting fails
            for i in range(self.depth_level):
                row[f'bid_{i+1}_price'] = None
                row[f'bid_{i+1}_size'] = None
                row[f'ask_{i+1}_price'] = None
                row[f'ask_{i+1}_size'] = None

        return row

    def _process_depth(self, msg: Dict):
        channel_parts = msg['channel'].split('.')
        contract_id = channel_parts[1]
        if contract_id not in self.contract_ids: return

        logger.debug(f"Processing depth for {contract_id} (PAXG={contract_id=='10000227'})")
        book = self.order_books[contract_id]

        try:
            for data_item in msg.get('content', {}).get('data', []):
                depth_type = data_item.get('depthType')
                logger.debug(f"Depth type: {depth_type} for {contract_id}")

                if depth_type == 'Snapshot':
                    book['bids'].clear()
                    book['asks'].clear()
                    logger.debug(f"Cleared order book for {contract_id}")

                for level in data_item.get('bids', []):
                    try:
                        price, size = level.get('price'), level.get('size')
                        if price is None or size is None:
                            logger.debug(f"Skipping bid level with None values: price={price}, size={size}")
                            continue

                        size_float = float(size)
                        if size_float == 0:
                            book['bids'].pop(str(price), None)
                        else:
                            book['bids'][str(price)] = str(size)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing bid level for {contract_id}: {e}, level: {level}")
                        continue

                for level in data_item.get('asks', []):
                    try:
                        price, size = level.get('price'), level.get('size')
                        if price is None or size is None:
                            logger.debug(f"Skipping ask level with None values: price={price}, size={size}")
                            continue

                        size_float = float(size)
                        if size_float == 0:
                            book['asks'].pop(str(price), None)
                        else:
                            book['asks'][str(price)] = str(size)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing ask level for {contract_id}: {e}, level: {level}")
                        continue

                snapshot_row = self._format_book_to_row(contract_id)
                self.data_buffers[contract_id]['depth'].append(snapshot_row)
                self.stats.update('depth_snapshots')
                self.last_data_time[contract_id]['depth'] = time.time()
                logger.debug(f"Added depth snapshot for {contract_id}")

        except Exception as e:
            logger.error(f"Error processing depth message for {contract_id}: {e}", exc_info=True)

    def _check_connection_health(self) -> bool:
        """Check if all contracts are receiving data within the health threshold"""
        current_time = time.time()
        unhealthy_contracts = []

        for contract_id in self.contract_ids:
            ticker_name = self.contract_map.get(contract_id, contract_id)
            last_times = self.last_data_time[contract_id]

            for data_type, last_time in last_times.items():
                if last_time > 0:  # Only check if we've received data before
                    time_since_last = current_time - last_time
                    if time_since_last > self.connection_health_threshold:
                        unhealthy_contracts.append(f"{ticker_name}.{data_type}")
                        logger.warning(f"No {data_type} data for {ticker_name} in {time_since_last:.1f}s")

        if unhealthy_contracts:
            logger.error(f"Connection health check failed. Unhealthy: {', '.join(unhealthy_contracts)}")
            return False

        return True

    async def _periodic_flush_task(self):
        while self.running.is_set(): await asyncio.sleep(10); await self._flush_buffers()

    async def _periodic_summary_task(self):
        while self.running.is_set(): await asyncio.sleep(30); self._print_summary()

    async def _periodic_health_check_task(self):
        """Periodically check connection health and trigger reconnection if needed"""
        while self.running.is_set():
            await asyncio.sleep(30)  # Check every 30 seconds
            if not self._check_connection_health():
                logger.error("Health check failed - forcing reconnection")
                # Set a flag to trigger reconnection
                self._force_reconnect = True

    async def _run_collection_task(self):
        data_types = ["ticker", "trades"]
        channels = [f"{dtype}.{cid}" for cid in self.contract_ids for dtype in data_types]

        depth_channels = [f"depth.{cid}.{self.depth_level}" for cid in self.contract_ids]
        channels.extend(depth_channels)

        while self.running.is_set():
            try:
                connect_url = f"{self.ws_url}?timestamp={int(time.time() * 1000)}"
                logger.info(f"Connecting to {connect_url}...")

                # Reset force reconnect flag
                self._force_reconnect = False

                async with websockets.connect(connect_url, ping_interval=15, ping_timeout=10) as ws:
                    logger.info("Connection successful. Subscribing to channels individually...")
                    for channel in channels:
                        sub_msg_str = json.dumps({"type": "subscribe", "channel": channel})
                        logger.debug(f"SENDING SUBSCRIBE MESSAGE: {sub_msg_str}")
                        await ws.send(sub_msg_str)
                        await asyncio.sleep(0.1)
                    logger.info(f"All {len(channels)} subscriptions sent. Awaiting data...")

                    # Listen for messages and check for forced reconnection
                    async for message in ws:
                        if self._force_reconnect:
                            logger.info("Forced reconnection requested - closing connection")
                            break
                        self._handle_message(message)

            except (websockets.exceptions.ConnectionClosed, websockets.exceptions.InvalidStatusCode) as e:
                logger.warning(f"WebSocket issue: {e}. Reconnecting in 5s...")
            except asyncio.TimeoutError:
                logger.warning("Connection timed out. Reconnecting in 5s...")
            except Exception as e:
                logger.error(f"Unexpected error in collection loop: {e}", exc_info=True)

            if self.running.is_set():
                if self._force_reconnect:
                    logger.info("Reconnecting immediately due to health check failure")
                    await asyncio.sleep(1)  # Brief pause before reconnect
                else:
                    await asyncio.sleep(5)

    def _print_summary(self):
        summary = self.stats.get_summary()
        current_time = time.time()

        print("\n" + "="*60); print(f"EDGEX DATA COLLECTION SUMMARY - {summary['last_update']}")
        print("="*60); print(f"Runtime: {summary['runtime_formatted']}")
        for dtype, count in summary['counters'].items():
            rate = summary['rates_per_minute'].get(dtype, 0)
            print(f"  {dtype:<20}: {count:>10,} ({rate:>6.1f}/min)")

        print("\nPER-CONTRACT DATA ACTIVITY:")
        for contract_id in self.contract_ids:
            ticker_name = self.contract_map.get(contract_id, contract_id)
            last_times = self.last_data_time[contract_id]
            status_line = f"  {ticker_name:<6}:"

            for data_type in ['ticker', 'trades', 'depth']:
                last_time = last_times[data_type]
                if last_time == 0:
                    status = "NEVER"
                else:
                    ago = current_time - last_time
                    if ago < 30:
                        status = "LIVE"
                    elif ago < 60:
                        status = f"{ago:.0f}s"
                    else:
                        status = f"{ago/60:.1f}m"
                status_line += f" {data_type[:3].upper()}:{status:<6}"

            print(status_line)

        print("="*60)

    async def _run_main_loop(self):
        self.running.set()
        self._tasks = [
            asyncio.create_task(self._periodic_flush_task()),
            asyncio.create_task(self._periodic_summary_task()),
            asyncio.create_task(self._periodic_health_check_task()),
            asyncio.create_task(self._run_collection_task())
        ]
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def start(self):
        tickers = ", ".join(self.contract_map.values())
        print(f"Starting EdgeX data collection for: {tickers}")
        try: asyncio.run(self._run_main_loop())
        except KeyboardInterrupt: print("\nShutdown signal received.")
        finally: self.stop()
            
    def stop(self):
        if not self.running.is_set(): return
        print("\nInitiating graceful shutdown...")
        self.running.clear()
        
        for task in self._tasks: task.cancel() 
        
        logger.info("Flushing final data buffers...")
        for cid, buffers in self.data_buffers.items():
            for dtype, buffer in buffers.items():
                if buffer: self._write_to_csv_sync(self.contract_files[cid][dtype], list(buffer))
        self.executor.shutdown(wait=True)
        
        self._print_summary()
        print(f"Data collection stopped. Files saved in: {self.output_dir}")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # CONTRACT_MAP = {
    #     "10000001": "BTC",
    #     "10000002": "ETH",
    #     "10000003": "SOL",
    #     "10000227": "PAXG"
    # }
    CONTRACT_MAP = {
        "10000227": "PAXG"
    }
    
    OUTPUT_DIR = "edgex_data"
    WS_URL = "wss://quote.edgex.exchange/api/v1/public/ws"

    collector = EdgeXDataCollector(
        contract_map=CONTRACT_MAP,
        output_dir=OUTPUT_DIR,
        ws_url=WS_URL,
    )
    collector.start()

if __name__ == "__main__":
    main()

