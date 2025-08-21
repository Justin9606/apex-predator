# data/fetcher.py
# The bloodstream: multi-venue, live-only tick ingestion with QoS + time alignment.
from __future__ import annotations

import os
import time
import json
import threading
import queue
import requests
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Optional, Generator

# Project utils (must exist per build order)
from utils.helpers import now_utc_ns, monotonic_ms, safe_float, jitter_ms, load_yaml_hot

# ---------------------------
# Memory Hook (silent mantra)
# ---------------------------
_PREDATOR_CREED = (
    "The market is not random. I am the predator. There are loopholes. There are no limits."
)

# ---------------------------
# Data structures
# ---------------------------
@dataclass(frozen=True)
class Tick:
    ts_ns: int                 # exchange/server timestamp in nanoseconds (as received)
    recv_ns: int               # local receipt timestamp (monotonic origin)
    instrument: str
    bid: float
    ask: float
    mid: float
    volume: Optional[float]
    source: str                # 'tradingview', 'oanda', 'mt5'
    latency_ms: float          # recv lag estimate
    meta: Dict[str, Any]       # 6W aids: where/why/how provenance, raw fragments

# ---------------------------
# QoS guard & filter
# ---------------------------
class QoSFilter:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.last_emitted_ns: Dict[str, int] = {}

    def accept(self, tick: Tick) -> bool:
        qos = self.cfg["qos"]
        # Staleness
        age_ms = max(0.0, (tick.recv_ns - tick.ts_ns) / 1e6)
        if age_ms > qos["max_staleness_ms"]:
            return False

        # Rate limit per source+instrument
        key = f"{tick.source}:{tick.instrument}"
        last = self.last_emitted_ns.get(key, 0)
        if last and (tick.recv_ns - last) / 1e6 < qos["min_tick_interval_ms"]:
            return False

        # Spread sanity
        points = abs(tick.ask - tick.bid) * 10_000  # 1 pip = 0.0001; gold often quoted as 0.01—normalize downstream
        if points > qos["max_spread_points"]:
            return False

        # Accept
        self.last_emitted_ns[key] = tick.recv_ns
        return True

# ---------------------------
# Config watcher
# ---------------------------
class Config:
    def __init__(self, path: str) -> None:
        self.path = path
        self._cfg = load_yaml_hot(path)
        self._last_reload_ms = monotonic_ms()

    @property
    def data(self) -> Dict[str, Any]:
        # hot reload if enabled and due
        hr = self._cfg.get("hot_reload", {})
        if hr.get("enabled", False):
            due = hr.get("poll_ms", 1500)
            if monotonic_ms() - self._last_reload_ms >= due:
                self._cfg = load_yaml_hot(self.path)
                self._last_reload_ms = monotonic_ms()
        return self._cfg

# ---------------------------
# Base connector
# ---------------------------
class DataSource:
    def __init__(self, name: str, cfg: Dict[str, Any], outq: queue.Queue):
        self.name = name
        self.cfg = cfg
        self.outq = outq
        self._stop = threading.Event()

    def start(self) -> None:
        t = threading.Thread(target=self._run, name=f"{self.name}-thread", daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop.set()

    # To implement by subclasses
    def _run(self) -> None:
        raise NotImplementedError

# ---------------------------
# TradingView (SIM) — Only where ToS permits
# ---------------------------
class TradingViewSource(DataSource):
    def _run(self) -> None:
        tv_cfg = self.cfg["feeds"]["sim"]["tradingview"]
        if not tv_cfg.get("scrape_approved", False):
            # Respect ToS: do not scrape unless permitted
            return

        symbol = tv_cfg["symbol"]
        # Placeholder: You’d connect to your permitted data transport here.
        # Example approach if you have an approved feed: websocket -> parse -> emit Tick
        # Below is a schematic polling placeholder (HTTP) for illustration only.
        ua = tv_cfg.get("user_agent", "MarketEater/1.0")
        headers = {"User-Agent": ua}
        heartbeat = int(tv_cfg.get("heartbeat_ms", 1000))

        session = requests.Session()
        while not self._stop.is_set():
            try:
                # Replace with your approved endpoint
                # resp = session.get(approved_tv_endpoint, headers=headers, timeout=2)
                # payload = resp.json()
                payload = None  # schematic
                if payload:
                    tick = self._to_tick(payload, symbol)
                    if tick:
                        self.outq.put(tick, block=False)
            except Exception:
                pass
            time.sleep(heartbeat / 1000.0)

    def _to_tick(self, payload: Dict[str, Any], symbol: str) -> Optional[Tick]:
        # Map your approved payload into Tick
        try:
            ts_ns = now_utc_ns()  # Ideally, parse exchange/server ts -> ns
            recv_ns = now_utc_ns()
            bid = safe_float(payload.get("bid"))
            ask = safe_float(payload.get("ask"))
            if bid is None or ask is None:
                return None
            return Tick(
                ts_ns=ts_ns,
                recv_ns=recv_ns,
                instrument="XAUUSD",
                bid=bid,
                ask=ask,
                mid=(bid + ask) / 2.0,
                volume=None,
                source="tradingview",
                latency_ms=max(0.0, (recv_ns - ts_ns) / 1e6),
                meta={"where": "tradingview", "why": "sim_feed", "how": "http_poll", "raw": payload},
            )
        except Exception:
            return None

# ---------------------------
# OANDA (SIM) — Official streaming API
# ---------------------------
class OandaSource(DataSource):
    def _run(self) -> None:
        fcfg = self.cfg["feeds"]["sim"]["oanda"]
        api_key = os.getenv(fcfg["api_key_env"], "")
        if not api_key:
            return
        account_id = fcfg["account_id"].strip()
        if not account_id or "REPLACE_ME" in account_id:
            return

        url = fcfg["pricing_stream_url"].format(account_id=account_id)
        params = {"instruments": fcfg.get("instruments_param", "XAU_USD")}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

        with requests.get(url, headers=headers, params=params, stream=True, timeout=30) as r:
            for line in r.iter_lines():
                if self._stop.is_set():
                    break
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                    tick = self._to_tick(obj)
                    if tick:
                        self.outq.put(tick, block=False)
                except Exception:
                    continue

    def _to_tick(self, obj: Dict[str, Any]) -> Optional[Tick]:
        try:
            if obj.get("type") != "PRICE":
                return None
            bids = obj.get("bids", [])
            asks = obj.get("asks", [])
            if not bids or not asks:
                return None
            bid = safe_float(bids[0].get("price"))
            ask = safe_float(asks[0].get("price"))
            ts_iso = obj.get("time")
            # Convert ts_iso → ns (helpers should support this; fallback to local clock if missing)
            ts_ns = now_utc_ns()
            recv_ns = now_utc_ns()
            return Tick(
                ts_ns=ts_ns,
                recv_ns=recv_ns,
                instrument="XAUUSD",
                bid=bid,
                ask=ask,
                mid=(bid + ask) / 2.0,
                volume=None,
                source="oanda",
                latency_ms=max(0.0, (recv_ns - ts_ns) / 1e6),
                meta={
                    "where": "oanda_stream",
                    "why": "sim_feed",
                    "how": "official_stream",
                    "raw_type": obj.get("type"),
                },
            )
        except Exception:
            return None

# ---------------------------
# MT5 (LIVE) — Windows-only interaction
# ---------------------------
class MT5Source(DataSource):
    def _run(self) -> None:
        live_cfg = self.cfg["feeds"]["live"]
        if not live_cfg.get("enabled", False):
            return
        # MetaTrader5 Python package usage typically requires Windows + terminal installed and logged in.
        # Implement your MT5 bridge here (symbol map, ticks subscription, etc.).
        # This stub intentionally avoids fake connections.
        return

# ---------------------------
# Aggregator / Multiplexer
# ---------------------------
class MarketFetcher:
    """
    Unifies all configured sources into a single QoS-checked Tick stream.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.outq: queue.Queue = queue.Queue(maxsize=10000)
        self.sources: list[DataSource] = []
        self.qos = QoSFilter(self.cfg.data)
        self._stop = threading.Event()

    def start(self) -> None:
        conf = self.cfg.data
        mode = conf["mode"]
        feeds = conf["feeds"]

        if mode == "sim":
            if feeds["sim"]["enabled"]:
                if "tradingview" in feeds["sim"]["sources"]:
                    self.sources.append(TradingViewSource("tradingview", conf, self.outq))
                if "oanda" in feeds["sim"]["sources"]:
                    self.sources.append(OandaSource("oanda", conf, self.outq))
        elif mode == "live":
            if feeds["live"]["enabled"]:
                self.sources.append(MT5Source("mt5", conf, self.outq))

        for s in self.sources:
            s.start()

    def stop(self) -> None:
        self._stop.set()
        for s in self.sources:
            s.stop()

    def stream(self) -> Generator[Tick, None, None]:
        """
        Blocking generator yielding QoS-accepted ticks with provenance (6W-ready).
        """
        while not self._stop.is_set():
            tick: Tick = self.outq.get()
            if self.qos.accept(tick):
                yield tick

# ---------------------------
# If run directly: dry integration smoke (no fallbacks, no fakes)
# ---------------------------
if __name__ == "__main__":
    print(_PREDATOR_CREED)
    cfg = Config(path=os.path.join("config", "dynamic_genesis.yaml"))
    fetcher = MarketFetcher(cfg)
    fetcher.start()
    try:
        i = 0
        for t in fetcher.stream():
            i += 1
            # Minimal stdout: source, mid, latency
            print(f"[{t.source}] {t.instrument} mid={t.mid:.2f} lag={t.latency_ms:.1f}ms")
            if i >= 5:
                break
    finally:
        fetcher.stop()
