"""Subscribe to the relay's ``xr_frame`` stream — shared across every robot.

Connects over WebSocket, keeps the newest controller/headset pose, and reports
how stale it is. What to do with those poses belongs to a robot's teleoperator.

Reconnects on its own, and accepts self-signed TLS for ``wss://`` since the
relay serves a LAN cert. Staleness is reported rather than gated on — see
``latest()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable

import websockets

logger = logging.getLogger(__name__)


def _ssl_context_for(url: str):
    """Permissive TLS for ``wss://``. See the module docstring — the relay's
    cert is self-signed by design, so validation would only refuse the
    handshake without adding security."""
    if not url.startswith("wss://"):
        return None
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class XRFrameClient:
    """Background relay client holding the newest ``xr_frame``::

        client = XRFrameClient("wss://127.0.0.1:8443/ws")
        client.connect(timeout_s=5.0)
        frame, age_s = client.latest()
        client.disconnect()
    """

    def __init__(
        self,
        ws_url: str,
        name: str = "xr-client",
        on_message: Callable[[dict], None] | None = None,
    ) -> None:
        self.ws_url = ws_url
        self._name = name
        self._lock = threading.Lock()
        self._frame: dict | None = None
        self._frame_time: float = 0.0
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self._connected = threading.Event()
        self._on_message = on_message

    # ---- lifecycle --------------------------------------------------------
    def connect(self, timeout_s: float = 5.0) -> None:
        if self.is_connected:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"{self._name}: connection thread is already running")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._main, name=self._name, daemon=True)
        self._thread.start()
        if not self._connected.wait(timeout=timeout_s):
            self.disconnect()
            raise RuntimeError(f"{self._name}: timed out connecting to {self.ws_url}")
        logger.info("%s connected to %s", self._name, self.ws_url)

    def disconnect(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._loop is not None and self._ws is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
                fut.result(timeout=1.0)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._connected.clear()
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None
            self._stop = None

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ---- data -------------------------------------------------------------
    def latest(self) -> tuple[dict | None, float]:
        """Newest frame and its age in seconds.

        Age rather than a freshness verdict: stow and disarm are safe on a stale
        pose and are what an operator reaches for when the link degrades.
        """
        with self._lock:
            if self._frame is None:
                return None, float("inf")
            return self._frame, time.time() - self._frame_time

    def send(self, payload: dict) -> None:
        """Fire-and-forget a message back to the relay (e.g. ``ik_state``)."""
        if self._loop is None or self._ws is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(payload)), self._loop)
        except Exception:
            logger.debug("%s: send failed", self._name, exc_info=True)

    # ---- internals --------------------------------------------------------
    def _main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        finally:
            self._loop.close()
            self._loop = None

    async def _run(self) -> None:
        assert self._stop is not None
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.ws_url, ssl=_ssl_context_for(self.ws_url)
                ) as ws:
                    self._ws = ws
                    self._connected.set()
                    while not self._stop.is_set():
                        raw = await ws.recv()
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        if msg.get("type") == "xr_frame":
                            with self._lock:
                                self._frame = msg
                                self._frame_time = time.time()
                        if self._on_message is not None:
                            try:
                                self._on_message(msg)
                            except Exception:
                                logger.exception("%s: message callback failed", self._name)
            except Exception as e:
                # Reconnect rather than ending the session: WiFi/tunnel jitter
                # drops this link routinely mid-run.
                self._connected.clear()
                self._ws = None
                if self._stop.is_set():
                    break
                logger.warning("%s: connection lost (%s) — retrying", self._name, e)
                await asyncio.sleep(0.5)
