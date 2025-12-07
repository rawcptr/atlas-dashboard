import json
import logging
import threading
from queue import Queue, Empty
from typing import Any, Optional

from websockets.sync.client import connect, ClientConnection

logger = logging.getLogger(__name__)


class AtlasTransport:
    def __init__(self, uri: str, stop_event: threading.Event) -> None:
        self.uri = uri
        self.ws: Optional[ClientConnection] = None
        self.message_queue: Queue[dict[str, Any]] = Queue()
        self.sender_thread: Optional[threading.Thread] = None
        self._stop_event = stop_event

    def connect(self) -> None:
        try:
            self.ws = connect(self.uri)
            logger.info("Atlas connected: %s", self.uri)
        except Exception as e:
            logger.error("Atlas connect failed: %s", e)
            raise

    def disconnect(self) -> None:
        try:
            if self.ws:
                self.ws.close()
        finally:
            self.ws = None
            logger.info("Atlas disconnected")

    def start_sender_thread(self) -> None:
        self._stop_event.clear()
        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=False)
        self.sender_thread.start()

    def stop_sender_thread(self) -> None:
        self._stop_event.set()
        if self.sender_thread:
            self.sender_thread.join(timeout=5)
        self.sender_thread = None

    def _sender_loop(self) -> None:
        while not self._stop_event.is_set() or not self.message_queue.empty():
            try:
                msg = self.message_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if self.ws:
                    self.ws.send(json.dumps(msg))
            except Exception as e:
                logger.error("Send error: %s", e)
        logger.debug("Sender thread stopped.")

    def queue_message(self, msg: dict[str, Any]) -> None:
        self.message_queue.put(msg)
