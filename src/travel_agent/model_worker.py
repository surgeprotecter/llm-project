"""Keep one async model event loop so in-flight HTTP requests can be cancelled."""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading

from travel_agent.control import ExecutionCancelled


class ModelWorker:
    def __init__(self):
        self.ready = threading.Event()
        self.loop = None
        self.active = None
        self.cancelled = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name="travel-model")
        self.thread.start()
        self.ready.wait(5)

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()

    def invoke(self, model, messages):
        if self.cancelled.is_set():
            raise ExecutionCancelled()
        future = asyncio.run_coroutine_threadsafe(model.ainvoke(messages), self.loop)
        self.active = future
        try:
            return future.result()
        except concurrent.futures.CancelledError as exc:
            raise ExecutionCancelled() from exc
        finally:
            self.active = None

    def cancel(self):
        self.cancelled.set()
        active, loop = self.active, self.loop
        if active and loop and loop.is_running():
            # Called from the HTTP thread, not from the loop thread.
            loop.call_soon_threadsafe(active.cancel)

    def close(self):
        self.cancel()
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
