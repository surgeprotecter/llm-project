"""Drive checkpointed graph slices without imposing a task-length limit."""
from __future__ import annotations

import signal
import threading

from travel_agent.control import ExecutionCancelled


class DurableAgentGraph:
    def __init__(self, graph, database, gateway, registry):
        self.graph, self.database = graph, database
        self.gateway, self.registry = gateway, registry

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def invoke(self, value, config=None, **kwargs):
        thread = (config or {}).get("configurable", {}).get("thread_id")
        self.gateway.reset_cancellation()
        self.registry.cancelled = False
        old_handler = None
        if threading.current_thread() is threading.main_thread():
            old_handler = signal.getsignal(signal.SIGINT)
            def cancel(signum, frame):
                self.gateway.cancel()
                self.registry.cancel()
                raise KeyboardInterrupt
            signal.signal(signal.SIGINT, cancel)
        try:
            while True:
                result = self.graph.invoke(value, config, **kwargs)
                saved = self.graph.get_state(config)
                if result.get("agent_done") or not saved.next:
                    result.pop("__interrupt__", None)
                    return result
                # Static interrupts are internal checkpoint boundaries, not questions.
                value = None
        except (KeyboardInterrupt, ExecutionCancelled):
            if thread:
                self.database.runtime.update_status(thread, "cancelled")
            raise KeyboardInterrupt from None
        except Exception as exc:
            if thread and any(code in str(exc) for code in ("Insufficient Balance", "Error code: 401", "Error code: 402", "Error code: 403")):
                self.database.runtime.update_status(thread, "blocked")
            raise
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGINT, old_handler)
