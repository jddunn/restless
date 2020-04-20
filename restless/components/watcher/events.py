import os, sys
import asyncio
import time

from watchdog.events import FileSystemEventHandler

EVENT_TYPE_MOVED = "moved"
EVENT_TYPE_DELETED = "deleted"
EVENT_TYPE_CREATED = "created"
EVENT_TYPE_MODIFIED = "modified"


class AsyncEventHandler(FileSystemEventHandler):
    def __init__(self, loop=None, event_cb=None):
        self._loop = loop or asyncio.get_event_loop()
        # prefer asyncio.create_task starting from Python 3.7
        if hasattr(asyncio, "create_task"):
            self._ensure_future = asyncio.create_task
        else:
            self._ensure_future = asyncio.ensure_future
        self.event_cb = event_cb

    def dispatch(self, event):
        _method_map = {
            "modified": self.on_modified,
            "moved": self.on_moved,
            "created": self.on_created,
            "deleted": self.on_deleted,
        }
        _method = _method_map[event.event_type]
        self._loop.run_until_complete(_method(event))
        # self._loop.call_soon_threadsafe(self._ensure_future, _method(event))

    async def on_moved(self, event):
        time.sleep(0.25)
        await self.event_cb(event.src_path)
        pass

    async def on_created(self, event):
        time.sleep(0.25)
        await self.event_cb(event.src_path)
        pass

    async def on_deleted(self, event):
        pass

    async def on_modified(self, event):
        time.sleep(0.25)
        pass
