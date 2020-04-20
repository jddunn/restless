import os, sys
import asyncio

from watchdog.events import FileSystemEventHandler

EVENT_TYPE_MOVED = "moved"
EVENT_TYPE_DELETED = "deleted"
EVENT_TYPE_CREATED = "created"
EVENT_TYPE_MODIFIED = "modified"


class AIOEventHandler(FileSystemEventHandler):
    """An asyncio-compatible event handler."""

    def __init__(self, loop=None, event_cb = None):
        self._loop = loop or asyncio.get_event_loop()
        # prefer asyncio.create_task starting from Python 3.7
        if hasattr(asyncio, "create_task"):
            self._ensure_future = asyncio.create_task
        else:
            self._ensure_future = asyncio.ensure_future
        self.event_cb = event_cb

    async def on_any_event(self, event):
        print(event)
        pass

    async def on_moved(self, event):
        print(event)
        await self.event_cb(event.src_path)
        pass

    async def on_created(self, event):
        print(event)
        await self.event_cb(event.src_path)
        pass

    async def on_deleted(self, event):
        print(event)
        pass

    async def on_modified(self, event):
        print(event)
        pass

    def dispatch(self, event):
        _method_map = {
            EVENT_TYPE_MODIFIED: self.on_modified,
            EVENT_TYPE_MOVED: self.on_moved,
            EVENT_TYPE_CREATED: self.on_created,
            EVENT_TYPE_DELETED: self.on_deleted,
        }
        handlers = [self.on_any_event, _method_map[event.event_type]]
        for handler in handlers:
            self._loop.call_soon_threadsafe(self._ensure_future, handler(event))

class EventHandler(FileSystemEventHandler):

    def __init__(self, loop=None, event_cb = None):
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
        asyncio.run_coroutine_threadsafe(_method(event), self._loop)
        # self._loop.call_soon_threadsafe(self._ensure_future, _method)
        _method(event)

    async def on_moved(self, event):
        print("EVENT: ", event)
        await self.event_cb(event.src_path)
        pass

    async def on_created(self, event):
        print("EVENT: ", event)
        await self.event_cb(event.src_path)
        pass

    async def on_deleted(self, event):
        print("EVENT: ", event)
        pass

    async def on_modified(self, event):
        print("EVENT: ", event)
        pass

