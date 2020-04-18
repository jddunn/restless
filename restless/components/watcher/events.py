import asyncio

from watchdog.events import FileSystemEventHandler

EVENT_TYPE_MOVED = "moved"
EVENT_TYPE_DELETED = "deleted"
EVENT_TYPE_CREATED = "created"
EVENT_TYPE_MODIFIED = "modified"


class AIOEventHandler(FileSystemEventHandler):
    """An asyncio-compatible event handler."""

    def __init__(self, loop=None):
        self._loop = loop or asyncio.get_event_loop()
        # prefer asyncio.create_task starting from Python 3.7
        if hasattr(asyncio, "create_task"):
            self._ensure_future = asyncio.create_task
        else:
            self._ensure_future = asyncio.ensure_future

    async def on_any_event(self, event):
        pass

    async def on_moved(self, event):
        pass

    async def on_created(self, event):
        pass

    async def on_deleted(self, event):
        pass

    async def on_modified(self, event):
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


class AsyncFileClassifyEventHandler(AIOEventHandler):
    """
    Subclass of asyncio-compatible event handler.
    From example: https://github.com/biesnecker/hachiko.
    """

    def __init__(self):
        parent = super(AsyncFileClassifyEventHandler).__init__()
        print(parent)

    async def on_created(self, event):
        # print("Created:", event.src_path)
        pass

    # If a file was moved, we should go ahead and rescan anyway
    async def on_moved(self, event):
        # print("Moved:", event.src_path)
        pass

    async def on_modified(self, event):
        # print("Modified:", event.src_path)
        pass


class EventHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        print("EVENT")
        print(event.event_type)
        print(event.src_path)
        print()
