from hachiko.hachiko import AIOEventHandler

class AsyncFileClassifyEventHandler(AIOEventHandler):
    """
    Subclass of asyncio-compatible event handler.
    From example: https://github.com/biesnecker/hachiko.
    """
    async def on_created(self, event):
        print('Created:', event.src_path)

    # If a file was moved, we should go ahead and rescan anyway
    async def on_moved(self, event):
        print('Moved:', event.src_path)

    async def on_modified(self, event):
        print('Modified:', event.src_path)
