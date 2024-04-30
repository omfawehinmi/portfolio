import uasyncio as asyncio
import time

class TimeoutError(Exception):
    pass

class Queue:
    def __init__(self):
        self.items = []
        self.lock = asyncio.Lock()

    async def enqueue(self, item):
        async with self.lock:
            self.items.append(item)

    async def dequeue(self, timeout=None):
        start_time = time.time()
        while True:
            async with self.lock:
                if not self.is_empty():
                    return self.items.pop(0)
            if timeout is not None and time.time() - start_time >= timeout:
                raise TimeoutError("Queue dequeue operation timed out")
            await asyncio.sleep(0.1)

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)