import threading
import asyncio
import websockets

class WebSocketThread(threading.Thread):
    def __init__(self, uri="ws://localhost:3000/inventory_manager/", status=None):
        threading.Thread.__init__(self)
        self.uri = uri
        self.status = status

    async def eventLoop(self):
        while True:
            self.websocket = await websockets.connect(self.uri)
            try:
                await asyncio.sleep(1)
                await self.send()
                await self.recv()
            except:
                print('except!')

    async def recv(self):
        greeting = await self.websocket.recv()
        print(f"< {greeting}")

    async def send(self):
        name = input("Enter message.\n")
        await self.websocket.send(name)

    def run(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.eventLoop())
        loop.close()

if __name__=='__main__':
    thread1 = WebSocketThread()
    thread1.start()
    thread1.join()
