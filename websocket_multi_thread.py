import threading
import asyncio
import websockets
import json
import numpy as np

class WebSocketThread(threading.Thread):
    def __init__(self, uri="ws://172.26.226.69:3000/robot", status=dict()):
        threading.Thread.__init__(self)
        self.uri = uri
        self.status = status

    async def eventLoop(self):
        while True:
            self.websocket = await websockets.connect(self.uri)
            try:
                await self.send()
                await self.recv()
                await asyncio.sleep(0.5)
            except:
                print('except!')
            # await self.send()
            # await self.recv()
            # await asyncio.sleep(0.5)

    async def recv(self):
        greeting = await self.websocket.recv()
        statusDict = json.loads(greeting)
        print(statusDict)
        # Command and path
        if statusDict['command'] == 'maintenance':
            self.status['command'] = 'maintenance'
        elif statusDict['command'] == 'startDelivery':
            self.status['command'] = 'move'
            self.status['path'] = statusDict['path']
        elif statusDict['command'] == 'keepDelivery' or statusDict['command'] == 'move':
            self.status['command'] = 'move'
        else:
            self.status['command'] = 'empty'

    async def send(self):

        await self.websocket.send(json.dumps(self.status))

    def run(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.eventLoop())
        loop.close()

if __name__=='__main__':
    thread1 = WebSocketThread()
    thread1.start()
    thread1.join()