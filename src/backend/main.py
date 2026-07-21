import json
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from classifier import Classifier

app = FastAPI(title="Gait Analysis System")
classifier = Classifier()


class ConnectionManager:
    def __init__(self):
        self.clients = []
        self.mobile = None

    async def connect_client(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)

    async def connect_mobile(self, ws: WebSocket):
        await ws.accept()
        self.mobile = ws

    def disconnect(self, ws: WebSocket):
        if ws in self.clients:
            self.clients.remove(ws)
        if self.mobile == ws:
            self.mobile = None

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.clients:
                self.clients.remove(ws)


manager = ConnectionManager()

root_dir = Path(__file__).resolve().parents[2]
frontend_dir = root_dir / "src" / "frontend"


@app.websocket("/ws/mobile")
async def mobile_ws(websocket: WebSocket):
    await manager.connect_mobile(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            packet = json.loads(data)
            ptype = packet.get("type")
            if ptype == "start":
                classifier.reset_session()
                await manager.broadcast({"event": "session_start", "server_time": datetime.now().isoformat()})
                continue
            if ptype == "stop":
                final = classifier.end_session()
                final["server_time"] = datetime.now().isoformat()
                await manager.broadcast(final)
                continue
            result = classifier.predict(packet)
            result["server_time"] = datetime.now().isoformat()
            await manager.broadcast(result)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[Mobile WS Error] {e}")
        manager.disconnect(websocket)


@app.websocket("/ws/client")
async def client_ws(websocket: WebSocket):
    await manager.connect_client(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
def index():
    return FileResponse(str(frontend_dir / "index.html"))


@app.get("/mobile")
def mobile_page():
    return FileResponse(str(frontend_dir / "mobile.html"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)