import os
from typing import Union, List
import logging
import json
from pathlib import Path

import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc import MediaStreamTrack
from av import VideoFrame
import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from aiortc.sdp import candidate_from_sdp
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import torch

from app.model import Model
from app.detrv2 import DETRv2
from app.dav2 import DAv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.connections = {}
        self.models = {}

    async def connect(self, websocket) -> str:
        await websocket.accept()
        client_id = str(uuid4())
        self.connections[client_id] = {'socket': websocket, 'model': None}
        return client_id

    def disconnect(self, client_id):
        del self.connections[client_id]

    async def send_json(self, client_id, message):
        await self.connections[client_id]['socket'].send_json(message)

manager = ConnectionManager()

# Model information
MODEL_DIR = Path(os.path.join(Path(__file__).parent.parent, 'models'))
MODEL_PATHS = sorted(list(Path(MODEL_DIR).rglob("*.engine")))
COCO_DIR = Path(os.path.join(Path(__file__).parent.parent, 'datasets', 'coco'))

class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, client_id):
        super().__init__()
        self.track = track 
        self.client_id = client_id
        
        # Use a single persistent thread to avoid thread creation overhead
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # State management
        self.last_processed_result = None
        self.process_future = None  # Use Future directly, not Task
        
        # Pre-allocate reusable buffers (will be sized on first frame)
        self._input_buffer = None
        self._target_dim = 518
        
        # Cache model reference to avoid dict lookups every frame
        self._cached_model = None
        self._cached_model_name = None

    def process_job(self, frame_array, model):
        """ Runs in a separate thread. Optimized for minimal allocations """
        try:
            h, w = frame_array.shape[:2]
            max_dim = max(h, w)
            
            # Only resize if beneficial
            if max_dim > self._target_dim:
                scale = self._target_dim / max_dim
                new_w, new_h = int(w * scale), int(h * scale)
                input_img = cv2.resize(frame_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                input_img = frame_array
                scale = 1.0

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            input_img = input_img.astype(np.float32) / 255.0
            input_img = (input_img - mean) / std

            input_tensor = np.ascontiguousarray(input_img.transpose(2, 0, 1))

            # Inference
            inference = model(input_tensor)
            output = model.postprocess(img=input_tensor, inference=inference)[0]

            # Resize
            if scale < 1.0:
                output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LINEAR)
            
            return output

        except Exception as e:
            print(f"Inference error: {e}")
            return None  # Return None to signal error, not the whole frame

    async def recv(self):
        frame = await self.track.recv()
        
        # Cache model lookup - only query dicts when model changes
        conn_data = manager.connections.get(self.client_id)
        model_name = conn_data.get('model') if conn_data else None
        
        if model_name != self._cached_model_name:
            self._cached_model = manager.models.get(model_name) if model_name else None
            self._cached_model_name = model_name
            # Clear stale results on model change
            if self._cached_model is None:
                self.last_processed_result = None
        
        model = self._cached_model

        # Pass-through if no model
        if not model:
            return frame

        # Check if previous job completed
        if self.process_future is not None:
            if self.process_future.done():
                try:
                    result = self.process_future.result()
                    if result is not None:
                        self.last_processed_result = result
                except Exception:
                    pass
                self.process_future = None

        # Start new job only if none is running
        if self.process_future is None:
            img_array = frame.to_ndarray(format="bgr24")
            self.process_future = self.executor.submit(
                self.process_job,
                img_array,
                model
            )

        # Return cached result or passthrough
        if self.last_processed_result is not None:
            new_frame = VideoFrame.from_ndarray(self.last_processed_result, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        
        return frame

@app.get("/")
def root():
    return {"message": "Backend is running"}

@app.get("/models")
def get_models() -> List[str]:
    return [ x.parent.stem for x in MODEL_PATHS ]

PCS = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Add to connection manager
    client_id = await manager.connect(websocket)
    await manager.send_json(client_id, {"type": "id", "id": client_id})

    pc = RTCPeerConnection()
    pc.clientId = client_id
    PCS.add(pc)

    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate is not None:
            await websocket.send_json({
                "type": "candidate",
                "candidate": candidate.to_json()
            })

    track_event = asyncio.Event()

    @pc.on("track")
    def on_track(track):
        # You can add the track to a custom processor if needed
        if track.kind == "video":
            track_event.set()
            pc.addTrack(VideoTransformTrack(track, pc.clientId))
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message['type'] == 'offer':
                offer = RTCSessionDescription(sdp=message['sdp'], type="offer")
                await pc.setRemoteDescription(offer)
                await track_event.wait()
                
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await websocket.send_json({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp
                })
            elif message['type'] == 'candidate':
                candidate = message["candidate"]
                if candidate:
                    rtc_ice_candidate = candidate_from_sdp(candidate['candidate'])
                    rtc_ice_candidate.sdpMid = candidate.get('sdpMid')
                    rtc_ice_candidate.sdpMLineIndex = candidate.get('sdpMLineIndex')
                    await pc.addIceCandidate(rtc_ice_candidate)
            # Set model
            elif message['type'] == 'model':
                # Remove access to other models
                manager.connections[message['clientId']]['model'] = None
                
                model_path = MODEL_DIR / Path(message['name']) / "model.engine"

                if model_path in MODEL_PATHS:
                    try:
                        manager.models[message['name']] = Model(model_path, COCO_DIR / "coco.yaml")
                    
                        # Give client access
                        manager.connections[client_id]['model'] = message['name']

                        # Destroy model if not being used
                        num_model_refs = list(filter(lambda x: x['model'] == message['name'], manager.connections.values()))
                        if len(num_model_refs) == 0:
                            del manager.models[message['name']]
                    except:
                        pass
                        
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        await pc.close()
        PCS.remove(pc)
        manager.disconnect(client_id)