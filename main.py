from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import ee
import os
import requests
import logging
from datetime import datetime
from PIL import Image
import io
import numpy as np

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create WebSocket connection storage
websocket_connections = []

os.makedirs("static/images", exist_ok=True)

# Initialize Google Earth Engine
def init_earth_engine():
    try:
        ee.Initialize(project='planar-chassis-388913')
        logger.info("Earth Engine initialized successfully.")
    except ee.EEException:
        logger.info("Earth Engine authentication required. Attempting to authenticate...")
        ee.Authenticate()
        ee.Initialize(project='planar-chassis-388913')
        logger.info("Earth Engine initialized after authentication.")


init_earth_engine()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Function to send logs via WebSocket
async def send_log(message):
    logger.info(message)
    for ws in websocket_connections:
        await ws.send_text(message)

# Function to fetch satellite images and perform NDVI calculations
def fetch_satellite_image(lat, lon, area_meters, start_date, end_date):
    area_km = np.sqrt(area_meters) / 1000
    region = ee.Geometry.Rectangle([
        lon - area_km / 2, lat - area_km / 2, 
        lon + area_km / 2, lat + area_km / 2
    ])
    
    image = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .sort("CLOUDY_PIXEL_PERCENTAGE") \
        .first()

    return image.clip(region), region

# Save Image to Disk
def save_image(image, filename):
    thumb_url = image.getThumbURL({'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']})
    response = requests.get(thumb_url)
    img = Image.open(io.BytesIO(response.content))
    img.save(f"static/images/{filename}.png")

# Process Image Tiles
async def process_data(lat, lon, area_meters):
    current_year = datetime.now().year
    scores = []
    images = []

    for year in range(current_year - 3, current_year + 1):
        image, region = fetch_satellite_image(lat, lon, area_meters, f"{year}-01-01", f"{year}-12-31")
        await send_log(f"""Fetching Images : Lat : {lat}, Lon : {lon} ;
For the dates - Start Date : {year}-01-01 --> End Date : {year}-12-31 """)
        save_image(image, f"satellite_{year}")
        await send_log(f"Image Saved : static/images/satellite_{year}.png")
        images.append(f"static/images/satellite_{year}.png")

        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        tile_size = 500  # Tile size in meters
        num_tiles = int(np.ceil(area_meters / tile_size))
        await send_log(f"Dividing the Entire area by 500 units --> Total Nos. : {num_tiles}")

        tile_scores = []
        for i in range(num_tiles):
            await send_log(f"Processing : {i}th Tile out of {num_tiles} Tiles")
            for j in range(num_tiles):
                sub_region = ee.Geometry.Rectangle([
                    lon - (i * tile_size) / 1000, lat - (j * tile_size) / 1000, 
                    lon - ((i + 1) * tile_size) / 1000, lat - ((j + 1) * tile_size) / 1000
                ])
                score = ndvi.reduceRegion(ee.Reducer.mean(), sub_region, 30).get("NDVI").getInfo()
                if score is not None:
                    await send_log(f"Processed : {i}th Tile out of {num_tiles} Tiles ; Score : {score}")
                    tile_scores.append(score)

        avg_score = sum(tile_scores) / len(tile_scores) if tile_scores else None
        scores.append((year, avg_score))
        await send_log(f"Calculation Success for {year} ; Total Score : {avg_score}")

    return scores, images

# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/calculate")
async def calculate(request: Request, lat: float = Form(...), lon: float = Form(...), area: int = Form(...)):
    await send_log(f"Processing request - Lat: {lat}, Lon: {lon}, Area: {area} sq. meters")
    
    scores, images = await process_data(lat, lon, area)

    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "scores": scores, "images": images}
    )
