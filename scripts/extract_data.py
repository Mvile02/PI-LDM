import os
import argparse
import logging
import pandas as pd
from traffic.data import opensky
from traffic.data import airports
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load credentials from .env in the root directory
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

def extract_raw_data(airport_code, start_time, end_time, output_file):
    """
    Downloads historical ADS-B data from OpenSky and saves it to a Parquet file.
    """
    airport = airports[airport_code]
    if airport is None:
        logger.error(f"Airport code {airport_code} not found in traffic databases.")
        return False
    
    logger.info(f"Downloading historical traffic around {airport_code} between {start_time} and {end_time}")
    
    # Define a bounding box around the airport (+/- 1.5 degrees)
    lat, lon = airport.latlon
    bounds = (lon - 1.5, lat - 1.5, lon + 1.5, lat + 1.5)
    
    try:
        traffic_data = opensky.history(
            start=start_time,
            stop=end_time,
            bounds=bounds
        )
        
        if traffic_data is None or len(traffic_data) == 0:
            logger.warning(f"No traffic data found for {airport_code} in the specified timeframe.")
            return False
            
        logger.info(f"Downloaded {len(traffic_data)} state vectors. Saving to {output_file}...")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the traffic data to parquet
        traffic_data.data.to_parquet(output_file)
        logger.info(f"Successfully saved raw data to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return False

if __name__ == "__main__":
    # --- CONFIGURATION AREA ---
    AIRPORT = "LSZH"
    START_TIME = "2026-03-01 00:00" # YYYY-MM-DD HH:MM
    END_TIME = "2026-03-01 23:59"
    OUTPUT_FILE = None  # Set to a filename string to override default
    # --------------------------

    # Default output path if not specified
    if not OUTPUT_FILE:
        start_ts = START_TIME.replace(" ", "_").replace(":", "")
        end_ts = END_TIME.replace(" ", "_").replace(":", "")
        OUTPUT_FILE = f"raw_{AIRPORT}_{start_ts}_to_{end_ts}.parquet"
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_path, "data", "raw", OUTPUT_FILE)
    
    extract_raw_data(AIRPORT, START_TIME, END_TIME, output_path)


