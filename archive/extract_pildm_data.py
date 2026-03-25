import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from traffic.data import opensky
from traffic.data import airports
from traffic.core import Traffic

from dotenv import load_dotenv

# Load credentials from .env file
# pyopensky will automatically read OPENSKY_USERNAME and OPENSKY_PASSWORD
# from os.environ when they are set by load_dotenv()
load_dotenv()



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants parameterizing the PI-LDM input shape
NUM_WAYPOINTS = 200

def download_and_process_landings(airport_code: str, start_time: str, end_time: str) -> np.ndarray:
    """
    Downloads historical ADS-B data from OpenSky, filters for landing trajectories
    at the specified airport, cleans/resamples them, and converts them into the 
    PI-LDM matrix representation (X in R^(4 x N)).
    """
    airport = airports[airport_code]
    if airport is None:
        raise ValueError(f"Airport code {airport_code} not found in traffic databases.")
    
    logger.info(f"Downloading historical traffic around {airport_code} between {start_time} and {end_time}")
    
    # Define a bounding box around the airport (e.g., +/- 1 degree lat/lon)
    # Using the airport's bounds or a standard radius
    # For a descent, we roughly want 50-100 NM (approx 1-2 degrees)
    lat, lon = airport.latlon
    bounds = (lon - 1.5, lat - 1.5, lon + 1.5, lat + 1.5)
    
    # Query OpenSky. return_flight=False gives a Traffic object containing all msgs
    traffic_data = opensky.history(
        start=start_time,
        stop=end_time,
        bounds=bounds
    )
    
    if traffic_data is None or len(traffic_data) == 0:
        logger.warning(f"No traffic data found for {airport_code} in the specified timeframe.")
        return np.array([])
        
    logger.info(f"Downloaded {len(traffic_data)} state vectors. Filtering landing trajectories...")
    
    # We define "landing" as flights that end near the airport and below a certain altitude.
    # The traffic library has utilities for this, or we can use custom filtering.
    # Let's extract individual flights
    
    processed_tensors = []
    metadata = []
    
    for flight in traffic_data:
        # A quick heuristic to check if the flight lands: 
        # 1. Does it descend?
        # 2. Does it end near the airport elevation?
        
        # Flight must have enough data points
        if len(flight) < 50:
            continue
            
        data = flight.data
        if 'geoaltitude' not in data.columns and 'baroaltitude' not in data.columns:
            continue
            
        alt_col = 'geoaltitude' if 'geoaltitude' in data.columns else 'baroaltitude'
        
        # Drop NaNs in essential columns by working directly on the pandas dataframe (.data)
        flight = flight.drop(columns=[c for c in ['track', 'groundspeed', alt_col] if c not in data.columns])
        
        # We need to re-assign flight.data to handle the pandas dropna
        flight.data = flight.data.dropna(subset=['track', 'groundspeed', alt_col])

        if len(flight) < 50:
            continue

        # Clean and smooth first to remove sensor glitches/outliers
        try:
            cleaned_flight = flight.filter() # basic median filter on positions/altitudes
            if len(cleaned_flight) < 50:
                continue
                
            c_data = cleaned_flight.data
            start_alt = c_data[alt_col].iloc[0]
            end_alt = c_data[alt_col].iloc[-1]
            
            # Robust filtering for LANDINGS:
            # 1. It must end at a low altitude near the airport (within 2000ft)
            # 2. It must be a significant descent (at least 5000ft lower at end than start)
            if end_alt > (airport.altitude + 2000) or (start_alt - end_alt) < 5000:
                # This excludes overflights, takeoffs, and high-altitude fragments
                continue
                
            # Resample for 200 points uniformly distributed over the descent time
            resampled = cleaned_flight.resample(NUM_WAYPOINTS)
            
            if resampled is None:
                continue
                
            r_data = resampled.data
            
            # Extract Aircraft Type logic using traffic.data.aircraft
            icao24 = getattr(flight, 'icao24', None)
            ac_type = "UNKNOWN"
            if icao24:
                from traffic.data import aircraft
                ac_db_entry = aircraft.get(icao24)
                if ac_db_entry is not None:
                    ac_type = ac_db_entry.get("typecode", "UNKNOWN")
            
            # Calculate Elapsed Time (delta t)
            # Assuming timestamps are datetimes
            start_t = r_data['timestamp'].min()
            elapsed_time = (r_data['timestamp'] - start_t).dt.total_seconds().values
            
            track = r_data['track'].values
            gs = r_data['groundspeed'].values
            alt = r_data[alt_col].values
            
            # Construct feature matrix (4 x 200)
            # X_i = [Track, GS, Alt, dt]
            X_i = np.vstack([
                track,
                gs,
                alt,
                elapsed_time
            ])
            
            # Basic validation: ensure shape is correct and no NaNs
            if X_i.shape == (4, NUM_WAYPOINTS) and not np.isnan(X_i).any():
                processed_tensors.append(X_i)
                metadata.append({
                    "callsign": flight.callsign,
                    "icao24": icao24,
                    "typecode": ac_type,
                    "airport": airport_code
                })
                logger.info(f"Successfully processed flight {flight.callsign} ({ac_type}) into target trajectory.")
                
        except Exception as e:
            logger.debug(f"Failed to process flight {flight.callsign}: {e}")
            continue

    if not processed_tensors:
         return np.array([]), pd.DataFrame()

    # Stack to shape (Batch, 4, 200)
    X_batch = np.stack(processed_tensors, axis=0)
    df_meta = pd.DataFrame(metadata)
    
    return X_batch, df_meta

if __name__ == "__main__":
    # Test execution for Zurich (LSZH) for a specific 2-hour window
    # Make sure Trino is running and configured
    start_time = "2026-03-01 10:00"
    end_time = "2026-03-01 12:00"
    
    X_lszh, meta_lszh = download_and_process_landings("LSZH", start_time, end_time)
    
    if len(X_lszh) > 0:
        logger.info(f"Final Tensor Shape for LSZH: {X_lszh.shape}")
        
        # Save tensors internally
        np.save("data/processed/X_lszh_sample.npy", X_lszh)
        meta_lszh.to_csv("data/processed/meta_lszh_sample.csv", index=False)
        logger.info("Saved samples to disk.")
    else:
        logger.warning("No valid landing trajectories compiled.")
