import os
import argparse
import logging
import pandas as pd
import numpy as np
from traffic.core import Traffic, Flight
from traffic.data import airports, aircraft

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
NUM_WAYPOINTS = 200

def filter_and_process(input_path, airport_code, output_base):
    """
    Loads raw traffic data, filters for landings, resamples, and saves as .npy and .csv.
    """
    logger.info(f"Loading raw data from {input_path}...")
    try:
        # Load parquet
        raw_df = pd.read_parquet(input_path)
        
        # --- Column Normalization ---
        # Map common OpenSky/Traffic column name variations
        rename_dict = {
            'time': 'timestamp',
            'lat': 'latitude',
            'lon': 'longitude',
            'baro_altitude': 'baroaltitude',
            'vertrate': 'vertical_rate',
            'velocity': 'groundspeed',
            'heading': 'track'
        }
        to_rename = {k: v for k, v in rename_dict.items() if k in raw_df.columns and v not in raw_df.columns}
        if to_rename:
            logger.info(f"Normalizing columns: {to_rename}")
            raw_df = raw_df.rename(columns=to_rename)
            
        # Ensure timestamp is datetime (raw OpenSky uses unix timestamps)
        if 'timestamp' in raw_df.columns and not pd.api.types.is_datetime64_any_dtype(raw_df['timestamp']):
            raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], unit='s', utc=True)
            
        traffic_data = Traffic(raw_df)
    except Exception as e:
        logger.error(f"Failed to load or normalize raw data: {e}")
        return False

    airport = airports[airport_code]
    if airport is None:
        logger.error(f"Airport code {airport_code} not found.")
        return False
    
    logger.info(f"Processing {len(traffic_data)} flights for {airport_code}...")
    
    processed_tensors = []
    metadata = []
    discard_reasons = {"too_short": 0, "no_alt": 0, "ends_high": 0, "no_descent": 0, "resample_fail": 0, "unexpected_error": 0}
    
    for i, flight in enumerate(traffic_data):
        if i % 100 == 0:
            logger.info(f"Progress: {i}/{len(traffic_data)} flights... Current discards: {discard_reasons}")
            
        # 1. Basic length check
        if len(flight) < 50:
            discard_reasons["too_short"] += 1
            continue
            
        data = flight.data
        alt_col = 'geoaltitude' if 'geoaltitude' in data.columns else 'baroaltitude'
        if alt_col not in data.columns:
            discard_reasons["no_alt"] += 1
            continue
            
        # 2. Drop NaNs and resample/filter
        try:
            from traffic.core import Flight
            flight_df = flight.data.dropna(subset=['track', 'groundspeed', alt_col])
            if len(flight_df) < 50:
                discard_reasons["too_short"] += 1
                continue
            
            flight = Flight(flight_df)
            cleaned_flight = flight.filter()
            if len(cleaned_flight) < 50:
                discard_reasons["too_short"] += 1
                continue
                
            c_data = cleaned_flight.data
            start_alt = c_data[alt_col].iloc[0]
            end_alt = c_data[alt_col].iloc[-1]
            
            # 3. Landing Heuristic:
            # Must end near airport and descend significantly
            max_end_alt = airport.altitude + 2500
            min_descent = 2000 # Loosened even more for testing
            
            if end_alt > max_end_alt:
                discard_reasons["ends_high"] += 1
                continue
            if (start_alt - end_alt) < min_descent:
                discard_reasons["no_descent"] += 1
                continue
                
            # 4. Resample
            resampled = cleaned_flight.resample(NUM_WAYPOINTS)
            if resampled is None:
                discard_reasons["resample_fail"] += 1
                continue
                
            r_data = resampled.data
            
            # 5. Extract Aircraft Type
            icao24 = getattr(flight, 'icao24', None)
            ac_type = "UNKNOWN"
            if icao24:
                ac_db_entry = aircraft.get(icao24)
                if ac_db_entry is not None:
                    ac_type = ac_db_entry.get("typecode", "UNKNOWN")
            
            # 6. Build Tensor (4 x 200)
            start_t = r_data['timestamp'].min()
            elapsed_time = (r_data['timestamp'] - start_t).dt.total_seconds().values
            
            X_i = np.vstack([
                r_data['track'].values,
                r_data['groundspeed'].values,
                r_data[alt_col].values,
                elapsed_time
            ])
            
            # 7. Final validation
            if X_i.shape == (4, NUM_WAYPOINTS) and not np.isnan(X_i).any():
                processed_tensors.append(X_i)
                metadata.append({
                    "callsign": flight.callsign,
                    "icao24": icao24,
                    "typecode": ac_type,
                    "airport": airport_code,
                    "timestamp": r_data['timestamp'].iloc[0]
                })
                logger.info(f"Processed flight {flight.callsign} ({ac_type})")
                
        except Exception as e:
            discard_reasons["unexpected_error"] += 1
            if discard_reasons["unexpected_error"] < 10:
                logger.warning(f"Failed to process flight {getattr(flight, 'callsign', '?')}: {e}")
            continue



    if not processed_tensors:
        logger.warning(f"No valid landing trajectories found. Discard reasons: {discard_reasons}")
        return False

    logger.info(f"Discard statistics: {discard_reasons}")


    # Stack to (Batch, 4, 200)
    X_batch = np.stack(processed_tensors, axis=0)
    df_meta = pd.DataFrame(metadata)
    
    # Save results
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    npy_path = os.path.join(processed_dir, f"{output_base}.npy")
    csv_path = os.path.join(processed_dir, f"{output_base}.csv")
    
    np.save(npy_path, X_batch)
    df_meta.to_csv(csv_path, index=False)
    
    logger.info(f"Saved {len(X_batch)} trajectories to {npy_path} and {csv_path}")
    return True

if __name__ == "__main__":
    # --- CONFIGURATION AREA ---
    # Path to the raw .parquet file generated by extract_data.py
    # INPUT_FILE = "data/raw/raw_LSZH_2026-03-01_1000_to_2026-03-01_1200.parquet"
    INPUT_FILE = r"C:\Users\usuario\AppData\Local\opensky\opensky\Cache\2e5494993196293f8f40bbc1c3b2caa2.parquet"
    # START_TIME = INPUT_FILE.split("_")[2] + "_" + INPUT_FILE.split("_")[3]
    # END_TIME = INPUT_FILE.split("_")[5] + "_" + INPUT_FILE.split("_")[6]
    # AIRPORT = INPUT_FILE.split("_")[1]
    # OUTPUT_BASE = f"X_{AIRPORT}_{START_TIME}_to_{END_TIME}_processed"
    AIRPORT = "LSZH"
    OUTPUT_BASE = "TEST"
    # --------------------------

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_input_path = os.path.join(base_path, INPUT_FILE)
    
    filter_and_process(full_input_path, AIRPORT, OUTPUT_BASE)

