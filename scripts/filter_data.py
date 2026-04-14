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

def check_landings(c_data, start_alt, end_alt, airport, discard_reasons):
    """Check if the flight matches the landings heuristic."""
    max_end_alt = airport.altitude + 2500
    min_descent = 2000
    
    if end_alt > max_end_alt:
        discard_reasons["ends_high"] += 1
        return False
    if (start_alt - end_alt) < min_descent:
        discard_reasons["no_descent"] += 1
        return False
    return True

def check_runway14(c_data, start_alt, end_alt, airport, discard_reasons):
    """Check if the flight is a Runway 14 departure or landing."""
    target_heading = 137
    heading_tolerance = 10
    
    is_landing = (end_alt <= airport.altitude + 2500) and ((start_alt - end_alt) >= 1500)
    is_departure = (start_alt <= airport.altitude + 2500) and ((end_alt - start_alt) >= 1500)
    
    if is_landing:
        final_heading = c_data['track'].iloc[-20:].median() if len(c_data) >= 20 else c_data['track'].iloc[-1]
        heading_diff = min((final_heading - target_heading) % 360, (target_heading - final_heading) % 360)
        if heading_diff > heading_tolerance:
            discard_reasons["wrong_runway"] = discard_reasons.get("wrong_runway", 0) + 1
            return False
    elif is_departure:
        initial_heading = c_data['track'].iloc[:20].median() if len(c_data) >= 20 else c_data['track'].iloc[0]
        heading_diff = min((initial_heading - target_heading) % 360, (target_heading - initial_heading) % 360)
        if heading_diff > heading_tolerance:
            discard_reasons["wrong_runway"] = discard_reasons.get("wrong_runway", 0) + 1
            return False
    else:
        discard_reasons["not_landing_or_departure"] = discard_reasons.get("not_landing_or_departure", 0) + 1
        return False
    return True

def trim_taxi_data(flight, airport, mode):
    """Trims taxiing data while preserving actual landing/takeoff phases."""
    try:
        df = flight.data.copy()
        if len(df) == 0:
            return flight
            
        alt_col = 'geoaltitude' if 'geoaltitude' in df.columns else 'baroaltitude'
        apt_alt = airport.altitude
        target_heading = 137
        
        start_alt = df[alt_col].iloc[0]
        end_alt = df[alt_col].iloc[-1]
        
        is_landing = (end_alt <= apt_alt + 2500) and ((start_alt - end_alt) >= 1500)
        is_departure = (start_alt <= apt_alt + 2500) and ((end_alt - start_alt) >= 1500)
        
        if is_landing:
            # Search for trimming ONLY after the aircraft has descended below 250ft above apt
            in_air_mask = (df[alt_col] > apt_alt + 250)
            if in_air_mask.any():
                last_in_air_idx = df.index[in_air_mask][-1]
                df_after_descent = df.loc[last_in_air_idx:]
                
                # Check groundspeed
                mask_speed = (df_after_descent['groundspeed'] <= 35)
                idx_speed = df_after_descent.index[mask_speed][0] if mask_speed.any() else None
                
                # Check Heading deviation
                # Convert to numpy to avoid pandas-specific operator issues
                track_vals = df_after_descent['track'].values.astype(float)
                h_diffs = (track_vals - target_heading) % 360
                h_diffs = np.minimum(h_diffs, 360 - h_diffs)
                mask_heading = (h_diffs > 30)
                
                idx_heading = df_after_descent.index[mask_heading][0] if mask_heading.any() else None
                
                cut_idx = None
                if idx_speed is not None and idx_heading is not None:
                    cut_idx = min(idx_speed, idx_heading)
                elif idx_speed is not None:
                    cut_idx = idx_speed
                elif idx_heading is not None:
                    cut_idx = idx_heading
                    
                if cut_idx is not None:
                    df = df.loc[:cut_idx]
                    
        elif is_departure:
            # Search for takeoff roll start ONLY among ground points
            in_air_mask = (df[alt_col] > apt_alt + 250)
            if in_air_mask.any():
                first_in_air_idx = df.index[in_air_mask][0]
                df_before_climb = df.loc[:first_in_air_idx]
                
                mask_takeoff = (df_before_climb['groundspeed'] >= 50)
                if mask_takeoff.any():
                    idx_takeoff = df_before_climb.index[mask_takeoff][0]
                    prev_df = df_before_climb.loc[:idx_takeoff]
                    mask_start = (prev_df['groundspeed'] <= 25)
                    if mask_start.any():
                        start_idx = prev_df.index[mask_start][-1]
                        df = df.loc[start_idx:]
                    
        return Flight(df)
    except Exception as e:
        # Log and return original flight if trimming fails
        logger.debug(f"Trimming failed for flight: {e}")
        return flight

def denoise_flight(flight):
    """
    Applies high-grade denoising with local MAD detection, global speed clipping,
    and physical acceleration capping (Forward-Backward).
    """
    try:
        df = flight.data.copy()
        alt_col = 'geoaltitude' if 'geoaltitude' in df.columns else 'baroaltitude'

        # Time deltas for physics caps
        if 'timestamp' in df.columns:
            times = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            dt = times.diff().dt.total_seconds().fillna(1.0).values
        else:
            dt = np.ones(len(df))

        # 1. Physical Continuity Integrator
        def apply_physical_cap(values, deltas, max_rate, circular=False):
            capped = values.astype(float).copy()
            for pass_idx in range(2):
                indices = range(1, len(capped)) if pass_idx == 0 else range(len(capped)-2, -1, -1)
                for i in indices:
                    prev = i - 1 if pass_idx == 0 else i + 1
                    dt_val = deltas[i] if pass_idx == 0 else deltas[i+1]
                    diff = capped[i] - capped[prev]
                    if circular:
                        diff = (diff + 180) % 360 - 180
                    limit = max_rate * dt_val
                    if abs(diff) > limit:
                        step = np.sign(diff) * limit
                        capped[i] = (capped[prev] + step) % 360 if circular else capped[prev] + step
            return capped

        if 'groundspeed' in df.columns:
            # 2.5 kts/s is the realistic max for a commercial jet on approach
            df['groundspeed'] = apply_physical_cap(df['groundspeed'].values, dt, 2.5)
            # GLOBAL CLIP: Civil threshold below 10,000 ft at LSZH
            if alt_col in df.columns:
                df.loc[(df['groundspeed'] > 320) & (df[alt_col] < 10000), 'groundspeed'] = np.nan
        
        if alt_col in df.columns:
            # 80 ft/s (~4800 fpm) is more than enough for a standard civil descent
            df[alt_col] = apply_physical_cap(df[alt_col].values, dt, 80.0)

        if 'track' in df.columns:
            # 5 deg/s is a standard 'brisk' turn rate
            df['track'] = apply_physical_cap(df['track'].values, dt, 5.0, circular=True)

        # 2. Hampel / Median cleaning
        for col in ['groundspeed', alt_col, 'track']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Pre-clean with window 15 median
                df[col] = df[col].rolling(window=15, center=True).median()
                df[col] = df[col].interpolate(method='linear').ffill().bfill()

        # 3. Sin/Cos Smoothing for Track
        if 'track' in df.columns:
            track_rad = np.radians(df['track'].values)
            t_sin = pd.Series(np.sin(track_rad)).rolling(window=71, center=True).median().rolling(window=31, center=True).mean().ffill().bfill()
            t_cos = pd.Series(np.cos(track_rad)).rolling(window=71, center=True).median().rolling(window=31, center=True).mean().ffill().bfill()
            df['track'] = np.degrees(np.arctan2(t_sin.values, t_cos.values)) % 360

        # 4. Heavy Signal Smoothing
        for col in ['groundspeed', alt_col]:
            if col in df.columns:
                df[col] = df[col].rolling(window=71, center=True).median().ffill().bfill()
                df[col] = df[col].rolling(window=31, center=True).mean().ffill().bfill()
        
        new_flight = Flight(df)
        for attr in ['icao24', 'callsign', 'registration', 'typecode']:
            if hasattr(flight, attr): setattr(new_flight, attr, getattr(flight, attr))
        
        return new_flight.simplify(tolerance=1e-3)

    except Exception as e:
        logger.debug(f"Denoising failed: {e}")
        return flight

def load_manual_aircraft_db(path):
    """Loads OpenSky aircraft database from a manual CSV file."""
    if not os.path.exists(path):
        logger.warning(f"Manual aircraft database not found at {path}. Falling back to default methods.")
        return None
    
    try:
        logger.info(f"Loading manual aircraft database from {path}...")
        # Only load icao24 and typecode to save memory
        df = pd.read_csv(path, usecols=['icao24', 'typecode'], low_memory=False)
        # Drop rows with no typecode and set index for fast lookup
        df = df.dropna(subset=['typecode'])
        # Create a lowercase mapping for the lookup
        return dict(zip(df['icao24'].str.lower(), df['typecode']))
    except Exception as e:
        logger.warning(f"Failed to load manual aircraft database: {e}. Ensure icao24 and typecode columns exist.")
        return None

def filter_and_process(input_path, airport_code, output_base, mode="landings", max_flights=None, denoise=True, aircraft_db_path=None):
    """
    Loads raw traffic data, filters based on mode, resamples, and saves as .npy and .csv.
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
    
    # Load manual aircraft database if provided
    manual_db = load_manual_aircraft_db(aircraft_db_path) if aircraft_db_path else None
    
    logger.info(f"Processing {len(traffic_data)} flights for {airport_code}...")
    
    processed_tensors = []
    metadata = []
    discard_reasons = {"too_short": 0, "no_alt": 0, "ends_high": 0, "no_descent": 0, "resample_fail": 0, "unexpected_error": 0}
    
    for i, flight in enumerate(traffic_data):
        if max_flights is not None and len(processed_tensors) >= max_flights:
            logger.info(f"Reached limit of {max_flights} flights. Stopping early.")
            break

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
            
            # 3. Mode Selection
            if mode == "landings":
                if not check_landings(c_data, start_alt, end_alt, airport, discard_reasons):
                    continue
            elif mode == "runway14":
                if not check_runway14(c_data, start_alt, end_alt, airport, discard_reasons):
                    continue
            elif mode == "all":
                pass # Accept everything
            else:
                logger.error(f"Unknown mode: {mode}")
                return False
            
            # --- DENOISE DATA ---
            if denoise:
                cleaned_flight = denoise_flight(cleaned_flight)
                # --- TRIM TAXI DATA ---
                cleaned_flight = trim_taxi_data(cleaned_flight, airport, mode)
                if len(cleaned_flight) < 50:
                    discard_reasons["too_short"] += 1
                    continue
            # --------------------
                
            # 4. Resample
            resampled = cleaned_flight.resample(NUM_WAYPOINTS)
            if resampled is None:
                discard_reasons["resample_fail"] += 1
                continue
                
            r_data = resampled.data
            
            # icao24 / ac_type lookup
            icao24 = getattr(flight, 'icao24', None)
            ac_type = "UNKNOWN"
            
            # 1. Try manual DB first (Local and reliable)
            if icao24 and manual_db:
                ac_type = manual_db.get(icao24.lower(), "UNKNOWN")
            
            # 2. Fallback to traffic library (If manual DB failed or missing)
            if ac_type == "UNKNOWN" and icao24 and aircraft is not None:
                try:
                    ac_db_entry = aircraft.get(icao24)
                    if ac_db_entry is not None:
                        ac_type = ac_db_entry.get("typecode", "UNKNOWN")
                except Exception:
                    pass
            
            start_t = r_data['timestamp'].min()
            elapsed_time = (r_data['timestamp'] - start_t).dt.total_seconds().values
            
            X_i = np.vstack([
                r_data['track'].values,
                r_data['groundspeed'].values,
                r_data[alt_col].values,
                elapsed_time
            ])
            
            # --- FINAL BRUTE-FORCE PHYSICAL SAFEGUARD ---
            if denoise:
                # Ultra-realistic capping on the final 200-point array.
                total_duration = X_i[3, -1]
                dt_avg = total_duration / (NUM_WAYPOINTS - 1)
                
                for j in range(1, NUM_WAYPOINTS):
                    # 1. GS Cap (2.5 kts/s)
                    gs_diff = X_i[1, j] - X_i[1, j-1]
                    gs_limit = 2.5 * dt_avg
                    if abs(gs_diff) > gs_limit:
                        X_i[1, j] = X_i[1, j-1] + np.sign(gs_diff) * gs_limit
                    
                    # 2. Alt Cap (80 ft/s)
                    alt_diff = X_i[2, j] - X_i[2, j-1]
                    alt_limit = 80.0 * dt_avg
                    if abs(alt_diff) > alt_limit:
                        X_i[2, j] = X_i[2, j-1] + np.sign(alt_diff) * alt_limit
                    
                    # 3. Track Cap (5 deg/s)
                    tr_diff = (X_i[0, j] - X_i[0, j-1] + 180) % 360 - 180
                    tr_limit = 5.0 * dt_avg
                    if abs(tr_diff) > tr_limit:
                        X_i[0, j] = (X_i[0, j-1] + np.sign(tr_diff) * tr_limit) % 360
            
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
    # Path to the raw .parquet file
    INPUT_FILE = "data/raw/LSZH_2019_10_01_to_11_30_benchmark2.parquet"
    
    AIRPORT = "LSZH"
    MODE = "runway14"  # Options: "landings", "runway14", "all"
    DENOISE = True # Set to True to clean data, False to see raw noise
    MAX_FLIGHTS = 500  # Set to None for full processing, or a number for testing
    
    # Optional: Path to a manually downloaded OpenSky aircraft database CSV
    # Download from: https://s3.opensky-network.org/data-samples/metadata/aircraft-database-complete-2025-08.csv
    AIRCRAFT_DB_FILE = "data/aircraft_db.csv"
    
    # Custom output name
    suffix = "raw" if not DENOISE else "denoised"
    OUTPUT_BASE = f"X_{AIRPORT}_2019_benchmark_{MODE}_{suffix}"
    # --------------------------

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_input_path = os.path.join(base_path, INPUT_FILE)
    full_db_path = os.path.join(base_path, AIRCRAFT_DB_FILE)
    
    filter_and_process(full_input_path, AIRPORT, OUTPUT_BASE, 
                       mode=MODE, max_flights=MAX_FLIGHTS, denoise=DENOISE,
                       aircraft_db_path=full_db_path)
