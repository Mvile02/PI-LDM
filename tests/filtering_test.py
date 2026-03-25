import pandas as pd
from traffic.core import Traffic

# 1. Path to LSZH "2026-03-01 10:00" to "2026-03-01 12:00" cached OpenSky parquet file
cache_filepath = r"C:\Users\usuario\AppData\Local\opensky\opensky\Cache\2e5494993196293f8f40bbc1c3b2caa2.parquet"

# 2. Load the raw data into a Pandas DataFrame
raw_df = pd.read_parquet(cache_filepath)

# 3. Convert it back into a powerful `traffic` object!
# This gives you access to all the built-in library methods again
traffic_data = Traffic(raw_df)
print(f"Loaded {len(traffic_data)} state vectors instantly!")

# --- NOW YOU CAN EXPERIMENT WITH FILTERS ---

# Example A: Filter for a specific aircraft type (e.g., only A320s)
# Note: You have to join with the aircraft DB first if you want typecodes natively
# Or filter by a specific callsign
specific_flight = traffic_data["SWR339H"]
if specific_flight:
    # Plot it to see what the raw trajectory looks like before processing!
    import matplotlib.pyplot as plt
    from traffic.drawing import fastip
    
    fig, ax = plt.subplots(figsize=(10, 10))
    specific_flight.plot(ax, cmap="viridis")
    plt.show()

# Example B: Only keep data points where altitude < 10,000 ft (closer to the runway)
low_altitude_traffic = traffic_data.query("geoaltitude < 10000 or baroaltitude < 10000")

# Example C: Use traffic's built-in .descent() heuristics instead of our custom ones
for flight in low_altitude_traffic:
    descent = flight.descent()
    if descent:
        print(f"Flight {flight.callsign} descended!")
