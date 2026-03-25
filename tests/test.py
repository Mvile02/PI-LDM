from traffic.data import opensky
from datetime import timedelta

# Query a specific flight by callsign
# This returns a 'Flight' object if return_flight=True
flight = opensky.history(
    start="2026-03-01 10:00",
    stop="2026-03-01 11:00",
    callsign="IBE3245",
    return_flight=True
)

# Query all traffic in a bounding box (West, South, East, North)
# This returns a 'Traffic' object (a collection of flights)
t_madrid = opensky.history(
    start="2026-03-05 14:00",
    stop="2026-03-05 15:00",
    bounds=(-4.5, 39.5, -3.0, 41.0) 
)

print(t_madrid)