from dotenv import load_dotenv
from traffic.data import opensky
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load credentials from .env in the root directory
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))


opensky_username = os.environ.get("OPENSKY_USERNAME")
opensky_password = os.environ.get("OPENSKY_PASSWORD")

if not opensky_username or not opensky_password:
    logger.error("Credentials not found in .env file!")
    exit(1)

logger.info(f"Testing authentication for user: {opensky_username}")

try:
    # A very small, restricted query
    df = opensky.history(
        start="2026-03-01 10:00",
        stop="2026-03-01 10:05",
        callsign="SWR123", # specific short flight
        return_flight=True
    )
    if df is not None:
        logger.info(f"Success! Retrieved {len(df)} data points for SWR123.")
    else:
        logger.info("Query successful, but no data found for this specific callsign in that 5 min window. Auth works!")
except Exception as e:
    logger.error(f"Authentication or query failed: {e}")
