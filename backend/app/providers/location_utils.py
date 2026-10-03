"""Geographic distance utilities. Never invent distances."""

import math
from typing import Optional


def calculate_distance_km(
    lat1: Optional[float],
    lon1: Optional[float],
    lat2: Optional[float],
    lon2: Optional[float],
) -> Optional[float]:
    """Calculate geographic distance in kilometers using Haversine formula.

    Returns None if any coordinate is missing, null, or invalid.
    Strictly adheres to the rule: NEVER invent distances.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None

    try:
        f_lat1 = float(lat1)
        f_lon1 = float(lon1)
        f_lat2 = float(lat2)
        f_lon2 = float(lon2)
    except (ValueError, TypeError):
        return None

    # Validate coordinate ranges (-90 to 90 for lat, -180 to 180 for lon)
    if not (-90.0 <= f_lat1 <= 90.0 and -90.0 <= f_lat2 <= 90.0):
        return None
    if not (-180.0 <= f_lon1 <= 180.0 and -180.0 <= f_lon2 <= 180.0):
        return None

    # Radius of Earth in kilometers
    radius = 6371.0

    dlat = math.radians(f_lat2 - f_lat1)
    dlon = math.radians(f_lon2 - f_lon1)

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(f_lat1))
        * math.cos(math.radians(f_lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return round(radius * c, 2)
