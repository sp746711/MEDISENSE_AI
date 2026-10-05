"""Geoapify Places API client for nearby hospital and pharmacy discovery.







Connects strictly to the official Geoapify Places API v2 (/v2/places).



Adheres strictly to the MediSense AI core principle:



NEVER fabricate healthcare records, coordinates, distances, contacts, or emergency status.



Only real places returned by Geoapify / OpenStreetMap are presented.



"""







import logging



import re



from contextlib import nullcontext
from typing import Any, Optional
import socket
import uuid

import httpx

# Prioritize IPv4 (AF_INET) over IPv6 (AF_INET6) to eliminate 21s TCP SYN timeout on Windows
_orig_getaddrinfo = socket.getaddrinfo

def _ipv4_preferred_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    res = _orig_getaddrinfo(host, port, family, type, proto, flags)
    return sorted(res, key=lambda x: 0 if x[0] == socket.AF_INET else 1)

if socket.getaddrinfo is not _ipv4_preferred_getaddrinfo:
    socket.getaddrinfo = _ipv4_preferred_getaddrinfo







from app.core.config import get_settings



from app.providers.location_utils import calculate_distance_km
from app.providers.osm_pharmacy import OSMPharmacyProvider







logger = logging.getLogger(__name__)











class RedactApiKeyFilter(logging.Filter):



    """Logging filter that ensures apiKey query parameter values are always redacted in logs."""







    def filter(self, record: logging.LogRecord) -> bool:



        if isinstance(record.msg, str):



            record.msg = re.sub(r"""apiKey=[^&\s\\'"]+""", "apiKey=[REDACTED]", record.msg)



        if record.args:



            new_args = []



            for arg in record.args:



                if isinstance(arg, str):



                    new_args.append(re.sub(r"""apiKey=[^&\s\\'"]+""", "apiKey=[REDACTED]", arg))



                else:



                    new_args.append(arg)



            record.args = tuple(new_args)



        return True











# Ensure httpx logger suppresses raw URL info logs containing query parameters



_httpx_logger = logging.getLogger("httpx")



_httpx_logger.setLevel(logging.WARNING)



_httpx_logger.addFilter(RedactApiKeyFilter())



logger.addFilter(RedactApiKeyFilter())



logging.getLogger().addFilter(RedactApiKeyFilter())











class GeoapifyPlacesClient:



    """Client adapter for Geoapify Places API v2."""







    def __init__(



        self,



        api_key: Optional[str] = None,



        base_url: Optional[str] = None,



        timeout_seconds: Optional[int] = None,



        medical_shop_category: Optional[str] = None,



    ) -> None:



        settings = get_settings()



        self.api_key = api_key if api_key is not None else settings.geoapify_api_key



        self.base_url = (base_url or settings.geoapify_base_url).rstrip("/")



        self.facility_category = settings.geoapify_facility_category



        self.medical_shop_category = (



            medical_shop_category



            if medical_shop_category is not None



            else settings.geoapify_medical_shop_category



        )



        self.default_radius_km = settings.geoapify_default_radius_km



        self.expanded_radius_km = settings.geoapify_expanded_radius_km



        self.page_size = settings.geoapify_page_size



        self.timeout_seconds = timeout_seconds or settings.geoapify_timeout_seconds

        self.osm_pharmacy = OSMPharmacyProvider(
            timeout_seconds=self.timeout_seconds
        )







    def is_configured(self) -> bool:



        """True only if an API key and base URL are configured."""



        return bool(



            self.api_key



            and self.api_key.strip()



            and self.base_url



            and self.base_url.strip()



        )







    def normalize_feature_to_facility(



        self,



        feature: dict[str, Any],



        user_lat: float,



        user_lon: float,



    ) -> Optional[dict[str, Any]]:



        """Normalize Geoapify place feature to canonical facility dictionary.







        Returns None if coordinates are missing or invalid.



        Never fabricates phone, coordinates, last_verified, or emergency status.



        """



        properties = feature.get("properties") or {}



        geometry = feature.get("geometry") or {}



        coords = geometry.get("coordinates") or []







        # Extract latitude and longitude



        lat = properties.get("lat")



        lon = properties.get("lon")



        if lat is None or lon is None:



            if isinstance(coords, (list, tuple)) and len(coords) >= 2:



                lon = coords[0]



                lat = coords[1]







        if lat is None or lon is None:



            return None







        try:



            f_lat = float(lat)



            f_lon = float(lon)



        except (ValueError, TypeError):



            return None







        # Validate coordinate ranges (-90 to 90 for lat, -180 to 180 for lon)



        if not (-90.0 <= f_lat <= 90.0 and -180.0 <= f_lon <= 180.0):



            return None







        place_id = str(properties.get("place_id") or properties.get("id") or "")



        name = properties.get("name") or properties.get("formatted") or "Healthcare Facility"







        # Address & location components



        address = properties.get("formatted") or properties.get("address_line1")



        city = (



            properties.get("city")



            or properties.get("suburb")



            or properties.get("town")



            or properties.get("village")



        )



        state = properties.get("state")



        district = (



            properties.get("district")



            or properties.get("county")



            or properties.get("state_district")



        )







        # Contact information: only if supplied by source



        datasource = properties.get("datasource") or {}



        raw = datasource.get("raw") if isinstance(datasource, dict) else {}



        if not isinstance(raw, dict):



            raw = {}







        contact = None



        if raw.get("phone"):



            contact = str(raw["phone"]).strip()



        elif properties.get("phone"):



            contact = str(properties["phone"]).strip()



        elif isinstance(properties.get("contact"), dict) and properties["contact"].get("phone"):



            contact = str(properties["contact"]["phone"]).strip()







        # Emergency capability: NEVER assume emergency availability for a hospital.



        # Only set if raw OSM data explicitly specifies "yes"



        raw_emergency = str(raw.get("emergency", "")).lower().strip()



        emergency_available = "yes" if raw_emergency == "yes" else None







        # Source URL: only if supplied by source



        source_url = None



        if raw.get("website"):



            source_url = str(raw["website"]).strip()



        elif properties.get("website"):



            source_url = str(properties["website"]).strip()



        elif datasource.get("url"):



            source_url = str(datasource["url"]).strip()







        # Distance calculation via straight-line Haversine



        dist_km = calculate_distance_km(user_lat, user_lon, f_lat, f_lon)







        return {



            "facility_id": place_id,



            "external_id": place_id,



            "name": name,



            "type": "Hospital",



            "state": state,



            "district": district,



            "city": city,



            "address": address,



            "latitude": f_lat,



            "longitude": f_lon,



            "contact": contact,



            "emergency_available": emergency_available,



            "has_emergency": bool(emergency_available == "yes"),



            "source": "Geoapify / OpenStreetMap",



            "source_url": source_url,



            "last_verified": None,  # NEVER set to current time



            "distance_km": dist_km,



        }







    def normalize_feature_to_medical_shop(



        self,



        feature: dict[str, Any],



        user_lat: float,



        user_lon: float,



    ) -> Optional[dict[str, Any]]:



        """Normalize Geoapify place feature to canonical medical shop dictionary.







        Returns None if coordinates are missing or invalid.



        Never fabricates phone, address, coordinates, or license numbers.



        """



        properties = feature.get("properties") or {}



        geometry = feature.get("geometry") or {}



        coords = geometry.get("coordinates") or []







        lat = properties.get("lat")



        lon = properties.get("lon")



        if lat is None or lon is None:



            if isinstance(coords, (list, tuple)) and len(coords) >= 2:



                lon = coords[0]



                lat = coords[1]







        f_lat: Optional[float] = None



        f_lon: Optional[float] = None



        dist_km: Optional[float] = None







        if lat is not None and lon is not None:



            try:



                cand_lat = float(lat)



                cand_lon = float(lon)



                if -90.0 <= cand_lat <= 90.0 and -180.0 <= cand_lon <= 180.0:



                    f_lat = cand_lat



                    f_lon = cand_lon



                    if user_lat is not None and user_lon is not None:



                        dist_km = calculate_distance_km(user_lat, user_lon, f_lat, f_lon)



            except (ValueError, TypeError):



                f_lat = None



                f_lon = None



                dist_km = None







        place_id = str(properties.get("place_id") or properties.get("id") or "")
        raw_name = properties.get("name")
        raw_dict = (properties.get("datasource") or {}).get("raw", {}) if isinstance(properties.get("datasource"), dict) else {}
        name = (
            raw_name
            or (raw_dict.get("name") if isinstance(raw_dict, dict) else None)
            or properties.get("brand")
            or (raw_dict.get("brand") if isinstance(raw_dict, dict) else None)
            or properties.get("operator")
            or (raw_dict.get("operator") if isinstance(raw_dict, dict) else None)
            or properties.get("formatted")
            or "Medical Shop"
        )







        address = properties.get("formatted") or properties.get("address_line1")



        city = (



            properties.get("city")



            or properties.get("suburb")



            or properties.get("town")



            or properties.get("village")



        )



        state = properties.get("state")



        district = (



            properties.get("district")



            or properties.get("county")



            or properties.get("state_district")



        )







        datasource = properties.get("datasource") or {}



        raw = datasource.get("raw") if isinstance(datasource, dict) else {}



        if not isinstance(raw, dict):



            raw = {}







        contact = None



        if raw.get("phone"):



            contact = str(raw["phone"]).strip()



        elif properties.get("phone"):



            contact = str(properties["phone"]).strip()



        elif isinstance(properties.get("contact"), dict) and properties["contact"].get("phone"):



            contact = str(properties["contact"]["phone"]).strip()







        source_url = None



        if raw.get("website"):



            source_url = str(raw["website"]).strip()



        elif properties.get("website"):



            source_url = str(properties["website"]).strip()



        elif datasource.get("url"):



            source_url = str(datasource["url"]).strip()







        if f_lat is not None and f_lon is not None and user_lat is not None and user_lon is not None:



            dist_km = calculate_distance_km(user_lat, user_lon, f_lat, f_lon)



        else:



            dist_km = None







        return {



            "shop_id": place_id,



            "external_id": place_id,



            "name": name,



            "location": address or city or None,



            "state": state,



            "district": district,



            "city": city,



            "address": address,



            "latitude": f_lat,



            "longitude": f_lon,



            "contact": contact,



            "source": "Geoapify / OpenStreetMap",



            "source_url": source_url,



            "last_verified": None,



            "distance_km": dist_km,



        }







    @staticmethod



    def deduplicate_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:



        """Deduplicate Geoapify place features preserving first occurrence.







        Deduplication priority:



        1. primary: place_id



        2. fallback: external ID (id)



        3. fallback if necessary: normalized coordinates + normalized name



        """



        seen_ids: set[str] = set()



        seen_coords_name: set[tuple[str, float, float]] = set()



        unique_features: list[dict[str, Any]] = []







        for feat in features:



            props = feat.get("properties") or {}



            place_id = str(props.get("place_id") or "").strip()



            external_id = str(props.get("id") or "").strip()



            target_id = place_id or external_id



            name = str(props.get("name") or props.get("formatted") or "").strip().lower()







            lat = props.get("lat")



            lon = props.get("lon")



            if lat is None or lon is None:



                geom = feat.get("geometry") or {}



                coords = geom.get("coordinates") or []



                if isinstance(coords, (list, tuple)) and len(coords) >= 2:



                    lon, lat = coords[0], coords[1]







            if target_id and target_id in seen_ids:



                continue







            coord_key = None



            if lat is not None and lon is not None and name:



                try:



                    f_lat = float(lat)



                    f_lon = float(lon)



                    if -90.0 <= f_lat <= 90.0 and -180.0 <= f_lon <= 180.0:



                        coord_key = (name, round(f_lat, 4), round(f_lon, 4))



                        if coord_key in seen_coords_name:



                            continue



                except (ValueError, TypeError):



                    coord_key = None







            if target_id:



                seen_ids.add(target_id)



            if coord_key:



                seen_coords_name.add(coord_key)







            unique_features.append(feat)







        return unique_features

    @staticmethod
    def _is_pharmacy_candidate(feature: dict[str, Any]) -> tuple[bool, str]:
        """Validate whether a discovered place feature is legitimately a medical shop, chemist, or pharmacy.

        Returns (is_valid, reason).
        """
        props = feature.get("properties") or {}

        # 1. Coordinates check
        lat = props.get("lat")
        lon = props.get("lon")
        if lat is None or lon is None:
            geom = feature.get("geometry") or {}
            coords = geom.get("coordinates") or []
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                lon, lat = coords[0], coords[1]

        if lat is None or lon is None:
            return False, "missing_coordinates"
        try:
            f_lat = float(lat)
            f_lon = float(lon)
            if not (-90.0 <= f_lat <= 90.0 and -180.0 <= f_lon <= 180.0):
                return False, "coordinates_out_of_range"
        except (ValueError, TypeError):
            return False, "invalid_coordinate_types"

        name = str(props.get("name") or "").strip().lower()
        formatted = str(props.get("formatted") or "").strip().lower()
        cats = [str(c).lower() for c in (props.get("categories") or [])]

        datasource = props.get("datasource") or {}
        raw = datasource.get("raw") if isinstance(datasource, dict) else {}
        if not isinstance(raw, dict):
            raw = {}

        raw_amenity = str(raw.get("amenity") or "").strip().lower()
        raw_shop = str(raw.get("shop") or "").strip().lower()
        raw_healthcare = str(raw.get("healthcare") or "").strip().lower()
        raw_dispensing = str(raw.get("dispensing") or "").strip().lower()

        # 1. Direct pharmacy classification from provider categories or OSM raw tags
        for c in cats:
            if any(term in c for term in ["pharmacy", "chemist", "drugstore", "druggist"]):
                return True, "matched_provider_category"

        if raw_amenity == "pharmacy" or raw_healthcare == "pharmacy":
            return True, "matched_osm_tag_pharmacy"
        if raw_shop in {"chemist", "pharmacy", "drugstore", "medical_supply"}:
            return True, "matched_osm_tag_shop"
        if raw_dispensing == "yes":
            return True, "matched_osm_dispensing"

        # 2. Text evidence in name, brand, operator, or formatted string
        brand = str(raw.get("brand") or "").strip().lower()
        operator = str(raw.get("operator") or "").strip().lower()
        text_corpus = f"{name} {brand} {operator} {formatted}"

        facility_exclusions = (
            "nursing home",
            "hospital",
            "medical college",
            "medical university",
            "centre for sight",
            "eye hospital",
            "diagnostic center",
            "pathology",
            "dental",
            "optical",
            "clinic",
        )
        has_exclusion = any(ex in text_corpus for ex in facility_exclusions)

        explicit_medicine_terms = (
            "pharmacy",
            "chemist",
            "medical store",
            "medical shop",
            "medicine shop",
            "medicine store",
            "drug store",
            "drugstore",
            "medical hall",
            "dispensary",
            "homeo hall",
        )
        if has_exclusion and not any(term in text_corpus for term in explicit_medicine_terms):
            return False, "non_pharmacy_healthcare_facility"

        pharmacy_keywords = (
            "pharmacy",
            "chemist",
            "medical store",
            "medical shop",
            "medical hall",
            "medicine shop",
            "medicine store",
            "medicines",
            "medicine",
            "medicals",
            "medical",
            "drug store",
            "drugstore",
            "druggist",
            "apothecary",
            "dispensary",
            "pharma",
            "homeo",
            "homeopathy",
            "homeopathic",
            "homeo hall",
            "ayurvedic",
            "ayurveda",
            "dawa",
            "dawakhana",
            "health store",
            "healthcare shop",
        )

        has_pharmacy_keyword = any(kw in text_corpus for kw in pharmacy_keywords)
        if not has_pharmacy_keyword:
            return False, "no_pharmacy_or_medicine_evidence"

        return True, "matched_medicine_keyword"

    @staticmethod
    def deduplicate_medical_shop_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:



        """Deduplicate normalized medical shop dictionaries preserving first occurrence.







        Deduplication priority:



        1. primary: place_id (shop_id)



        2. fallback: external ID (external_id)



        3. fallback if necessary: normalized coordinates + normalized name



        """



        seen_ids: set[str] = set()



        seen_coords_name: set[tuple[str, float, float]] = set()



        unique_items: list[dict[str, Any]] = []







        for item in items:



            shop_id = str(item.get("shop_id") or "").strip()



            external_id = str(item.get("external_id") or "").strip()



            target_id = shop_id or external_id



            name = str(item.get("name") or "").strip().lower()



            lat = item.get("latitude")



            lon = item.get("longitude")







            if target_id and target_id in seen_ids:



                continue







            coord_key = None



            if lat is not None and lon is not None and name:



                try:



                    f_lat = float(lat)



                    f_lon = float(lon)



                    coord_key = (name, round(f_lat, 4), round(f_lon, 4))



                    if coord_key in seen_coords_name:



                        continue



                except (ValueError, TypeError):



                    coord_key = None







            if target_id:



                seen_ids.add(target_id)



            if coord_key:



                seen_coords_name.add(coord_key)







            unique_items.append(item)







        return unique_items







    def _query_places(



        self,



        category: str,



        latitude: float,



        longitude: float,



        radius_km: float,



        page: int = 1,



        page_size: int = 10,



        limit: Optional[int] = None,



        offset: Optional[int] = None,
        name: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> dict[str, Any]:



        """Perform a single bounded HTTP GET request to Geoapify Places API v2.







        Note: Geoapify parameters use longitude,latitude order inside proximity and circle expressions.



        """



        if page > 2:



            return {



                "status": "success",



                "message": "Maximum of 2 pages reached.",



                "features": [],



                "source_metadata": {



                    "source": "Geoapify / OpenStreetMap",



                    "configured": True,



                    "source_url": "https://api.geoapify.com/v2/places",



                },



            }







        if not self.is_configured():



            return {



                "status": "configuration_missing",



                "message": "Nearby healthcare search is not configured.",



                "features": [],



                "source_metadata": {



                    "source": "Geoapify / OpenStreetMap",



                    "configured": False,



                    "source_url": "https://api.geoapify.com/v2/places",



                },



            }







        radius_meters = int(radius_km * 1000)



        query_limit = limit if limit is not None else page_size



        query_offset = offset if offset is not None else (page - 1) * page_size



        params = {



            "categories": category,



            "filter": f"circle:{longitude},{latitude},{radius_meters}",



            "bias": f"proximity:{longitude},{latitude}",



            "limit": query_limit,



            "offset": query_offset,



            "apiKey": self.api_key,
        }
        if name and name.strip():
            params["name"] = name.strip()







        try:



            headers = {



                "Accept": "application/json",



                "User-Agent": "MediSenseAI-GeoapifyPlacesClient/1.0",



            }



            cm = nullcontext(client) if client is not None else httpx.Client(timeout=httpx.Timeout(connect=2.5, read=float(self.timeout_seconds), write=2.5, pool=2.5))
            with cm as active_client:
                response = active_client.get(self.base_url, params=params, headers=headers)







                if response.status_code == 429:



                    return {



                        "status": "rate_limited",



                        "message": "Nearby healthcare data source rate limit reached. Please try again shortly.",



                        "features": [],



                        "source_metadata": {



                            "source": "Geoapify / OpenStreetMap",



                            "configured": True,



                            "source_url": "https://api.geoapify.com/v2/places",



                        },



                    }







                if response.status_code in (401, 403):



                    return {



                        "status": "authorization_error",



                        "message": "Nearby healthcare data source authorization failed.",



                        "features": [],



                        "source_metadata": {



                            "source": "Geoapify / OpenStreetMap",



                            "configured": True,



                            "source_url": "https://api.geoapify.com/v2/places",



                        },



                    }







                if response.status_code >= 500:



                    return {



                        "status": "source_error",



                        "message": "Nearby healthcare data source is temporarily unavailable.",



                        "features": [],



                        "source_metadata": {



                            "source": "Geoapify / OpenStreetMap",



                            "configured": True,



                            "source_url": "https://api.geoapify.com/v2/places",



                        },



                    }







                if response.status_code != 200:



                    return {



                        "status": "source_error",



                        "message": f"External healthcare place source returned an error (HTTP {response.status_code}).",



                        "features": [],



                        "source_metadata": {



                            "source": "Geoapify / OpenStreetMap",



                            "configured": True,



                            "source_url": "https://api.geoapify.com/v2/places",



                        },



                    }







                try:



                    payload = response.json()



                except Exception:



                    return {



                        "status": "source_error",



                        "message": "Malformed response from healthcare data source.",



                        "features": [],



                        "source_metadata": {



                            "source": "Geoapify / OpenStreetMap",



                            "configured": True,



                            "source_url": "https://api.geoapify.com/v2/places",



                        },



                    }







                if not isinstance(payload, dict):



                    return {



                        "status": "source_error",



                        "message": "Malformed response from healthcare data source.",



                        "features": [],



                        "source_metadata": {



                            "source": "Geoapify / OpenStreetMap",



                            "configured": True,



                            "source_url": "https://api.geoapify.com/v2/places",



                        },



                    }







                features = payload.get("features", [])



                return {



                    "status": "success",



                    "features": features if isinstance(features, list) else [],



                    "source_metadata": {



                        "source": "Geoapify / OpenStreetMap",



                        "configured": True,



                        "source_url": "https://api.geoapify.com/v2/places",



                    },



                }







        except httpx.TimeoutException:



            logger.warning("Geoapify Places API request timed out after %ds", self.timeout_seconds)



            return {



                "status": "unavailable",



                "message": "Nearby healthcare data source timed out.",



                "features": [],



                "source_metadata": {



                    "source": "Geoapify / OpenStreetMap",



                    "configured": True,



                    "source_url": "https://api.geoapify.com/v2/places",



                },



            }



        except httpx.RequestError as exc:



            logger.warning("Geoapify Places API request network error: %s", exc)



            return {



                "status": "unavailable",



                "message": "Nearby healthcare data source is currently unreachable.",



                "features": [],



                "source_metadata": {



                    "source": "Geoapify / OpenStreetMap",



                    "configured": True,



                    "source_url": "https://api.geoapify.com/v2/places",



                },



            }



        except Exception as exc:



            logger.error("Unexpected error in GeoapifyPlacesClient: %s", exc)



            return {



                "status": "unavailable",



                "message": "Nearby healthcare data source is currently unreachable.",



                "features": [],



                "source_metadata": {



                    "source": "Geoapify / OpenStreetMap",



                    "configured": True,



                    "source_url": "https://api.geoapify.com/v2/places",



                },



            }







    def search_places(



        self,



        category: str,



        latitude: float,



        longitude: float,



        radius_km: Optional[float] = None,



        page: int = 1,



        page_size: int = 10,



    ) -> dict[str, Any]:



        """Generic places query with 25km -> 50km radius expansion and max 2 pages enforcement."""



        if page > 2:



            return {



                "status": "success",



                "features": [],



                "page": page,



                "page_size": page_size,



                "has_next": False,



                "has_previous": True,



                "expanded": False,



                "expansion_message": None,



                "message": "Maximum of 2 pages reached.",



                "source_metadata": {



                    "source": "Geoapify / OpenStreetMap",



                    "configured": True,



                    "source_url": "https://api.geoapify.com/v2/places",



                },



            }







        default_radius = radius_km or self.default_radius_km



        expanded_radius = self.expanded_radius_km







        res = self._query_places(



            category=category,



            latitude=latitude,



            longitude=longitude,



            radius_km=default_radius,



            page=page,



            page_size=page_size,



        )







        if res["status"] != "success":



            return res







        features = res.get("features", [])



        expanded = False



        expansion_message = None







        # Expansion ONLY when 0 features found on page 1



        if not features and page == 1 and (radius_km is None or radius_km == default_radius):



            exp_res = self._query_places(



                category=category,



                latitude=latitude,



                longitude=longitude,



                radius_km=expanded_radius,



                page=1,



                page_size=page_size,



            )



            if exp_res["status"] == "success" and exp_res.get("features"):



                features = exp_res["features"]



                expanded = True



                expansion_message = (



                    f"No matching places were found within {int(default_radius)} km. "



                    f"Search expanded to approximately {int(expanded_radius)} km."



                )







        return {



            "status": "success",



            "features": features,



            "page": page,



            "page_size": page_size,



            "has_next": (page < 2) and (len(features) == page_size),



            "has_previous": (page > 1),



            "expanded": expanded,



            "expansion_message": expansion_message,



            "source_metadata": res.get("source_metadata", {}),



        }







    def search_hospitals(



        self,



        latitude: float,



        longitude: float,



        radius_km: Optional[float] = None,



        emergency_only: bool = False,



        page: int = 1,



        page_size: int = 10,



    ) -> dict[str, Any]:



        """Search nearby hospitals using Geoapify Places API with 25km -> 50km radius expansion and max 2 pages."""



        # Block page > 2 immediately without calling Geoapify



        if page > 2:



            return {



                "status": "success",



                "message": "Maximum of 2 pages reached.",



                "data": [],



                "facilities": [],



                "page": page,



                "page_size": page_size,



                "has_next": False,



                "has_previous": True,



                "expanded": False,



                "expansion_message": None,



                "source_metadata": {



                    "source": "Geoapify / OpenStreetMap",



                    "configured": True,



                    "source_url": "https://api.geoapify.com/v2/places",



                },



            }







        default_radius = radius_km or self.default_radius_km



        expanded_radius = self.expanded_radius_km







        # 1. Primary query at default radius (25km)



        res = self._query_places(



            category=self.facility_category,



            latitude=latitude,



            longitude=longitude,



            radius_km=default_radius,



            page=page,



            page_size=page_size,



        )







        if res["status"] != "success":



            return {



                "status": res["status"],



                "message": res["message"],



                "data": [],



                "facilities": [],



                "page": page,



                "page_size": page_size,



                "has_next": False,



                "has_previous": page > 1,



                "expanded": False,



                "expansion_message": None,



                "source_metadata": res.get("source_metadata", {}),



            }







        features = res["features"]



        raw_items = [



            self.normalize_feature_to_facility(feat, latitude, longitude)



            for feat in features



        ]



        valid_items = [item for item in raw_items if item is not None]







        # Proximity sort: nearest first



        valid_items.sort(key=lambda x: x["distance_km"] if x.get("distance_km") is not None else float("inf"))







        expanded = False



        expansion_message = None



        tried_expansion = False







        # 2. Phase 2 expansion: ONLY when 0 results on page 1 and using default radius (25km)



        if not valid_items and page == 1 and (radius_km is None or radius_km == default_radius):



            tried_expansion = True



            exp_res = self._query_places(



                category=self.facility_category,



                latitude=latitude,



                longitude=longitude,



                radius_km=expanded_radius,



                page=1,



                page_size=page_size,



            )



            if exp_res["status"] == "success" and exp_res["features"]:



                exp_items = [



                    self.normalize_feature_to_facility(feat, latitude, longitude)



                    for feat in exp_res["features"]



                ]



                valid_exp_items = [item for item in exp_items if item is not None]



                if valid_exp_items:



                    valid_exp_items.sort(



                        key=lambda x: x["distance_km"] if x.get("distance_km") is not None else float("inf")



                    )



                    valid_items = valid_exp_items



                    expanded = True



                    expansion_message = (



                        f"No matching hospitals were found within {int(default_radius)} km. "



                        f"Search expanded to approximately {int(expanded_radius)} km."



                    )







        # Emergency capability handling



        if emergency_only:



            verified_emergency = [item for item in valid_items if item.get("emergency_available") == "yes"]



            if verified_emergency:



                valid_items = verified_emergency



                msg = f"Found {len(valid_items)} hospital(s) with verified emergency services."



            else:



                if valid_items:



                    msg = (



                        "Nearby hospital locations were found, but emergency-service availability "



                        "is not verified by this geographic place source."



                    )



                else:



                    if tried_expansion or (radius_km is not None and radius_km >= expanded_radius):



                        msg = f"No matching hospitals were found within {int(expanded_radius)} km of your confirmed location."



                    else:



                        msg = f"No matching hospitals were found within {int(default_radius)} km of your confirmed location."



        else:



            if valid_items:



                msg = f"Found {len(valid_items)} matching hospital(s)."



            else:



                if tried_expansion or (radius_km is not None and radius_km >= expanded_radius):



                    msg = f"No matching hospitals were found within {int(expanded_radius)} km of your confirmed location."



                else:



                    msg = f"No matching hospitals were found within {int(default_radius)} km of your confirmed location."







        # Maximum 2 pages: page 2 has_next is ALWAYS False



        has_next = (page < 2) and (len(valid_items) == page_size)



        has_previous = (page > 1)







        return {



            "status": "success",



            "data": valid_items,



            "facilities": valid_items,



            "page": page,



            "page_size": page_size,



            "has_next": has_next,



            "has_previous": has_previous,



            "expanded": expanded,



            "expansion_message": expansion_message,



            "message": msg,



            "source_metadata": {



                "source": "Geoapify / OpenStreetMap",



                "configured": True,



                "source_url": "https://api.geoapify.com/v2/places",



            },



        }







    def search_pharmacies(
        self,
        latitude: float,
        longitude: float,
        radius_km: Optional[float] = None,
        page: int = 1,
        page_size: int = 10,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Search real nearby pharmacies/medical shops using progressive radius expansion.

        Sequence: 1 km -> 3 km -> 5 km -> 10 km -> 25 km -> 50 km maximum.
        Strictly sequential: 1 km completes BEFORE 3 km starts, 3 km completes BEFORE 5 km starts.
        Expands to the next radius ONLY if the current radius returns zero valid records.
        Never expands merely because results are fewer than 10 or 20.
        All candidate distances are calculated using Haversine from confirmed user coordinates.
        Never fabricates pharmacies, coordinates, or distances.
        """
        page = max(int(page), 1)
        page_size = min(max(int(page_size), 1), 10)
        req_id = request_id.strip() if request_id and request_id.strip() else f"req_{uuid.uuid4().hex[:8]}"

        disclaimer = (
            "Medical shop listings are location-discovery results only. "
            "They do not provide medication prescribing, dosage, or treatment advice."
        )

        if page > 2:
            return {
                "status": "success",
                "message": "Maximum of 2 pages reached.",
                "data": [],
                "medical_shops": [],
                "total_count": 0,
                "page": page,
                "page_size": page_size,
                "has_next": False,
                "has_previous": True,
                "expanded": False,
                "expansion_message": None,
                "disclaimer": disclaimer,
                "request_id": req_id,
            }

        try:
            user_lat = float(latitude)
            user_lon = float(longitude)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "message": "Valid latitude and longitude are required.",
                "data": [],
                "medical_shops": [],
                "page": page,
                "page_size": page_size,
                "has_next": False,
                "has_previous": page > 1,
                "request_id": req_id,
            }

        if not (-90.0 <= user_lat <= 90.0 and -180.0 <= user_lon <= 180.0):
            return {
                "status": "error",
                "message": "Latitude or longitude is outside the valid range.",
                "data": [],
                "medical_shops": [],
                "page": page,
                "page_size": page_size,
                "has_next": False,
                "has_previous": page > 1,
                "request_id": req_id,
            }

        # Progressive search radii levels: strictly 1 -> 2 -> 3 -> 5 -> 10 -> 25 -> 50 km
        search_radii: list[float] = [1.0, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0]

        # Approved pharmacy categories
        configured = [
            value.strip()
            for value in re.split(r"[,;|]", str(self.medical_shop_category or ""))
            if value.strip()
        ]
        approved = [
            "healthcare.pharmacy",
            "commercial.health_and_beauty.pharmacy",
            "commercial.chemist",
            "healthcare.clinic_or_praxis",
            "commercial.health_and_beauty",
        ]
        categories: list[str] = []
        for category in [*configured, *approved]:
            if category not in categories:
                categories.append(category)

        candidate_limit = 100
        successful_radius: Optional[float] = None
        accumulated_candidates: list[dict[str, Any]] = []
        geo_stats: list[dict[str, Any]] = []
        geo_errors: list[str] = []
        osm_error: Optional[str] = None
        expansion_reason: Optional[str] = None

        logger.info(
            "MEDICAL_SHOP_REQUEST_START: request_id=%s lat=%s lon=%s initial_radius=%skm candidate_radii=%s",
            req_id,
            user_lat,
            user_lon,
            search_radii[0],
            search_radii,
        )

        # Pooled HTTP client session across all categories and radii for connection keep-alive
        timeout_cfg = httpx.Timeout(connect=2.5, read=float(self.timeout_seconds), write=2.5, pool=2.5)
        with httpx.Client(timeout=timeout_cfg) as session_client:
            for current_radius in search_radii:
                radius_label = int(current_radius) if current_radius.is_integer() else current_radius
                logger.info(
                    "MEDICAL_SHOP_RADIUS_START: request_id=%s radius=%s",
                    req_id,
                    radius_label,
                )

                # 1. Geoapify Primary Category Search
                cat_query_string = ",".join(categories)
                cat_result = self._query_places(
                    category=cat_query_string,
                    latitude=user_lat,
                    longitude=user_lon,
                    radius_km=current_radius,
                    limit=candidate_limit,
                    offset=0,
                    client=session_client,
                )

                geo_cat_raw = 0
                geo_cat_items: list[dict[str, Any]] = []

                if cat_result.get("status") != "success":
                    msg = str(cat_result.get("message") or "Geoapify category query failed.")
                    geo_errors.append(f"category@{current_radius}km: {msg}")
                else:
                    cat_feats = cat_result.get("features") or []
                    geo_cat_raw = len(cat_feats)
                    for feat in self.deduplicate_features(cat_feats):
                        is_cand, reason = self._is_pharmacy_candidate(feat)
                        p = feat.get("properties") or {}
                        p_name = p.get("name") or p.get("formatted") or "(unnamed)"
                        p_lat = p.get("lat")
                        p_lon = p.get("lon")
                        p_cats = p.get("categories")
                        if not is_cand:
                            logger.info(
                                "MEDICAL_SHOP_REJECT: reason=%s provider=Geoapify name=%s coordinates=(%s,%s) category=%s",
                                reason,
                                p_name,
                                p_lat,
                                p_lon,
                                p_cats,
                            )
                            continue
                        item = self.normalize_feature_to_medical_shop(feat, user_lat, user_lon)
                        if item is not None:
                            i_lat = item.get("latitude")
                            i_lon = item.get("longitude")
                            if i_lat is not None and i_lon is not None and -90.0 <= i_lat <= 90.0 and -180.0 <= i_lon <= 180.0:
                                dist = calculate_distance_km(user_lat, user_lon, i_lat, i_lon)
                                if dist is not None and 0.0 <= float(dist) <= current_radius + 0.05:
                                    item["distance_km"] = round(float(dist), 3)
                                    geo_cat_items.append(item)

                geo_cat_items = self.deduplicate_medical_shop_items(geo_cat_items)
                geo_cat_items.sort(key=lambda x: float(x["distance_km"]))
                geo_cat_accepted = len(geo_cat_items)
                geo_cat_rejected = max(0, geo_cat_raw - geo_cat_accepted)
                nearest_cat_str = f"{geo_cat_items[0]['distance_km']:.3f}" if geo_cat_items else "None"

                logger.info(
                    "MEDICAL_SHOP_GEOAPIFY_CATEGORY: request_id=%s radius=%s source=geoapify query=%s raw_count=%d accepted_count=%d rejected_count=%d nearest_distance=%s",
                    req_id,
                    radius_label,
                    cat_query_string,
                    geo_cat_raw,
                    geo_cat_accepted,
                    geo_cat_rejected,
                    nearest_cat_str,
                )

                # 2. Geoapify Secondary Discovery (text / name search under healthcare & commercial)
                geo_sec_raw = 0
                geo_sec_items: list[dict[str, Any]] = []
                secondary_terms = ["pharmacy", "medical", "medicine", "chemist", "drug"]

                for term in secondary_terms:
                    sec_res = self._query_places(
                        category="healthcare,commercial",
                        name=term,
                        latitude=user_lat,
                        longitude=user_lon,
                        radius_km=current_radius,
                        limit=candidate_limit,
                        offset=0,
                        client=session_client,
                    )
                    if sec_res.get("status") == "success":
                        feats = sec_res.get("features") or []
                        geo_sec_raw += len(feats)
                        for feat in feats:
                            is_cand, reason = self._is_pharmacy_candidate(feat)
                            p = feat.get("properties") or {}
                            p_name = p.get("name") or p.get("formatted") or "(unnamed)"
                            p_lat = p.get("lat")
                            p_lon = p.get("lon")
                            p_cats = p.get("categories")
                            if not is_cand:
                                logger.info(
                                    "MEDICAL_SHOP_REJECT: reason=%s provider=Geoapify name=%s coordinates=(%s,%s) category=%s",
                                    reason,
                                    p_name,
                                    p_lat,
                                    p_lon,
                                    p_cats,
                                )
                                continue
                            item = self.normalize_feature_to_medical_shop(feat, user_lat, user_lon)
                            if item is not None:
                                i_lat = item.get("latitude")
                                i_lon = item.get("longitude")
                                if i_lat is not None and i_lon is not None and -90.0 <= i_lat <= 90.0 and -180.0 <= i_lon <= 180.0:
                                    dist = calculate_distance_km(user_lat, user_lon, i_lat, i_lon)
                                    if dist is not None and 0.0 <= float(dist) <= current_radius + 0.05:
                                        item["distance_km"] = round(float(dist), 3)
                                        geo_sec_items.append(item)

                geo_sec_items = self.deduplicate_medical_shop_items(geo_sec_items)
                geo_sec_items.sort(key=lambda x: float(x["distance_km"]))
                geo_sec_accepted = len(geo_sec_items)
                geo_sec_rejected = max(0, geo_sec_raw - geo_sec_accepted)
                nearest_sec_str = f"{geo_sec_items[0]['distance_km']:.3f}" if geo_sec_items else "None"

                logger.info(
                    "MEDICAL_SHOP_GEOAPIFY_TEXT: request_id=%s radius=%s source=geoapify query=%s raw_count=%d accepted_count=%d rejected_count=%d nearest_distance=%s",
                    req_id,
                    radius_label,
                    ",".join(secondary_terms),
                    geo_sec_raw,
                    geo_sec_accepted,
                    geo_sec_rejected,
                    nearest_sec_str,
                )

                # 3. OSM search for current progressive radius sequentially
                osm_res = self.osm_pharmacy.search_pharmacies(
                    latitude=user_lat,
                    longitude=user_lon,
                    radius_km=current_radius,
                    request_id=req_id,
                )
                osm_items = [
                    item for item in (osm_res.get("data") or [])
                    if isinstance(item, dict)
                ] if osm_res.get("status") == "success" else []
                if osm_res.get("status") != "success":
                    osm_error = str(osm_res.get("message") or osm_res.get("error") or "OpenStreetMap search failed.")
                else:
                    osm_error = None

                # 4. Merge candidates across Geoapify categories, Geoapify secondary, and OSM
                all_radius_candidates = [*geo_cat_items, *geo_sec_items, *osm_items]

                logger.info(
                    "MEDICAL_SHOP_MERGED: request_id=%s radius=%s count=%d",
                    req_id,
                    radius_label,
                    len(all_radius_candidates),
                )

                # 5. Distance Filter & Validation
                valid_radius_items: list[dict[str, Any]] = []
                for item in all_radius_candidates:
                    try:
                        c_lat = float(item.get("latitude"))
                        c_lon = float(item.get("longitude"))
                    except (TypeError, ValueError):
                        continue
                    if not (-90.0 <= c_lat <= 90.0 and -180.0 <= c_lon <= 180.0):
                        continue
                    dist = calculate_distance_km(user_lat, user_lon, c_lat, c_lon)
                    if dist is not None and 0.0 <= float(dist) <= current_radius + 0.05:
                        item["distance_km"] = round(float(dist), 3)
                        valid_radius_items.append(item)

                nearest_filter_str = (
                    f"{min(float(i['distance_km']) for i in valid_radius_items):.3f}"
                    if valid_radius_items
                    else "None"
                )

                logger.info(
                    "MEDICAL_SHOP_DISTANCE_FILTER: request_id=%s radius=%s valid=%d nearest=%s",
                    req_id,
                    radius_label,
                    len(valid_radius_items),
                    nearest_filter_str,
                )

                # 6. Deduplication within radius
                dedup_radius_items = self.deduplicate_medical_shop_items(valid_radius_items)
                dedup_radius_items.sort(key=lambda item: float(item["distance_km"]))
                nearest_dedup_str = (
                    f"{dedup_radius_items[0]['distance_km']:.3f}"
                    if dedup_radius_items
                    else "None"
                )

                logger.info(
                    "MEDICAL_SHOP_DEDUP: request_id=%s radius=%s count=%d nearest=%s",
                    req_id,
                    radius_label,
                    len(dedup_radius_items),
                    nearest_dedup_str,
                )

                # 7. Accumulate into running candidates pool without discarding closer results
                accumulated_candidates = self.deduplicate_medical_shop_items(
                    [*accumulated_candidates, *dedup_radius_items]
                )
                accumulated_candidates.sort(key=lambda item: float(item["distance_km"]))
                nearest_accum_str = (
                    f"{accumulated_candidates[0]['distance_km']:.3f}"
                    if accumulated_candidates
                    else "None"
                )

                logger.info(
                    "MEDICAL_SHOP_RADIUS_COMPLETE: request_id=%s radius=%s radius_count=%d accumulated_total=%d nearest=%s",
                    req_id,
                    radius_label,
                    len(dedup_radius_items),
                    len(accumulated_candidates),
                    nearest_accum_str,
                )

                # 8. Check stopping criteria: Coverage Confirmed vs Insufficient Coverage
                coverage_confirmed = False
                if current_radius <= 1.0 and len(accumulated_candidates) >= 1:
                    coverage_confirmed = True
                    expansion_reason = "CONFIRMED_LOCAL_COVERAGE_AT_1KM"
                elif current_radius <= 2.0 and len(accumulated_candidates) >= 2:
                    coverage_confirmed = True
                    expansion_reason = "CONFIRMED_LOCAL_COVERAGE_AT_2KM"
                elif current_radius <= 3.0 and len(accumulated_candidates) >= 3:
                    coverage_confirmed = True
                    expansion_reason = "CONFIRMED_LOCAL_COVERAGE_AT_3KM"
                elif len(accumulated_candidates) >= page_size:
                    coverage_confirmed = True
                    expansion_reason = "CONFIRMED_SUFFICIENT_POOL_REACHED"
                elif current_radius >= search_radii[-1]:
                    coverage_confirmed = True
                    expansion_reason = "MAX_RADIUS_REACHED"

                if coverage_confirmed:
                    successful_radius = current_radius
                    logger.info(
                        "STOP_REASON=%s: request_id=%s radius=%s accumulated=%d nearest=%s",
                        expansion_reason,
                        req_id,
                        radius_label,
                        len(accumulated_candidates),
                        nearest_accum_str,
                    )
                    break
                else:
                    logger.info(
                        "MEDICAL_SHOP_EXPANDING: request_id=%s current_radius=%s accumulated_count=%d target_pool=%d reason=INSUFFICIENT_COVERAGE",
                        req_id,
                        radius_label,
                        len(accumulated_candidates),
                        page_size,
                    )

        if not accumulated_candidates:
            logger.info(
                "STOP_REASON=NO_VALID_RESULTS_AT_MAX_RADIUS: request_id=%s max_radius=%skm",
                req_id,
                search_radii[-1],
            )

        # 9. Result formatting, capping at 20 (max 2 pages), and pagination
        all_items = accumulated_candidates[:20]
        total_count = len(all_items)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paged_items = all_items[start_idx:end_idx]
        has_next = page == 1 and total_count > end_idx

        nearest_val = f"{all_items[0]['distance_km']:.3f}" if all_items else "None"
        final_radius_label = (
            int(successful_radius)
            if (successful_radius and successful_radius.is_integer())
            else (successful_radius or search_radii[-1])
        )

        if not all_items:
            expanded = True
            expansion_message = (
                f"No matching medical shops were found within approximately {int(search_radii[-1])} km."
            )
            message = f"No matching medical shops were found within approximately {int(search_radii[-1])} km."
        elif all_items[0]["distance_km"] <= search_radii[0]:
            expanded = False
            expansion_message = None
            message = f"Found {total_count} matching medical shop(s)."
        else:
            expanded = True
            expansion_message = (
                f"No matching pharmacies/medical shops were found within {int(search_radii[0])} km. "
                f"Search expanded to approximately {final_radius_label} km."
            )
            message = f"Found {total_count} matching medical shop(s)."

        logger.info(
            "MEDICAL_SHOP_FINAL: request_id=%s nearest_candidate_distance=%s candidate_count=%d search_radius_used=%s expansion_reason=%s",
            req_id,
            nearest_val,
            total_count,
            final_radius_label,
            expansion_reason or "COVERAGE_COMPLETE",
        )

        return {
            "status": "success",
            "data": paged_items,
            "medical_shops": paged_items,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "has_next": has_next,
            "has_previous": page > 1,
            "expanded": expanded,
            "expansion_message": expansion_message,
            "message": message,
            "disclaimer": disclaimer,
            "request_id": req_id,
            "source_metadata": {
                "source": "Geoapify + OpenStreetMap / Overpass",
                "configured": True,
                "source_urls": [
                    "https://api.geoapify.com/v2/places",
                    "https://overpass-api.de/api/interpreter",
                ],
                "search_radius_km": successful_radius or search_radii[-1],
                "geoapify_categories": geo_stats,
                "geoapify_errors": geo_errors,
                "osm_error": osm_error,
            },
        }


