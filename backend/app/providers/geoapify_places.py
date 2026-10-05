"""Geoapify Places API client for nearby hospital and pharmacy discovery.







Connects strictly to the official Geoapify Places API v2 (/v2/places).



Adheres strictly to the MediSense AI core principle:



NEVER fabricate healthcare records, coordinates, distances, contacts, or emergency status.



Only real places returned by Geoapify / OpenStreetMap are presented.



"""







import logging



import re



from typing import Any, Optional



import httpx







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

        # Overpass is a separate public service from Geoapify and may need
        # a longer read timeout than the Geoapify request.
        self.osm_pharmacy = OSMPharmacyProvider(
            timeout_seconds=max(self.timeout_seconds, 30)
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



        name = properties.get("name") or properties.get("formatted") or "Medical Shop / Pharmacy"







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







        try:



            headers = {



                "Accept": "application/json",



                "User-Agent": "MediSenseAI-GeoapifyPlacesClient/1.0",



            }



            with httpx.Client(timeout=float(self.timeout_seconds)) as client:



                response = client.get(self.base_url, params=params, headers=headers)







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
    ) -> dict[str, Any]:
        """
        Search nearby medical shops/pharmacies from:

        1. Geoapify Places API
        2. Direct OpenStreetMap / Overpass

        Rules:
        - Uses the user's confirmed current latitude/longitude.
        - Searches within 25 km first.
        - Expands to 50 km ONLY when the combined 25 km result
          contains zero valid records.
        - Maximum 2 application pages.
        - Maximum 20 visible results.
        - Sorts final merged results by Haversine distance.
        - Never fabricates coordinates, names, addresses, phones, or distances.
        """

        # ------------------------------------------------------------
        # 1. HARD BLOCK: maximum 2 pages
        # ------------------------------------------------------------

        if page > 2:
            return {
                "status": "success",
                "message": "Maximum of 2 pages reached.",
                "data": [],
                "medical_shops": [],
                "page": page,
                "page_size": page_size,
                "has_next": False,
                "has_previous": True,
                "expanded": False,
                "expansion_message": None,
                "disclaimer": (
                    "Medical shop listings do not include medication "
                    "prescribing or dosage advice."
                ),
                "source_metadata": {
                    "source": "Geoapify + OpenStreetMap / Overpass",
                    "configured": True,
                    "source_urls": [
                        "https://api.geoapify.com/v2/places",
                        "https://overpass-api.de/api/interpreter",
                    ],
                },
            }

        # ------------------------------------------------------------
        # 2. SEARCH RADIUS
        # ------------------------------------------------------------

        default_radius = radius_km or self.default_radius_km
        expanded_radius = self.expanded_radius_km

        # Maximum visible results = 20.
        # Keep 50 Geoapify candidates before validation/deduplication.
        candidate_limit = 50

        # ------------------------------------------------------------
        # 3. GEOAPIFY SEARCH
        # ------------------------------------------------------------

        geo_items: list[dict[str, Any]] = []
        geo_features_count = 0
        geo_error: Optional[str] = None

        geo_res = self._query_places(
            category=self.medical_shop_category,
            latitude=latitude,
            longitude=longitude,
            radius_km=default_radius,
            limit=candidate_limit,
            offset=0,
        )

        if geo_res["status"] == "success":
            features = geo_res.get("features", [])
            geo_features_count = len(features)

            logger.info(
                "Medical Shop Geoapify response: "
                "user_lat=%s, user_lon=%s, "
                "radius=%skm, raw_features=%d",
                latitude,
                longitude,
                default_radius,
                len(features),
            )

            deduped_features = self.deduplicate_features(features)

            normalized_geo = [
                self.normalize_feature_to_medical_shop(
                    feature,
                    latitude,
                    longitude,
                )
                for feature in deduped_features
            ]

            geo_items = [
                item
                for item in normalized_geo
                if item is not None
            ]

            geo_items = self.deduplicate_medical_shop_items(
                geo_items
            )

        else:
            geo_error = str(
                geo_res.get("message") or "Geoapify search failed."
            )

            logger.warning(
                "Geoapify medical shop search failed: %s",
                geo_error,
            )

        # ------------------------------------------------------------
        # 4. OPENSTREETMAP / OVERPASS SEARCH
        # ------------------------------------------------------------

        osm_items: list[dict[str, Any]] = []
        osm_error: Optional[str] = None
        osm_raw_count = 0

        osm_res = self.osm_pharmacy.search_pharmacies(
            latitude=latitude,
            longitude=longitude,
            radius_km=default_radius,
        )

        if osm_res["status"] == "success":
            osm_items = [
                item
                for item in (osm_res.get("data") or [])
                if isinstance(item, dict)
            ]
            osm_raw_count = int(osm_res.get("raw_count") or len(osm_items))

            logger.info(
                "Medical Shop OSM response: radius=%skm, raw_elements=%d, valid_items=%d",
                default_radius,
                osm_raw_count,
                len(osm_items),
            )

        else:
            osm_error = str(
                osm_res.get("message") or "OpenStreetMap search failed."
            )

            logger.warning(
                "OpenStreetMap medical shop search failed: %s",
                osm_error,
            )

        # ------------------------------------------------------------
        # 5. MERGE GEOAPIFY + OSM
        # ------------------------------------------------------------

        merged_items = [
            *geo_items,
            *osm_items,
        ]

        merged_items = self.deduplicate_medical_shop_items(
            merged_items
        )

        merged_items.sort(
            key=lambda item: (
                item["distance_km"]
                if item.get("distance_km") is not None
                else float("inf")
            )
        )

        valid_items = merged_items

        for index, item in enumerate(valid_items[:20], start=1):
            logger.info(
                "Medical Shop merged sorted #%d: "
                "name=%s distance=%s source=%s",
                index,
                item.get("name"),
                (
                    f"{item['distance_km']:.3f}km"
                    if item.get("distance_km") is not None
                    else "None"
                ),
                item.get("source"),
            )

        # ------------------------------------------------------------
        # 6. 25 KM -> 50 KM EXPANSION
        # ------------------------------------------------------------

        expanded = False
        expansion_message: Optional[str] = None
        tried_expansion = False

        if (
            not valid_items
            and page == 1
            and (
                radius_km is None
                or radius_km == default_radius
            )
        ):
            tried_expansion = True

            logger.info(
                "No valid medical shops within %skm. "
                "Starting one-time %skm expansion.",
                default_radius,
                expanded_radius,
            )

            # ------------------------
            # Geoapify 50 km
            # ------------------------

            exp_geo_items: list[dict[str, Any]] = []

            exp_geo_res = self._query_places(
                category=self.medical_shop_category,
                latitude=latitude,
                longitude=longitude,
                radius_km=expanded_radius,
                limit=candidate_limit,
                offset=0,
            )

            if exp_geo_res["status"] == "success":
                exp_features = exp_geo_res.get("features") or []

                exp_deduped = self.deduplicate_features(
                    exp_features
                )

                exp_normalized = [
                    self.normalize_feature_to_medical_shop(
                        feature,
                        latitude,
                        longitude,
                    )
                    for feature in exp_deduped
                ]

                exp_geo_items = [
                    item
                    for item in exp_normalized
                    if item is not None
                ]

                exp_geo_items = self.deduplicate_medical_shop_items(
                    exp_geo_items
                )

            # ------------------------
            # OpenStreetMap 50 km
            # ------------------------

            exp_osm_items: list[dict[str, Any]] = []

            exp_osm_res = self.osm_pharmacy.search_pharmacies(
                latitude=latitude,
                longitude=longitude,
                radius_km=expanded_radius,
            )

            if exp_osm_res["status"] == "success":
                exp_osm_items = [
                    item
                    for item in (exp_osm_res.get("data") or [])
                    if isinstance(item, dict)
                ]

            # ------------------------
            # Merge expanded results
            # ------------------------

            expanded_items = [
                *exp_geo_items,
                *exp_osm_items,
            ]

            expanded_items = self.deduplicate_medical_shop_items(
                expanded_items
            )

            expanded_items.sort(
                key=lambda item: (
                    item["distance_km"]
                    if item.get("distance_km") is not None
                    else float("inf")
                )
            )

            if expanded_items:
                valid_items = expanded_items
                expanded = True

                expansion_message = (
                    f"No matching pharmacies/medical shops were found "
                    f"within {int(default_radius)} km. Search expanded "
                    f"to approximately {int(expanded_radius)} km."
                )

        # ------------------------------------------------------------
        # 7. MAXIMUM 20 RESULT RECORDS
        # ------------------------------------------------------------

        valid_items = valid_items[:20]

        # ------------------------------------------------------------
        # 8. DISTANCE SUMMARY
        # ------------------------------------------------------------

        nearest_dist = (
            valid_items[0]["distance_km"]
            if valid_items
            else None
        )

        farthest_dist = (
            valid_items[-1]["distance_km"]
            if valid_items
            else None
        )

        valid_coord_count = sum(
            1
            for item in valid_items
            if item.get("distance_km") is not None
        )

        logger.info(
            "Medical Shops merged search: "
            "user_lat=%s, user_lon=%s, radius=%skm, "
            "geo_candidates=%d, osm_candidates=%d, "
            "final_valid=%d, nearest=%s, farthest=%s, page=%d",
            latitude,
            longitude,
            expanded_radius if expanded else default_radius,
            geo_features_count,
            len(osm_items),
            valid_coord_count,
            (
                f"{nearest_dist:.3f}km"
                if nearest_dist is not None
                else "None"
            ),
            (
                f"{farthest_dist:.3f}km"
                if farthest_dist is not None
                else "None"
            ),
            page,
        )

        # ------------------------------------------------------------
        # 9. MESSAGE
        # ------------------------------------------------------------

        if valid_items:
            msg = (
                f"Found {len(valid_items)} matching medical shop(s)."
            )
        else:
            if tried_expansion:
                msg = (
                    f"No matching medical shops were found within "
                    f"approximately {int(expanded_radius)} km "
                    f"of your confirmed location."
                )
            elif radius_km is not None and radius_km >= expanded_radius:
                msg = (
                    f"No matching medical shops were found within "
                    f"{int(expanded_radius)} km of your confirmed location."
                )
            else:
                msg = (
                    f"No matching medical shops were found within "
                    f"{int(default_radius)} km of your confirmed location."
                )

        # ------------------------------------------------------------
        # 10. PAGINATION
        # ------------------------------------------------------------

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        paged_items = valid_items[start_idx:end_idx]

        has_next = (
            page == 1
            and len(valid_items) > end_idx
        )

        has_previous = page > 1

        # ------------------------------------------------------------
        # 11. RETURN
        # ------------------------------------------------------------

        return {
            "status": "success",
            "data": paged_items,
            "medical_shops": paged_items,
            "total_count": len(valid_items),
            "page": page,
            "page_size": page_size,
            "has_next": has_next,
            "has_previous": has_previous,
            "expanded": expanded,
            "expansion_message": expansion_message,
            "message": msg,
            "disclaimer": (
                "Medical shop listings are location-discovery results only. "
                "They do not provide medication prescribing, dosage, or "
                "treatment advice."
            ),
            "source_metadata": {
                "source": "Geoapify + OpenStreetMap / Overpass",
                "configured": True,
                "source_urls": [
                    "https://api.geoapify.com/v2/places",
                    "https://overpass-api.de/api/interpreter",
                ],
                "geoapify_error": geo_error,
                "osm_error": osm_error,
                "geoapify_candidate_count": geo_features_count,
                "osm_candidate_count": osm_raw_count,
            },
        }
