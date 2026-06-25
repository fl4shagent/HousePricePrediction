import pandas as pd
import numpy as np
import aiohttp
import asyncio
import nest_asyncio
import os
from geopy.distance import geodesic

nest_asyncio.apply()

RAFFLES_PLACE = (1.2840, 103.8514)


async def _geocode_one(session, sem, block, street, max_retries=3):
    """Geocode a single building via OneMap API with retry."""
    search_val = f"{block} {street}"
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={search_val}&returnGeom=Y&getAddrDetails=Y&pageNum=1"

    for attempt in range(max_retries):
        try:
            async with sem:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 429:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    result = await response.json()
                    top = result["results"][0]
                    return float(top["LATITUDE"]), float(top["LONGITUDE"])
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(2 * (attempt + 1))
        except (KeyError, IndexError):
            return None, None

    return None, None


def geocode_buildings(buildings_df: pd.DataFrame, cache_path: str,
                      concurrency: int = 10, batch_size: int = 500) -> pd.DataFrame:
    """Geocode unique buildings via OneMap API with batching, retry, and checkpointing."""
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        has_coords = cached["latitude"].notna().sum()
        print(f"Loaded {len(cached)} cached buildings ({has_coords} with coords)")
        if has_coords > len(cached) * 0.9:
            merged = buildings_df.merge(cached, on=["block", "street_name"], how="left")
            missing = merged[merged["latitude"].isna()]
            if len(missing) == 0:
                print("All buildings already geocoded")
                return cached
            to_geocode = missing[["block", "street_name"]].drop_duplicates()
            print(f"{len(to_geocode)} buildings still need geocoding")
        else:
            print(f"Cache has too many failures ({has_coords}/{len(cached)}), re-geocoding all")
            cached = pd.DataFrame({"block": pd.Series(dtype="str"), "street_name": pd.Series(dtype="str"),
                                    "latitude": pd.Series(dtype="float"), "longitude": pd.Series(dtype="float")})
            to_geocode = buildings_df[["block", "street_name"]].drop_duplicates()
    else:
        cached = pd.DataFrame({"block": pd.Series(dtype="str"), "street_name": pd.Series(dtype="str"),
                                "latitude": pd.Series(dtype="float"), "longitude": pd.Series(dtype="float")})
        to_geocode = buildings_df[["block", "street_name"]].drop_duplicates()
        print(f"Geocoding {len(to_geocode)} buildings")

    sem = asyncio.Semaphore(concurrency)
    all_results = []
    total = len(to_geocode)

    for batch_start in range(0, total, batch_size):
        batch = to_geocode.iloc[batch_start:batch_start + batch_size]

        async def run_batch(batch_df):
            async with aiohttp.ClientSession() as session:
                tasks = [
                    _geocode_one(session, sem, row["block"], row["street_name"])
                    for _, row in batch_df.iterrows()
                ]
                return await asyncio.gather(*tasks)

        results = asyncio.run(run_batch(batch))
        all_results.extend(results)

        success = sum(1 for r in results if r[0] is not None)
        print(f"  Batch {batch_start}-{min(batch_start + batch_size, total)}: "
              f"{success}/{len(batch)} successful")

        # Checkpoint after each batch
        checkpoint = to_geocode.iloc[:batch_start + batch_size].copy()
        checkpoint["latitude"] = [r[0] for r in all_results]
        checkpoint["longitude"] = [r[1] for r in all_results]
        combined = pd.concat([cached, checkpoint], ignore_index=True).drop_duplicates(
            subset=["block", "street_name"], keep="last"
        )
        combined.to_csv(cache_path, index=False)

    new_data = to_geocode.copy()
    new_data["latitude"] = [r[0] for r in all_results]
    new_data["longitude"] = [r[1] for r in all_results]

    combined = pd.concat([cached, new_data], ignore_index=True).drop_duplicates(
        subset=["block", "street_name"], keep="last"
    )
    combined.to_csv(cache_path, index=False)

    success = combined["latitude"].notna().sum()
    failed = combined["latitude"].isna().sum()
    print(f"\nTotal: {len(combined)} buildings, {success} geocoded, {failed} failed")

    return combined


def geocode_schools(schools_df: pd.DataFrame, cache_path: str, concurrency: int = 10) -> pd.DataFrame:
    """Geocode schools via OneMap API with retry. Falls back to postal_code for failures."""
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        success = cached["lat"].notna().sum() if "lat" in cached.columns else 0
        if success > len(cached) * 0.9:
            print(f"Loaded {len(cached)} cached schools ({success} with coords)")
            return cached
        print(f"Cache has too many failures ({success}/{len(cached)}), re-geocoding")

    sem = asyncio.Semaphore(concurrency)

    async def geocode_one(session, search_val, retries=3):
        url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={search_val}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
        for attempt in range(retries):
            try:
                async with sem:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                        if response.status == 429:
                            await asyncio.sleep(5 * (attempt + 1))
                            continue
                        result = await response.json()
                        top = result["results"][0]
                        return float(top["LATITUDE"]), float(top["LONGITUDE"])
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(2 * (attempt + 1))
            except (KeyError, IndexError):
                return None, None
        return None, None

    async def run(names):
        async with aiohttp.ClientSession() as session:
            tasks = [geocode_one(session, name) for name in names]
            return await asyncio.gather(*tasks)

    # Geocode by school name
    print(f"Geocoding {len(schools_df)} schools by name...")
    results = asyncio.run(run(schools_df["school_name"].tolist()))
    schools_df = schools_df.copy()
    schools_df["lat"] = [r[0] for r in results]
    schools_df["lng"] = [r[1] for r in results]

    success = schools_df["lat"].notna().sum()
    print(f"  By name: {success}/{len(schools_df)} successful")

    # Fallback: geocode by postal_code for failures
    failed_mask = schools_df["lat"].isna()
    if failed_mask.any():
        failed = schools_df[failed_mask]
        print(f"  Retrying {len(failed)} schools by postal_code...")
        fb_results = asyncio.run(run(failed["postal_code"].astype(str).tolist()))
        for i, (idx, _) in enumerate(failed.iterrows()):
            schools_df.at[idx, "lat"] = fb_results[i][0]
            schools_df.at[idx, "lng"] = fb_results[i][1]

        final_success = schools_df["lat"].notna().sum()
        print(f"  After fallback: {final_success}/{len(schools_df)} successful")

    schools_df.to_csv(cache_path, index=False)
    print(f"Saved {len(schools_df)} schools to {cache_path}")
    return schools_df


def compute_nearest_mrt(building_lat, building_lng, mrt_df, transaction_date):
    """Find nearest MRT station that was open at the transaction date."""
    open_stations = mrt_df[mrt_df["opening_date"] <= transaction_date]
    if len(open_stations) == 0:
        return np.nan, None

    min_dist = float("inf")
    closest = None
    for _, stn in open_stations.iterrows():
        dist = geodesic((building_lat, building_lng), (stn["lat"], stn["lng"])).km
        if dist < min_dist:
            min_dist = dist
            closest = stn["station_name"]

    return min_dist, closest


def compute_nearest_from_locations(building_lat, building_lng, locations_df, lat_col="lat", lng_col="lng"):
    """Find nearest location from a dataframe of lat/lng points."""
    min_dist = float("inf")
    closest_idx = None
    for idx, row in locations_df.iterrows():
        if pd.isna(row[lat_col]) or pd.isna(row[lng_col]):
            continue
        dist = geodesic((building_lat, building_lng), (row[lat_col], row[lng_col])).km
        if dist < min_dist:
            min_dist = dist
            closest_idx = idx

    return min_dist, closest_idx


def compute_cbd_distance(lat, lng):
    """Haversine distance from a point to Raffles Place (CBD)."""
    if pd.isna(lat) or pd.isna(lng):
        return np.nan
    return geodesic((lat, lng), RAFFLES_PLACE).km


def compute_nearest_school_by_level(building_lat, building_lng, schools_df):
    """Find closest school for each education level."""
    levels = {
        "PRIMARY": None,
        "SECONDARY": None,
        "MIXED": None,
    }
    dists = {
        "PRIMARY": float("inf"),
        "SECONDARY": float("inf"),
        "MIXED": float("inf"),
    }

    for _, school in schools_df.iterrows():
        if pd.isna(school["lat"]) or pd.isna(school["lng"]):
            continue
        dist = geodesic((building_lat, building_lng), (school["lat"], school["lng"])).km

        level = str(school["mainlevel_code"]).upper()
        if "PRIMARY" in level and "MIXED" not in level:
            key = "PRIMARY"
        elif "SECONDARY" in level and "MIXED" not in level:
            key = "SECONDARY"
        elif "MIXED" in level:
            key = "MIXED"
        else:
            continue

        if dist < dists[key]:
            dists[key] = dist
            levels[key] = school["school_name"]

    return levels["PRIMARY"], levels["SECONDARY"], levels["MIXED"]


def classify_elite_school(school_name, schools_df):
    """Check if a school is elite (SAP, Autonomous, Gifted, or IP)."""
    if school_name is None:
        return False
    row = schools_df[schools_df["school_name"] == school_name]
    if len(row) == 0:
        return False
    row = row.iloc[0]
    return any(str(row.get(col, "No")).strip().upper() == "YES"
               for col in ["sap_ind", "autonomous_ind", "gifted_ind", "ip_ind"])
