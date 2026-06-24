import os
import requests
import pandas as pd
import time

# os is used by fetch_all_resale_transactions


def fetch_datagov_csv(dataset_id: str, save_path: str, max_retries: int = 5) -> pd.DataFrame:
    """Download a dataset from data.gov.sg using the poll-download API."""
    initiate_url = f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download"

    for attempt in range(max_retries):
        try:
            resp = requests.get(initiate_url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != 0:
                raise RuntimeError(f"Failed to initiate download: {data}")

            download_url = data["data"].get("url")
            if not download_url:
                for _ in range(10):
                    time.sleep(3)
                    resp = requests.get(initiate_url, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    download_url = data["data"].get("url")
                    if download_url:
                        break

            if not download_url:
                raise RuntimeError("Timed out waiting for download URL")

            df = pd.read_csv(download_url)
            df.to_csv(save_path, index=False)
            print(f"Saved {len(df)} rows to {save_path}")
            return df

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"Rate limited (429). Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
            else:
                raise


RESALE_DATASET_IDS = {
    "2000_2012": "d_43f493c6c50d54243cc1eab0df142d6a",      # 2000 - Feb 2012
    "2012_2014": "d_2d5ff9ea31397b66239f245f57751537",      # Mar 2012 - Dec 2014
    "2015_2016": "d_ea9ed51da2787afaf8e51f827c304208",      # Jan 2015 - Dec 2016
    "2017_onwards": "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",   # Jan 2017 onwards
}


def fetch_all_resale_transactions(raw_dir: str) -> pd.DataFrame:
    """Fetch all HDB resale transaction datasets and concatenate them.

    Skips datasets that are already downloaded to avoid redundant API calls.
    """
    dfs = []
    for label, dataset_id in RESALE_DATASET_IDS.items():
        save_path = f"{raw_dir}/resale_transactions_{label}.csv"
        if os.path.exists(save_path):
            print(f"\n--- {label}: already exists, loading from disk ---")
            df = pd.read_csv(save_path)
            print(f"Loaded {len(df)} rows from {save_path}")
        else:
            print(f"\n--- Fetching {label} ---")
            df = fetch_datagov_csv(dataset_id, save_path)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Standardize column names across all periods
    df_all.columns = df_all.columns.str.strip().str.lower()

    combined_path = f"{raw_dir}/resale_transactions.csv"
    df_all.to_csv(combined_path, index=False)
    print(f"\nCombined: {len(df_all)} total rows saved to {combined_path}")
    return df_all


def load_mrt_from_lta_shapefile(shapefile_path: str) -> pd.DataFrame:
    """Load MRT/LRT stations from LTA DataMall shapefile with official coordinates.

    Converts SVY21 polygon centroids to WGS84 lat/lng.
    Filters out depots, facility buildings, and sub stations.
    """
    import geopandas as gpd

    gdf = gpd.read_file(shapefile_path)

    exclude_keywords = ["DEPOT", "FACILITY", "SUB STATION", "TUNNEL STRUCTURE"]
    mask = ~gdf["STN_NAM_DE"].str.upper().apply(
        lambda x: any(kw in x for kw in exclude_keywords)
    )
    gdf = gdf[mask].copy()

    gdf["geometry"] = gdf.geometry.centroid
    gdf = gdf.to_crs(epsg=4326)
    gdf["lat"] = gdf.geometry.y
    gdf["lng"] = gdf.geometry.x

    gdf["station_name"] = gdf["STN_NAM_DE"].apply(
        lambda x: x.replace(" MRT STATION", "").replace(" LRT STATION", "").title()
    )
    gdf["type"] = gdf["TYP_CD_DES"]

    gdf = gdf.drop_duplicates(subset=["station_name", "type"], keep="first")

    return gdf[["station_name", "type", "lat", "lng"]].reset_index(drop=True)


# Opening dates from LTA press releases and Wikipedia.
# The LTA shapefile has coordinates but NOT opening dates.
# Unbuilt stations use 2099-12-31 so they are never included in historical calculations.
STATION_OPENING_DATES = {
    "Jurong East": "1990-03-10", "Bukit Batok": "1990-03-10",
    "Bukit Gombak": "1990-03-10", "Choa Chu Kang": "1990-03-10",
    "Yew Tee": "1996-02-10", "Kranji": "1996-02-10",
    "Marsiling": "1996-02-10", "Woodlands": "1996-02-10",
    "Admiralty": "1996-02-10", "Sembawang": "1996-02-10",
    "Canberra": "2019-11-02", "Yishun": "1988-12-20",
    "Khatib": "1988-12-20", "Yio Chu Kang": "1987-11-07",
    "Ang Mo Kio": "1987-11-07", "Bishan": "1987-11-07",
    "Braddell": "1987-11-07", "Toa Payoh": "1987-11-07",
    "Novena": "1987-11-07", "Newton": "1987-11-07",
    "Orchard": "1987-12-12", "Somerset": "1987-12-12",
    "Dhoby Ghaut": "1987-12-12", "City Hall": "1987-12-12",
    "Raffles Place": "1987-12-12", "Marina Bay": "1989-11-04",
    "Marina South Pier": "2014-11-23", "Marina South": "2014-11-23",
    "Pasir Ris": "1989-12-16", "Tampines": "1989-12-16",
    "Simei": "1989-12-16", "Tanah Merah": "1989-12-16",
    "Bedok": "1989-12-16", "Kembangan": "1989-12-16",
    "Eunos": "1989-12-16", "Paya Lebar": "1989-12-16",
    "Aljunied": "1989-12-16", "Kallang": "1989-12-16",
    "Lavender": "1989-12-16", "Bugis": "1989-12-16",
    "Tanjong Pagar": "1987-12-12", "Outram Park": "1988-03-12",
    "Tiong Bahru": "1988-03-12", "Redhill": "1988-03-12",
    "Queenstown": "1988-03-12", "Commonwealth": "1988-03-12",
    "Buona Vista": "1988-03-12", "Dover": "2001-10-18",
    "Clementi": "1988-03-12", "Chinese Garden": "1988-11-05",
    "Lakeside": "1988-11-05", "Boon Lay": "1990-03-06",
    "Pioneer": "1990-03-06", "Joo Koon": "1990-03-06",
    "Gul Circle": "2017-06-18", "Tuas Crescent": "2017-06-18",
    "Tuas West Road": "2017-06-18", "Tuas Link": "2017-06-18",
    "Expo": "2001-01-10", "Changi Airport": "2002-02-08",
    "Harbourfront": "2003-06-20", "Chinatown": "2003-06-20",
    "Clarke Quay": "2003-06-20", "Little India": "2003-06-20",
    "Farrer Park": "2003-06-20", "Boon Keng": "2003-06-20",
    "Potong Pasir": "2003-06-20", "Woodleigh": "2011-06-20",
    "Serangoon": "2003-06-20", "Kovan": "2003-06-20",
    "Hougang": "2003-06-20", "Buangkok": "2006-01-15",
    "Sengkang": "2003-06-20", "Punggol": "2003-06-20",
    "Punggol Coast": "2024-12-07",
    "Bras Basah": "2010-04-17", "Esplanade": "2010-04-17",
    "Promenade": "2010-04-17", "Nicoll Highway": "2010-04-17",
    "Stadium": "2010-04-17", "Mountbatten": "2010-04-17",
    "Dakota": "2010-04-17", "Macpherson": "2010-04-17",
    "Tai Seng": "2009-05-28", "Bartley": "2009-05-28",
    "Lorong Chuan": "2009-05-28", "Marymount": "2009-05-28",
    "Caldecott": "2011-10-08", "Botanic Gardens": "2011-10-08",
    "Farrer Road": "2011-10-08", "Holland Village": "2011-10-08",
    "One-North": "2011-10-08", "Kent Ridge": "2011-10-08",
    "Haw Par Villa": "2011-10-08", "Pasir Panjang": "2011-10-08",
    "Labrador Park": "2011-10-08", "Telok Blangah": "2011-10-08",
    "Bayfront": "2012-01-14",
    "Bukit Panjang": "2015-12-27", "Cashew": "2015-12-27",
    "Hillview": "2015-12-27", "Hume": "2025-06-21",
    "Beauty World": "2015-12-27", "King Albert Park": "2015-12-27",
    "Sixth Avenue": "2015-12-27", "Tan Kah Kee": "2015-12-27",
    "Stevens": "2017-10-21", "Rochor": "2017-10-21",
    "Downtown": "2017-10-21", "Telok Ayer": "2017-10-21",
    "Fort Canning": "2017-10-21", "Bencoolen": "2017-10-21",
    "Jalan Besar": "2017-10-21", "Bendemeer": "2017-10-21",
    "Geylang Bahru": "2017-10-21", "Mattar": "2017-10-21",
    "Ubi": "2017-10-21", "Kaki Bukit": "2017-10-21",
    "Bedok North": "2017-10-21", "Bedok Reservoir": "2017-10-21",
    "Tampines West": "2017-10-21", "Tampines East": "2017-10-21",
    "Upper Changi": "2017-10-21",
    "Woodlands North": "2020-01-31", "Woodlands South": "2020-01-31",
    "Springleaf": "2021-08-28", "Lentor": "2021-08-28",
    "Mayflower": "2021-08-28", "Bright Hill": "2021-08-28",
    "Upper Thomson": "2021-08-28", "Mount Pleasant": "2025-12-01",
    "Napier": "2022-11-13", "Orchard Boulevard": "2022-11-13",
    "Great World": "2022-11-13", "Havelock": "2022-11-13",
    "Maxwell": "2022-11-13", "Shenton Way": "2022-11-13",
    "Gardens By The Bay": "2022-11-13",
    "Tanjong Rhu": "2024-06-23", "Katong Park": "2024-06-23",
    "Tanjong Katong": "2024-06-23", "Marine Parade": "2024-06-23",
    "Marine Terrace": "2024-06-23", "Siglap": "2024-06-23",
    "Bayshore": "2024-06-23",
    "South View": "1999-11-06", "Keat Hong": "1999-11-06",
    "Teck Whye": "1999-11-06", "Phoenix": "1999-11-06",
    "Petir": "1999-11-06", "Pending": "1999-11-06",
    "Bangkit": "1999-11-06", "Fajar": "1999-11-06",
    "Segar": "1999-11-06", "Jelapang": "1999-11-06",
    "Senja": "1999-11-06", "Ten Mile Junction": "1999-11-06",
    "Compassvale": "2003-01-18", "Rumbia": "2003-01-18",
    "Bakau": "2003-01-18", "Kangkar": "2003-01-18",
    "Ranggung": "2003-01-18", "Cheng Lim": "2005-01-29",
    "Farmway": "2005-01-29", "Kupang": "2005-01-29",
    "Thanggam": "2005-01-29", "Fernvale": "2005-01-29",
    "Layar": "2005-01-29", "Tongkang": "2005-01-29",
    "Renjong": "2005-01-29", "Soo Teck": "2005-01-29",
    "Cove": "2005-01-29", "Meridian": "2005-01-29",
    "Coral Edge": "2005-01-29", "Riviera": "2005-01-29",
    "Kadaloor": "2005-01-29", "Oasis": "2005-01-29",
    "Damai": "2005-01-29", "Sam Kee": "2016-12-29",
    "Teck Lee": "2016-12-29", "Punggol Point": "2016-12-29",
    "Samudera": "2016-12-29", "Nibong": "2016-12-29",
    "Sumang": "2016-12-29",
    # Unbuilt stations — far-future date so they never appear in historical calculations
    "Bukit Brown": "2099-12-31",
    "Founders' Memorial": "2099-12-31",
    "Bocc": "2099-12-31",
}


def get_mrt_stations(shapefile_path: str) -> pd.DataFrame:
    """Load MRT/LRT stations from LTA shapefile and merge with opening dates.

    Coordinates: LTA DataMall (official, authoritative)
    Opening dates: public records (LTA press releases)
    """
    df = load_mrt_from_lta_shapefile(shapefile_path)

    date_lookup = {k.upper(): v for k, v in STATION_OPENING_DATES.items()}
    df["opening_date"] = df["station_name"].apply(
        lambda x: date_lookup.get(x.upper())
    )
    df["opening_date"] = pd.to_datetime(df["opening_date"])

    unmatched = df[df["opening_date"].isna()]
    if len(unmatched) > 0:
        print(f"WARNING: {len(unmatched)} stations without opening dates:")
        print(unmatched[["station_name", "type"]].to_string())

    return df


def get_shopping_malls() -> pd.DataFrame:
    """Return major Singapore shopping malls with lat/lng."""
    malls = [
        ("VivoCity", 1.264280, 103.822117),
        ("ION Orchard", 1.303920, 103.831940),
        ("Plaza Singapura", 1.300694, 103.845278),
        ("Bugis Junction", 1.299722, 103.855278),
        ("Bugis+", 1.300833, 103.854722),
        ("Suntec City", 1.295278, 103.859722),
        ("Marina Square", 1.291111, 103.857778),
        ("Raffles City", 1.293611, 103.852500),
        ("Funan", 1.291389, 103.849444),
        ("The Shoppes at Marina Bay Sands", 1.283889, 103.859167),
        ("Orchard Central", 1.301111, 103.839722),
        ("Wisma Atria", 1.304167, 103.833333),
        ("Ngee Ann City", 1.303611, 103.833611),
        ("Paragon", 1.304167, 103.835833),
        ("Mandarin Gallery", 1.303056, 103.835278),
        ("313@Somerset", 1.301111, 103.838056),
        ("Orchard Gateway", 1.300833, 103.839722),
        ("Far East Plaza", 1.306944, 103.833611),
        ("Lucky Plaza", 1.304167, 103.831944),
        ("Tanglin Mall", 1.307778, 103.823889),
        ("Great World City", 1.293611, 103.831111),
        ("HarbourFront Centre", 1.264722, 103.820833),
        ("Clementi Mall", 1.314722, 103.764444),
        ("West Mall", 1.350000, 103.749167),
        ("JCube", 1.333611, 103.740278),
        ("IMM", 1.334722, 103.746667),
        ("Westgate", 1.334167, 103.742778),
        ("Jem", 1.333611, 103.743056),
        ("Junction 8", 1.350278, 103.873889),
        ("NEX", 1.350833, 103.872222),
        ("Compass One", 1.392500, 103.895278),
        ("Waterway Point", 1.406667, 103.902222),
        ("Rivervale Mall", 1.392500, 103.904722),
        ("Tampines Mall", 1.353333, 103.945278),
        ("Tampines 1", 1.352778, 103.944444),
        ("Century Square", 1.352500, 103.944167),
        ("Our Tampines Hub", 1.353056, 103.940833),
        ("Eastpoint Mall", 1.342222, 103.953889),
        ("Bedok Mall", 1.324167, 103.930000),
        ("Bedok Point", 1.324722, 103.930278),
        ("Parkway Parade", 1.301389, 103.905278),
        ("i12 Katong", 1.304722, 103.900556),
        ("Paya Lebar Quarter", 1.317778, 103.892500),
        ("SingPost Centre", 1.316944, 103.893889),
        ("Hougang Mall", 1.371111, 103.893333),
        ("Heartland Mall (Kovan)", 1.360000, 103.885278),
        ("AMK Hub", 1.369722, 103.848889),
        ("Djitsun Mall", 1.369722, 103.849444),
        ("Northpoint City", 1.429167, 103.835833),
        ("Causeway Point", 1.436389, 103.785833),
        ("Lot One", 1.385278, 103.744167),
        ("Jurong Point", 1.339722, 103.706944),
        ("Pioneer Mall", 1.337500, 103.697222),
        ("White Sands", 1.372778, 103.949444),
        ("Elias Mall", 1.374167, 103.950556),
        ("Loyang Point", 1.364167, 103.962778),
        ("Changi City Point", 1.334722, 103.962222),
        ("Sun Plaza", 1.432500, 103.773889),
        ("Canberra Plaza", 1.443056, 103.829722),
        ("Sembawang Shopping Centre", 1.449167, 103.820000),
        ("Bukit Panjang Plaza", 1.378333, 103.763889),
        ("Hillion Mall", 1.378056, 103.763611),
        ("Junction 10", 1.378611, 103.761389),
        ("The Clementi Mall", 1.314722, 103.764444),
        ("Rochester Mall", 1.305556, 103.788889),
        ("Star Vista", 1.306944, 103.790833),
        ("Holland Village Shopping Centre", 1.312222, 103.795833),
        ("Tiong Bahru Plaza", 1.286667, 103.827222),
        ("Anchorpoint", 1.288889, 103.802500),
        ("IKEA Alexandra", 1.287778, 103.800833),
        ("Alexandra Retail Centre", 1.288333, 103.801111),
        ("Chinatown Point", 1.285278, 103.844444),
        ("People's Park Complex", 1.284444, 103.842222),
        ("People's Park Centre", 1.284167, 103.841389),
        ("Toa Payoh Hub", 1.332778, 103.847222),
        ("Square 2", 1.320278, 103.843889),
        ("United Square", 1.320278, 103.843889),
        ("Thomson Plaza", 1.353056, 103.834444),
        ("Yew Tee Point", 1.397222, 103.747222),
        ("Admiralty Place", 1.440556, 103.800556),
        ("Woodlands Civic Centre", 1.437222, 103.786111),
        ("Yishun Town Square", 1.429444, 103.835556),
        ("Wisteria Mall", 1.397222, 103.747500),
        ("Oasis Terraces", 1.403611, 103.913056),
        ("Punggol Plaza", 1.404444, 103.902500),
        ("The Seletar Mall", 1.390833, 103.876944),
        ("Greenwich V", 1.397222, 103.905833),
    ]

    df = pd.DataFrame(malls, columns=["mall_name", "lat", "lng"])
    return df


MATURE_ESTATES = {
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH",
    "CENTRAL AREA", "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA",
    "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON",
    "TAMPINES", "TOA PAYOH",
}
