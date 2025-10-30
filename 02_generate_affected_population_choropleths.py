"""
Jamaica Flood Monitoring - Parameterized Provenance + Flood Density Analysis

General-purpose script that can be run for any date range.

Usage:
    python flood_provenance_analysis.py --end-date 2025-10-27 --days-back 7
    python flood_provenance_analysis.py --end-date 2024-07-15 --days-back 3

Key Logic for Flood Density:
- Only uses flood pixels from their LATEST provenance date
- If a newer observation covers an area, we ignore older data from that area
- This avoids counting "stale" flood pixels that may have receded

Output:
- Saves provenance + flood density map as PNG
"""

import sys
import argparse
from datetime import datetime, timedelta
import json
import math
import os
import tempfile
from pathlib import Path

import pystac_client
import stackstac
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import Point, mapping
from fsspec.implementations.http import HTTPFileSystem
from scipy.ndimage import gaussian_filter
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
import rioxarray
import exactextract
from adjustText import adjust_text
import ocha_stratus as stratus
from dotenv import load_dotenv

from src.ds_flood_gfm.geo_utils import (
    get_highest_admin_level,
    calculate_admin_population,
    load_fieldmaps_parquet,
    generate_cache_key,
    generate_rdylgn_colors,
)
from src.ds_flood_gfm.country_config import (
    get_country_config,
    get_bbox,
    GHSL_RASTER_BLOB_PATH,
)
from src.ds_flood_gfm.blob import save_plot_to_blob

load_dotenv()

# Constants
PIXEL_AREA_RATIO = 25  # (100m GHSL pixel / 20m GFM pixel)² = (100/20)² = 25
HISTOGRAM_BINS = 200  # Number of bins for 2D histogram density maps
GAUSSIAN_SIGMA = 2  # Sigma for Gaussian smoothing filter


def save_cache(
    cache_dir,
    cache_key,
    flood_points,
    provenance_indexed,
    provenance_target,
    unique_dates,
    metadata,
):
    """Save processed data to cache."""
    cache_path = Path(cache_dir) / cache_key
    cache_path.mkdir(parents=True, exist_ok=True)

    # Save flood points as GeoParquet
    if len(flood_points) > 0:
        gdf_floods = gpd.GeoDataFrame(flood_points, crs="EPSG:4326")
        gdf_floods.to_parquet(cache_path / "flood_points.parquet")

    # Save provenance raster as GeoTIFF
    transform = from_bounds(
        float(provenance_target.x.min()),
        float(provenance_target.y.min()),
        float(provenance_target.x.max()),
        float(provenance_target.y.max()),
        provenance_indexed.shape[1],
        provenance_indexed.shape[0],
    )

    with rasterio.open(
        cache_path / "provenance.tif",
        "w",
        driver="GTiff",
        height=provenance_indexed.shape[0],
        width=provenance_indexed.shape[1],
        count=1,
        dtype=provenance_indexed.dtype,
        crs="EPSG:4326",
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(provenance_indexed, 1)

    # Save metadata as JSON
    metadata_serializable = {
        "unique_dates": [str(d) for d in unique_dates],
        "x_min": float(provenance_target.x.min()),
        "x_max": float(provenance_target.x.max()),
        "y_min": float(provenance_target.y.min()),
        "y_max": float(provenance_target.y.max()),
        **metadata,
    }

    with open(cache_path / "metadata.json", "w") as f:
        json.dump(metadata_serializable, f, indent=2)

    print(f"\n✓ Saved cache to: {cache_path}")


def load_cache(cache_dir, cache_key):
    """Load processed data from cache."""
    cache_path = Path(cache_dir) / cache_key

    if not cache_path.exists():
        return None

    # Check all required files exist
    required_files = ["metadata.json", "provenance.tif"]
    if not all((cache_path / f).exists() for f in required_files):
        return None

    print(f"\n✓ Loading from cache: {cache_path}")

    # Load metadata
    with open(cache_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Load flood points if they exist
    flood_points_file = cache_path / "flood_points.parquet"
    if flood_points_file.exists():
        gdf_floods = gpd.read_parquet(flood_points_file)
        flood_points = gdf_floods.to_dict("records")
        # Restore geometry as Point objects
        for fp in flood_points:
            fp["geometry"] = Point(fp["lon"], fp["lat"])
    else:
        flood_points = []

    # Load provenance raster
    with rasterio.open(cache_path / "provenance.tif") as src:
        provenance_indexed = src.read(1)

    # Convert unique_dates back to datetime
    unique_dates = [np.datetime64(d) for d in metadata["unique_dates"]]

    return {
        "flood_points": flood_points,
        "provenance_indexed": provenance_indexed,
        "unique_dates": unique_dates,
        "metadata": metadata,
    }


def main(
    end_date_str,
    n_latest,
    iso3="JAM",
    cache_dir="data/cache",
    use_cache=True,
    flood_mode="latest",
    output_dir="outputs/plots",
):
    """
    Run flood provenance analysis using the N most recent observations before end_date.

    Parameters:
    -----------
    end_date_str : str
        End date in YYYY-MM-DD format
    n_latest : int
        Number of most recent observations to use
    iso3 : str
        ISO3 country code (default: JAM for Jamaica)
    cache_dir : str
        Directory for caching processed data
    use_cache : bool
        Whether to use cached data if available
    flood_mode : str
        'latest' = only use flood pixels from latest provenance (default)
        'cumulative' = sum across all dates (conservative extent)
    output_dir : str
        Directory for output files (default: outputs/plots)
    """

    # Use population raster constant
    population_raster = GHSL_RASTER_BLOB_PATH

    # Parse dates - look back 15 days to find available data
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    search_start_date = end_date - timedelta(days=15)

    start_date_str = search_start_date.strftime("%Y-%m-%d")

    print("=" * 80)
    print(f"FLOOD PROVENANCE ANALYSIS: {iso3}")
    print(f"End date: {end_date_str}")
    print(f"Searching for {n_latest} most recent observations (looking back 15 days)")
    print("=" * 80)

    # Ensure output directory exists
    # Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get bounding box from country config
    bbox = get_bbox(iso3)
    print(f"\nBounding box: {bbox}")

    # Load admin1 boundaries (try blob, fallback to HTTP)
    try:
        gdf_admin1 = load_fieldmaps_parquet(iso3, adm_level=1, admin_source="blob")
        print(f"Loaded {len(gdf_admin1)} admin1 divisions from blob storage")
    except Exception as e:
        print(f"Blob load failed, using HTTP fallback")
        gdf_admin1 = load_fieldmaps_parquet(iso3, adm_level=1, admin_source="http")
        print(f"Loaded {len(gdf_admin1)} admin1 divisions from HTTP")

    # Create ADM0 (country boundary) by dissolving admin1
    gdf_aoi = gdf_admin1.dissolve()

    # ========== STEP 1: Quick STAC query to get available dates (FAST!) ==========
    stac_api = "https://stac.eodc.eu/api/v1"
    client = pystac_client.Client.open(stac_api)

    # Use intersects with actual geometry instead of bbox to avoid false positives
    # (STAC catalog sometimes returns tiles that don't actually cover the AOI)
    aoi_geojson = mapping(gdf_aoi.geometry.iloc[0])

    search = client.search(
        collections=["GFM"],
        intersects=aoi_geojson,
        datetime=f"{start_date_str}/{end_date_str}",
    )
    items = search.item_collection()
    print(f"Found {len(items)} STAC items (using intersects query)")

    if len(items) == 0:
        print("ERROR: No STAC items found for this date range!")
        return

    # Extract unique dates from STAC items (fast - no raster loading!)
    print("Extracting observation dates from STAC metadata...")
    all_dates_set = set()
    for item in items:
        dt = pd.to_datetime(item.datetime)
        all_dates_set.add(dt.date())

    all_dates = sorted(list(all_dates_set))
    print(f"All dates found: {[str(d) for d in all_dates]}")

    if len(all_dates) == 0:
        print("ERROR: No valid dates in STAC items!")
        return

    # Select only the N most recent dates
    dates_to_use = all_dates[-n_latest:] if len(all_dates) >= n_latest else all_dates
    print(
        f"Using {len(dates_to_use)} most recent dates: {[str(d) for d in dates_to_use]}"
    )

    # Convert to numpy datetime64 for cache key generation
    dates = np.array([np.datetime64(d) for d in dates_to_use])

    # ========== STEP 2: Check cache with actual observation dates ==========
    cache_key = generate_cache_key(iso3, dates, population_raster, flood_mode)
    print(f"\nCache key: {cache_key}")

    if use_cache:
        cached_data = load_cache(cache_dir, cache_key)
        if cached_data:
            print("✓ Using cached data - creating visualization...")
            latest_date = str(dates[-1])[:10]

            # Reconstruct minimal objects for visualization
            flood_points = cached_data["flood_points"]
            provenance_indexed = cached_data["provenance_indexed"]
            unique_dates = cached_data["unique_dates"]
            metadata = cached_data["metadata"]

            # Create mock xarray for extent info
            x_coords = np.linspace(
                metadata["x_min"], metadata["x_max"], provenance_indexed.shape[1]
            )
            y_coords = np.linspace(
                metadata["y_max"], metadata["y_min"], provenance_indexed.shape[0]
            )
            provenance_target = xr.DataArray(
                provenance_indexed,
                coords={"y": y_coords, "x": x_coords},
                dims=["y", "x"],
            )

            # Call visualization directly
            create_map_from_cache(
                latest_date,
                flood_points,
                provenance_indexed,
                provenance_target,
                unique_dates,
                gdf_aoi,
                gdf_admin1,
                output_dir,
                population_raster,
                iso3,
                flood_mode,
            )

            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE (from cache)")
            print("=" * 80)
            return
    # ========== END CACHE CHECK ==========

    # ========== STEP 3: Cache miss - build xarray stack (SLOW!) ==========
    print("\n✗ Cache miss - building xarray stack...")
    print("Building xarray stack...")
    stack = stackstac.stack(
        items, epsg=4326, chunksize=512  # 512×512 spatial chunks for memory efficiency
    )
    print(f"Stack created with chunks: {stack.chunks}")
    stack_flood = stack.sel(band="ensemble_flood_extent")
    stack_flood_clipped = stack_flood.sel(
        x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1])
    )

    # Create daily composites
    print("Creating daily composites...")
    stack_flood_max = stack_flood_clipped.groupby("time.date").max()
    stack_flood_max = stack_flood_max.rename({"date": "time"})
    stack_flood_max["time"] = stack_flood_max.time.astype("datetime64[ns]")

    # Filter stack to only selected dates
    stack_flood_max = stack_flood_max.sel(time=dates)

    # Create 'ever_has_data' mask (vectorized - single operation instead of loop)
    print("\nCreating 'ever_has_data' mask (vectorized)...")
    ever_has_data = (~stack_flood_max.isnull()).any(dim="time")

    pixels_with_data = ever_has_data.sum().values
    pixels_no_data = (~ever_has_data).sum().values
    print(f"  Pixels with data: {pixels_with_data:,}")
    print(f"  Pixels without data: {pixels_no_data:,}")
    print(
        f"  Coverage: {100 * pixels_with_data / (pixels_with_data + pixels_no_data):.1f}%"
    )

    # Track provenance (vectorized - NO LOOPS!)
    print("\nTracking provenance (vectorized)...")
    flood_filled = stack_flood_max.ffill(dim="time")
    provenance_filled = (
        stack_flood_max.time.where(~stack_flood_max.isnull())
        .ffill(dim="time")
        .where(ever_has_data)
    )

    # Use the latest available date instead of the requested end_date
    latest_date = str(dates[-1])[:10]
    print(f"\nCreating provenance + flood map for {latest_date} (latest available)...")
    create_map_from_stac(
        latest_date,
        flood_filled,
        provenance_filled,
        stack_flood_max,
        dates,
        gdf_aoi,
        gdf_admin1,
        output_dir,
        population_raster,
        cache_dir,
        cache_key,
        flood_mode,
        iso3,
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def create_map_from_cache(
    target_date,
    flood_points,
    provenance_indexed,
    provenance_target,
    unique_dates,
    gdf_aoi,
    gdf_admin1,
    output_dir,
    population_raster=None,
    iso3="JAM",
    flood_mode="latest",
):
    """Create visualization from cached data (no processing - visualization only)."""
    print("Creating visualization from cached data...")

    # Get country-specific legend placement
    country_config = get_country_config(iso3)
    legend_loc = country_config["legend_location"]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Colors: oldest to most recent (red -> orange -> yellow -> green)
    colors = generate_rdylgn_colors(len(unique_dates))
    cmap_prov = ListedColormap(colors)

    # Plot grey background for no data
    grey_layer = np.ones(provenance_indexed.shape) * 0.5
    ax.imshow(
        grey_layer,
        cmap="Greys",
        extent=[
            float(provenance_target.x.min()),
            float(provenance_target.x.max()),
            float(provenance_target.y.min()),
            float(provenance_target.y.max()),
        ],
        origin="upper",
        vmin=0,
        vmax=1,
        zorder=1,
    )

    # Plot provenance
    prov_masked = np.ma.masked_where(provenance_indexed == -1, provenance_indexed)
    ax.imshow(
        prov_masked,
        cmap=cmap_prov,
        extent=[
            float(provenance_target.x.min()),
            float(provenance_target.x.max()),
            float(provenance_target.y.min()),
            float(provenance_target.y.max()),
        ],
        origin="upper",
        interpolation="nearest",
        vmin=0,
        vmax=len(unique_dates) - 1,
        zorder=2,
    )

    # Create flood density heatmap (population-weighted if available)
    if len(flood_points) > 0:
        gdf_floods = gpd.GeoDataFrame(flood_points, crs="EPSG:4326")

        # Check if we have population data
        has_population = population_raster and "population" in gdf_floods.columns

        if has_population:
            print(
                f"Creating AFFECTED POPULATION density heatmap from {len(gdf_floods)} points..."
            )
        else:
            print(
                f"Creating FLOOD PIXEL density heatmap from {len(gdf_floods)} points..."
            )

        # Extract coordinates
        x_coords = gdf_floods.geometry.x.values
        y_coords = gdf_floods.geometry.y.values

        # Define extent for histogram
        x_min_flood = float(provenance_target.x.min())
        x_max_flood = float(provenance_target.x.max())
        y_min_flood = float(provenance_target.y.min())
        y_max_flood = float(provenance_target.y.max())

        # Create 2D histogram (density map)
        bins_x = HISTOGRAM_BINS
        bins_y = HISTOGRAM_BINS

        if has_population:
            # Population-weighted histogram
            weights = gdf_floods["population"].values
            heatmap, xedges, yedges = np.histogram2d(
                x_coords,
                y_coords,
                bins=[bins_x, bins_y],
                range=[[x_min_flood, x_max_flood], [y_min_flood, y_max_flood]],
                weights=weights,
            )
        else:
            # Simple pixel count histogram
            heatmap, xedges, yedges = np.histogram2d(
                x_coords,
                y_coords,
                bins=[bins_x, bins_y],
                range=[[x_min_flood, x_max_flood], [y_min_flood, y_max_flood]],
            )

        # Apply Gaussian smoothing
        heatmap_smooth = gaussian_filter(heatmap.T, sigma=GAUSSIAN_SIGMA)

        # Mask zeros
        heatmap_smooth = np.ma.masked_where(heatmap_smooth == 0, heatmap_smooth)

        # Colormap: purple for population, blue for flood pixels
        if has_population:
            # Purple colormap for population density
            colors_density = ["#00000000", "#E8DAEF", "#BB8FCE", "#8E44AD", "#5B2C6F"]
        else:
            # Electric blue colormap for flood pixel density
            colors_density = ["#00000000", "#00BFFF", "#0080FF", "#0040FF"]

        cmap_density = LinearSegmentedColormap.from_list(
            "density", colors_density, N=100
        )

        # Plot density heatmap
        im_density = ax.imshow(
            heatmap_smooth,
            extent=[x_min_flood, x_max_flood, y_min_flood, y_max_flood],
            origin="lower",
            cmap=cmap_density,
            alpha=0.7,
            interpolation="bilinear",
            zorder=4,
        )

        # Add colorbar for density
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(im_density, cax=cax)

        if has_population:
            cbar.set_label(
                "Affected Population Density", rotation=270, labelpad=20, fontsize=10
            )
            print(f"Plotted affected population density heatmap")
        else:
            cbar.set_label(
                "Flood Point Density", rotation=270, labelpad=20, fontsize=10
            )
            print(f"Plotted flood pixel density heatmap")

    # Add boundaries
    gdf_aoi.boundary.plot(ax=ax, color="black", linewidth=1.5, alpha=0.8, zorder=5)
    gdf_admin1.boundary.plot(ax=ax, color="black", linewidth=0.8, alpha=0.6, zorder=5)

    # Set extent
    x_min, x_max = float(provenance_target.x.min()), float(provenance_target.x.max())
    y_min, y_max = float(provenance_target.y.min()), float(provenance_target.y.max())
    x_buffer = (x_max - x_min) * 0.02
    y_buffer = (y_max - y_min) * 0.02
    ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

    # Title
    ax.set_title(
        f"Data Provenance Map - {target_date}\n(Latest observation date per pixel)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5, zorder=0)

    # Legend
    legend_elements = [Patch(facecolor="#808080", label="No data")]
    for i, date in enumerate(unique_dates):
        legend_elements.append(
            Patch(facecolor=colors[i], label=str(pd.Timestamp(date))[:10])
        )

    ax.legend(
        handles=legend_elements,
        title="Latest Observation Date",
        loc="upper right",
        fontsize=10,
        title_fontsize=11,
        framealpha=0.95,
        edgecolor="black",
        fancybox=False,
    )

    plt.tight_layout()

    # Save with descriptive filename
    density_type = (
        "population"
        if (
            population_raster
            and len(flood_points) > 0
            and "population" in gdf_floods.columns
        )
        else "flood"
    )
    output_filename = (
        f"{iso3}_{density_type}_provenance_{target_date.replace('-', '')}.png"
    )
    output_path = f"{output_dir}/{output_filename}"
    save_plot_to_blob(plt, output_path)
    #plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_path}")
    plt.close()

    # ========== GENERATE CHOROPLETH (if population data available) ==========
    if (
        population_raster
        and len(flood_points) > 0
        and "population" in gdf_floods.columns
    ):
        print("\n" + "=" * 80)
        print("GENERATING CHOROPLETH MAP")
        print("=" * 80)

        try:
            # Get admin level from country config
            country_config = get_country_config(iso3)
            adm_level = country_config["choropleth_adm_level"]
            gdf_admin = load_fieldmaps_parquet(
                iso3, adm_level=adm_level, admin_source="blob"
            )
            print(f"Using ADM{adm_level} boundaries: {len(gdf_admin)} divisions")

            # Load ADM1 for overlay
            gdf_admin1_overlay = load_fieldmaps_parquet(
                iso3, adm_level=1, admin_source="blob"
            )
            print(f"Loading ADM1 for overlay: {len(gdf_admin1_overlay)} divisions")

            # Calculate affected population by admin division
            gdf_choropleth = calculate_admin_population(
                gdf_floods, gdf_admin, adm_level
            )

            # Separate unassigned pseudo-divisions for reporting (but exclude from visualization)
            adm_name_col = f"adm{adm_level}_name"
            is_unassigned = gdf_choropleth[adm_name_col].str.contains(
                "Unassigned", na=False
            )
            gdf_unassigned = gdf_choropleth[is_unassigned].copy()
            gdf_choropleth_viz = gdf_choropleth[~is_unassigned].copy()

            # Print summary (including unassigned)
            affected_divs_viz = gdf_choropleth_viz[
                gdf_choropleth_viz["affected_pop"] > 0
            ]
            total_pop = gdf_choropleth["affected_pop"].sum()
            print(
                f"Divisions with affected population: {len(affected_divs_viz)}/{len(gdf_choropleth_viz)}"
            )
            if len(gdf_unassigned) > 0:
                unassigned_pop = gdf_unassigned["affected_pop"].sum()
                print(
                    f"  + {len(gdf_unassigned)} unassigned pseudo-divisions: {unassigned_pop:.0f} people"
                )
            print(f"Total affected population: {total_pop:,.0f}")

            # Create choropleth figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")

            # Find best available name column (fallback if adm_name is None)
            adm_name_col = f"adm{adm_level}_name"

            # Check if the primary name column has values, if not fallback to lower admin levels
            if gdf_choropleth_viz[adm_name_col].isna().all():
                print(
                    f"  Note: {adm_name_col} is empty, falling back to lower admin levels for labels"
                )
                for fallback_level in range(adm_level - 1, 0, -1):
                    fallback_col = f"adm{fallback_level}_name"
                    if (
                        fallback_col in gdf_choropleth_viz.columns
                        and not gdf_choropleth_viz[fallback_col].isna().all()
                    ):
                        print(f"  Using {fallback_col} for labels")
                        adm_name_col = fallback_col
                        break

            # Plot choropleth with improved color scaling
            # Use vmin=0 and vmax=max_pop to ensure full color range is used
            max_pop = gdf_choropleth_viz["affected_pop"].max()

            # For very small values, use a classification scheme instead of continuous
            if max_pop > 0 and max_pop < 100:
                # Use a categorical/binned approach for small values
                # Create bins: 0, 1-5, 5-10, 10-20, 20+
                bins = [0, 1, 5, 10, 20, max_pop + 1]
                norm = BoundaryNorm(bins, ncolors=256)

                # Create custom colormap with white for 0
                colors_list = [
                    "white",
                    "#ffffcc",
                    "#ffeda0",
                    "#fed976",
                    "#feb24c",
                    "#fd8d3c",
                    "#fc4e2a",
                    "#e31a1c",
                    "#bd0026",
                    "#800026",
                ]
                cmap_custom = LinearSegmentedColormap.from_list(
                    "white_ylorrd", colors_list, N=256
                )

                gdf_choropleth_viz.plot(
                    column="affected_pop",
                    ax=ax,
                    cmap=cmap_custom,
                    edgecolor="#c0c0c0",  # Happy medium - darker grey, visible but not distracting
                    linewidth=0.15,
                    legend=True,
                    norm=norm,
                    legend_kwds={
                        "label": "Affected Population",
                        "orientation": "vertical",
                        "shrink": 0.6,
                    },
                )
            else:
                # Use continuous scale with vmin=0 for larger populations
                # Create custom colormap with white for 0
                colors_list = [
                    "white",
                    "#ffffcc",
                    "#ffeda0",
                    "#fed976",
                    "#feb24c",
                    "#fd8d3c",
                    "#fc4e2a",
                    "#e31a1c",
                    "#bd0026",
                    "#800026",
                ]
                cmap_custom = LinearSegmentedColormap.from_list(
                    "white_ylorrd", colors_list, N=256
                )

                gdf_choropleth_viz.plot(
                    column="affected_pop",
                    ax=ax,
                    cmap=cmap_custom,
                    edgecolor="#c0c0c0",  # Happy medium - darker grey, visible but not distracting
                    linewidth=0.15,
                    legend=True,
                    vmin=0,
                    vmax=max_pop,
                    legend_kwds={
                        "label": "Affected Population",
                        "orientation": "vertical",
                        "shrink": 0.6,
                    },
                )

            # Add ADM3 boundaries colored by data provenance (using exactextract - blazingly fast!)
            print("  Extracting modal provenance for ADM3 boundaries (exactextract)...")

            # Map provenance index to colors: oldest=red, newest=green
            colors = generate_rdylgn_colors(len(unique_dates))
            date_to_color = {i: colors[i] for i in range(len(unique_dates))}

            # Write provenance raster to temp file for exactextract
            # exactextract needs a file path, not in-memory array
            # Get provenance as numpy array (handle both xarray and numpy)
            if isinstance(provenance_indexed, xr.DataArray):
                prov_data = provenance_indexed.values.astype("int16")
                x_min, x_max = float(provenance_indexed.x.min()), float(
                    provenance_indexed.x.max()
                )
                y_min, y_max = float(provenance_indexed.y.min()), float(
                    provenance_indexed.y.max()
                )
            else:
                # Already a numpy array (from cache)
                prov_data = provenance_indexed.astype("int16")
                # Get bounds from provenance_target
                x_min, x_max = float(provenance_target.x.min()), float(
                    provenance_target.x.max()
                )
                y_min, y_max = float(provenance_target.y.min()), float(
                    provenance_target.y.max()
                )

            # Create affine transform from coordinates
            height, width = prov_data.shape
            transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

            # Write to temporary file
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp_raster_path = tmp.name

            with rasterio.open(
                tmp_raster_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="int16",
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(prov_data, 1)

            # Use exactextract to get mode of provenance_indexed for each polygon
            # This is MUCH faster than rasterstats!
            modal_prov = exactextract.exact_extract(
                tmp_raster_path,
                gdf_admin,
                ["mode"],
                include_cols=[f"adm{adm_level}_id"],
                output="pandas",
            )

            # Clean up temp file
            os.unlink(tmp_raster_path)

            # Map results to colors
            prov_colors = []
            for _, row in modal_prov.iterrows():
                mode_val = row["mode"]
                if pd.isna(mode_val) or mode_val == -1:
                    prov_colors.append("lightgray")
                else:
                    prov_colors.append(date_to_color.get(int(mode_val), "lightgray"))

            gdf_admin["prov_color"] = prov_colors

            # Dissolve by provenance color to create one polygon per color group
            print("  Dissolving by provenance color...")
            dissolved_provenance = gdf_admin.dissolve(by="prov_color", as_index=False)

            # Create mapping from color back to date for legend
            color_to_date = {
                color: unique_dates[i] for i, color in date_to_color.items()
            }

            # Plot no-data areas with grey fill (70% opacity)
            no_data_polys = dissolved_provenance[
                dissolved_provenance["prov_color"] == "lightgray"
            ]
            if len(no_data_polys) > 0:
                no_data_polys.plot(
                    ax=ax, facecolor="lightgrey", edgecolor="none", alpha=0.7, zorder=1
                )

            # Plot dissolved provenance boundaries (thick colored lines)
            legend_elements = []

            for idx, row in dissolved_provenance.iterrows():
                color = row["prov_color"]
                if color != "lightgray":  # Plot colored provenance zones
                    gpd.GeoSeries([row.geometry]).boundary.plot(
                        ax=ax, edgecolor=color, linewidth=2.5, alpha=0.9, zorder=3
                    )

            # Build legend in chronological order (oldest to newest, top to bottom)
            # Sort dates and add legend entries in that order
            sorted_dates = sorted(unique_dates)
            unique_dates_list = list(
                unique_dates
            )  # Convert numpy array to list for .index()
            for date in sorted_dates:
                # Find which index this date corresponds to
                date_idx = unique_dates_list.index(date)
                color = date_to_color.get(date_idx)
                if color:
                    date_str = str(date)[:10]
                    legend_elements.append(
                        Patch(
                            facecolor="none",
                            edgecolor=color,
                            linewidth=2.5,
                            label=date_str,
                        )
                    )

            # Add no-data to legend
            if len(no_data_polys) > 0:
                legend_elements.append(
                    Patch(
                        facecolor="lightgrey",
                        alpha=0.7,
                        edgecolor="none",
                        label="No data",
                    )
                )

            # Plot all ADM3 boundaries in light grey (happy medium)
            gdf_admin.boundary.plot(
                ax=ax, edgecolor="#c0c0c0", linewidth=0.15, alpha=0.4, zorder=2
            )

            print(f"  Provenance boundaries (dissolved) and ADM3 borders plotted")

            # Add ADM1 boundaries on top (transparent fill, thicker lines)
            gdf_admin1_overlay.plot(
                ax=ax, facecolor="none", edgecolor="black", linewidth=1.5, alpha=0.8
            )

            # Add labels for top affected divisions (using adjustText to avoid overlaps)
            top_n = min(5, len(affected_divs_viz))
            if top_n > 0:
                top_affected = affected_divs_viz.nlargest(top_n, "affected_pop")
                texts = []
                for idx, row in top_affected.iterrows():
                    centroid = row.geometry.centroid
                    text = ax.text(
                        centroid.x,
                        centroid.y,
                        f"{row[adm_name_col]}\n({row['affected_pop']:.0f})",
                        fontsize=8,
                        weight="bold",
                        ha="center",
                        va="center",
                        color="black",
                        path_effects=[
                            withStroke(linewidth=3, foreground="white", alpha=0.7)
                        ],
                    )
                    texts.append(text)

                # Adjust text positions to avoid overlaps (like ggrepel!)
                adjust_text(
                    texts,
                    ax=ax,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5, alpha=0.6),
                )

            # Set axis limits to match provenance target extent (with 2% buffer)
            x_min, x_max = float(provenance_target.x.min()), float(
                provenance_target.x.max()
            )
            y_min, y_max = float(provenance_target.y.min()), float(
                provenance_target.y.max()
            )
            x_buffer = (x_max - x_min) * 0.02
            y_buffer = (y_max - y_min) * 0.02
            ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
            ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

            # Title and labels
            mode_label = "Cumulative" if flood_mode == "cumulative" else "Latest"
            ax.set_title(
                f"Affected Population by Admin{adm_level} Division ({mode_label} Mode) - {target_date}\n"
                f"Total: {total_pop:,.0f} people in {len(affected_divs_viz)} divisions",
                fontsize=14,
                fontweight="bold",
                pad=15,
            )
            # Remove axis labels and tick labels for cleaner map
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)
            ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5, zorder=0)

            # Add provenance legend
            if legend_elements:
                legend = ax.legend(
                    handles=legend_elements,
                    title="Data Provenance",
                    loc=legend_loc,  # From country config: JAM='lower left', HTI='upper left', CUB='lower right'
                    fontsize=9,
                    title_fontsize=10,
                    framealpha=0.85,  # Transparent white background (0=transparent, 1=opaque)
                    facecolor="white",
                    edgecolor="black",
                )

            plt.tight_layout()

            # Save choropleth
            mode_suffix = "cumulative" if flood_mode == "cumulative" else "latest"
            choropleth_filename = f"{iso3}_population_{mode_suffix}_adm{adm_level}_{target_date.replace('-', '')}.png"
            choropleth_path = f"{output_dir}/{choropleth_filename}"
            save_plot_to_blob(plt, choropleth_path)
            # plt.savefig(
            #     choropleth_path, dpi=150, bbox_inches="tight", facecolor="white"
            # )
            print(f"Saved: {choropleth_path}")
            plt.close()

        except Exception as e:
            print(f"WARNING: Could not generate choropleth: {e}")


def create_map_from_stac(
    target_date,
    flood_filled,
    provenance_filled,
    stack_flood_max,
    dates,
    gdf_aoi,
    gdf_admin1,
    output_dir,
    population_raster=None,
    cache_dir=None,
    cache_key=None,
    flood_mode="latest",
    iso3="JAM",
):
    """Process STAC flood data, extract flood pixels, cache results, and create visualizations.

    This function:
    1. Extracts flood pixels from STAC xarray data based on provenance
    2. Samples population values at flood locations (if population_raster provided)
    3. Saves processed data to cache (flood points, provenance raster, metadata)
    4. Creates provenance + density heatmap visualization
    5. Creates affected population choropleth by admin division

    Cache location: {cache_dir}/{cache_key}/ containing:
        - flood_points.parquet: GeoDataFrame of flood pixel locations with population
        - provenance.tif: Raster showing latest observation date per pixel
        - metadata.json: Processing metadata (dates, pixel counts, etc.)

    Parameters:
    -----------
    target_date : str
        Target date for visualization (YYYY-MM-DD)
    flood_filled : xr.DataArray
        Forward-filled flood extent data
    provenance_filled : xr.DataArray
        Forward-filled provenance dates
    stack_flood_max : xr.DataArray
        Original flood stack (daily max composites)
    dates : np.array
        Array of observation dates
    gdf_aoi : gpd.GeoDataFrame
        Area of interest boundary
    gdf_admin1 : gpd.GeoDataFrame
        Admin1 boundaries for overlay
    output_dir : str
        Directory for output PNG files
    population_raster : str, optional
        Path to population raster (if None, creates flood pixel density instead)
    cache_dir : str, optional
        Base cache directory
    cache_key : str, optional
        Cache subdirectory key
    flood_mode : str
        'latest' = only use flood pixels from latest provenance (default)
        'cumulative' = sum across all dates (conservative extent)
    iso3 : str
        ISO3 country code (default: JAM)
    """
    # Get country-specific legend placement
    country_config = get_country_config(iso3)
    legend_loc = country_config["legend_location"]

    # Extract data for target date
    # Use .compute() to load into memory (persist() would keep in distributed memory but requires dask.distributed)
    flood_target = flood_filled.sel(time=target_date).compute()
    provenance_target = provenance_filled.sel(time=target_date).compute()

    # Get unique provenance dates
    unique_dates = pd.unique(provenance_target.values.ravel())
    unique_dates = unique_dates[~pd.isna(unique_dates)]
    unique_dates = np.sort(unique_dates)

    print(f"\nProvenance breakdown:")
    total_valid = np.sum(~pd.isna(provenance_target.values))
    for date in unique_dates:
        count = np.sum(provenance_target.values == date)
        pct = 100 * count / total_valid
        print(f"  {str(pd.Timestamp(date))[:10]}: {count:,} pixels ({pct:.1f}%)")

    no_data_count = np.sum(pd.isna(provenance_target.values))
    pct_no_data = 100 * no_data_count / provenance_target.size
    print(f"  NO DATA: {no_data_count:,} pixels ({pct_no_data:.1f}% of bbox)")

    # Extract flood pixels based on flood_mode
    if flood_mode == "cumulative":
        print("\nExtracting flood pixels (CUMULATIVE mode - all dates combined)...")

        # Compute once and use vectorized any() to find pixels flooded on ANY date
        # This is much faster than looping and using set tracking
        stack_flood_computed = stack_flood_max.compute()

        # Print per-date stats for transparency
        for date in unique_dates:
            flood_count = np.sum(stack_flood_computed.sel(time=date).values == 1)
            print(f"  {str(pd.Timestamp(date))[:10]}: {flood_count:,} flood pixels")

        # Vectorized union: find pixels where flood == 1 on ANY date
        any_flooded = (stack_flood_computed == 1).any(dim="time").values

        # Extract coordinates efficiently (single operation)
        y_coords, x_coords = np.where(any_flooded)
        x_geo = provenance_target.x.values[x_coords]
        y_geo = provenance_target.y.values[y_coords]

        # Create flood points (using list comprehension for efficiency)
        flood_points = [
            {
                "geometry": Point(x, y),
                "date": "cumulative",  # No single date - it's a union across all dates
                "lon": x,
                "lat": y,
            }
            for x, y in zip(x_geo, y_geo)
        ]

        print(
            f"\nTotal unique flood pixels (cumulative across all dates): {len(flood_points):,}"
        )

    else:  # flood_mode == "latest"
        print("\nExtracting flood pixels (LATEST mode - by provenance date)...")
        flood_points = []

        # Compute all dates once outside loop to avoid redundant computation
        stack_flood_computed = stack_flood_max.compute()

        for date in unique_dates:
            # Get original flood data for this date (no compute - just indexing)
            original_flood = stack_flood_computed.sel(time=date)

            # Mask: provenance == date AND flood == 1
            is_this_provenance = provenance_target.values == date
            is_flooded = original_flood.values == 1

            flood_mask = is_this_provenance & is_flooded
            flood_count = np.sum(flood_mask)

            print(f"  {str(pd.Timestamp(date))[:10]}: {flood_count:,} flood pixels")

            # Convert to points
            if flood_count > 0:
                y_coords, x_coords = np.where(flood_mask)
                x_geo = provenance_target.x.values[x_coords]
                y_geo = provenance_target.y.values[y_coords]

                for x, y in zip(x_geo, y_geo):
                    flood_points.append(
                        {
                            "geometry": Point(x, y),
                            "date": str(pd.Timestamp(date))[:10],
                            "lon": x,
                            "lat": y,
                        }
                    )

        print(
            f"\nTotal unique flood pixels (latest provenance only): {len(flood_points):,}"
        )

    # Sample population values at flood locations if raster provided
    if population_raster and len(flood_points) > 0:
        print(f"\nSampling population values from: {population_raster}")
        try:
            print(f"  Accessing blob: {population_raster}")

            # Open blob COG directly (no "blob://" prefix needed)
            da_pop = stratus.open_blob_cog(
                population_raster, container_name="raster"
            ).squeeze(drop=True)

            # Clip to country bbox for faster processing
            min_x, min_y, max_x, max_y = gdf_aoi.total_bounds
            da_pop_clip = da_pop.rio.clip_box(
                minx=min_x, miny=min_y, maxx=max_x, maxy=max_y
            )

            # Sample at flood point locations using xarray selection
            # GHSL is 100m res (10,000 m²), GFM is 20m res (400 m²)
            total_affected_pop_raw = 0
            total_affected_pop_adjusted_raw = 0
            total_affected_pop_adjusted = 0
            for fp in flood_points:
                lon, lat = fp["lon"], fp["lat"]
                try:
                    # Use sel with nearest neighbor
                    pop_val_raw = float(
                        da_pop_clip.sel(x=lon, y=lat, method="nearest").values
                    )
                    if np.isnan(pop_val_raw) or pop_val_raw < 0:
                        pop_val_raw = 0

                    # Store all three population values
                    fp["population_raw"] = (
                        pop_val_raw  # Raw GHSL value (100m pixel = 10,000 m²)
                    )
                    fp["population_adjusted_raw"] = (
                        pop_val_raw / PIXEL_AREA_RATIO
                    )  # Fractional adjusted value
                    fp["population_adjusted"] = math.ceil(
                        fp["population_adjusted_raw"]
                    )  # Round up to nearest integer
                    fp["population"] = fp[
                        "population_adjusted"
                    ]  # Default to adjusted (integer)

                    total_affected_pop_raw += pop_val_raw
                    total_affected_pop_adjusted_raw += fp["population_adjusted_raw"]
                    total_affected_pop_adjusted += fp["population_adjusted"]
                except Exception as e:
                    fp["population_raw"] = 0
                    fp["population_adjusted_raw"] = 0
                    fp["population_adjusted"] = 0
                    fp["population"] = 0

            print(
                f"  Total affected population (raw GHSL values): {total_affected_pop_raw:,.1f}"
            )
            print(
                f"  Total affected population (adjusted raw): {total_affected_pop_adjusted_raw:,.2f}"
            )
            print(
                f"  Total affected population (adjusted, rounded up): {total_affected_pop_adjusted:,.0f}"
            )
            print(
                f"  Average population per flood pixel (adjusted): {total_affected_pop_adjusted/len(flood_points):.2f}"
            )

        except Exception as e:
            raise RuntimeError(f"Could not sample population raster: {e}") from e

    # Create indexed provenance array
    date_to_idx = {date: i for i, date in enumerate(unique_dates)}
    provenance_indexed = np.full(provenance_target.shape, -1, dtype=np.int32)
    for date, idx in date_to_idx.items():
        provenance_indexed[provenance_target.values == date] = idx

    # ========== SAVE TO CACHE ==========
    if cache_dir and cache_key:
        metadata = {
            "total_pixels": len(flood_points),
            "provenance_breakdown": {
                str(pd.Timestamp(date))[:10]: int(
                    np.sum(provenance_target.values == date)
                )
                for date in unique_dates
            },
            "no_data_pixels": int(np.sum(pd.isna(provenance_target.values))),
        }
        save_cache(
            cache_dir,
            cache_key,
            flood_points,
            provenance_indexed,
            provenance_target,
            unique_dates,
            metadata,
        )
    # ========== END SAVE CACHE ==========

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Colors: oldest to most recent (red -> orange -> yellow -> green)
    colors = generate_rdylgn_colors(len(unique_dates))
    cmap_prov = ListedColormap(colors)

    # Plot grey background for no data
    grey_layer = np.ones(provenance_indexed.shape) * 0.5
    ax.imshow(
        grey_layer,
        cmap="Greys",
        extent=[
            float(provenance_target.x.min()),
            float(provenance_target.x.max()),
            float(provenance_target.y.min()),
            float(provenance_target.y.max()),
        ],
        origin="upper",
        vmin=0,
        vmax=1,
        zorder=1,
    )

    # Plot provenance
    prov_masked = np.ma.masked_where(provenance_indexed == -1, provenance_indexed)
    ax.imshow(
        prov_masked,
        cmap=cmap_prov,
        extent=[
            float(provenance_target.x.min()),
            float(provenance_target.x.max()),
            float(provenance_target.y.min()),
            float(provenance_target.y.max()),
        ],
        origin="upper",
        interpolation="nearest",
        vmin=0,
        vmax=len(unique_dates) - 1,
        zorder=2,
    )

    # Create flood density heatmap (population-weighted if available)
    if len(flood_points) > 0:
        gdf_floods = gpd.GeoDataFrame(flood_points, crs="EPSG:4326")

        # Check if we have population data
        has_population = population_raster and "population" in gdf_floods.columns

        if has_population:
            print(
                f"Creating AFFECTED POPULATION density heatmap from {len(gdf_floods)} points..."
            )
        else:
            print(
                f"Creating FLOOD PIXEL density heatmap from {len(gdf_floods)} points..."
            )

        # Extract coordinates
        x_coords = gdf_floods.geometry.x.values
        y_coords = gdf_floods.geometry.y.values

        # Define extent for histogram
        x_min_flood = float(provenance_target.x.min())
        x_max_flood = float(provenance_target.x.max())
        y_min_flood = float(provenance_target.y.min())
        y_max_flood = float(provenance_target.y.max())

        # Create 2D histogram (density map)
        bins_x = HISTOGRAM_BINS
        bins_y = HISTOGRAM_BINS

        if has_population:
            # Population-weighted histogram
            weights = gdf_floods["population"].values
            heatmap, xedges, yedges = np.histogram2d(
                x_coords,
                y_coords,
                bins=[bins_x, bins_y],
                range=[[x_min_flood, x_max_flood], [y_min_flood, y_max_flood]],
                weights=weights,
            )
        else:
            # Simple pixel count histogram
            heatmap, xedges, yedges = np.histogram2d(
                x_coords,
                y_coords,
                bins=[bins_x, bins_y],
                range=[[x_min_flood, x_max_flood], [y_min_flood, y_max_flood]],
            )

        # Apply Gaussian smoothing
        heatmap_smooth = gaussian_filter(heatmap.T, sigma=GAUSSIAN_SIGMA)

        # Mask zeros
        heatmap_smooth = np.ma.masked_where(heatmap_smooth == 0, heatmap_smooth)

        # Colormap: purple for population, blue for flood pixels
        if has_population:
            # Purple colormap for population density
            colors_density = ["#00000000", "#E8DAEF", "#BB8FCE", "#8E44AD", "#5B2C6F"]
        else:
            # Electric blue colormap for flood pixel density
            colors_density = ["#00000000", "#00BFFF", "#0080FF", "#0040FF"]

        cmap_density = LinearSegmentedColormap.from_list(
            "density", colors_density, N=100
        )

        # Plot density heatmap
        im_density = ax.imshow(
            heatmap_smooth,
            extent=[x_min_flood, x_max_flood, y_min_flood, y_max_flood],
            origin="lower",
            cmap=cmap_density,
            alpha=0.7,
            interpolation="bilinear",
            zorder=4,
        )

        # Add colorbar for density
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(im_density, cax=cax)

        if has_population:
            cbar.set_label(
                "Affected Population Density", rotation=270, labelpad=20, fontsize=10
            )
            print(f"Plotted affected population density heatmap")
        else:
            cbar.set_label(
                "Flood Point Density", rotation=270, labelpad=20, fontsize=10
            )
            print(f"Plotted flood pixel density heatmap")

    # Add boundaries
    gdf_aoi.boundary.plot(ax=ax, color="black", linewidth=1.5, alpha=0.8, zorder=5)
    gdf_admin1.boundary.plot(ax=ax, color="black", linewidth=0.8, alpha=0.6, zorder=5)

    # Set extent
    x_min, x_max = float(provenance_target.x.min()), float(provenance_target.x.max())
    y_min, y_max = float(provenance_target.y.min()), float(provenance_target.y.max())
    x_buffer = (x_max - x_min) * 0.02
    y_buffer = (y_max - y_min) * 0.02
    ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

    # Title
    ax.set_title(
        f"Data Provenance Map - {target_date}\n(Latest observation date per pixel)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5, zorder=0)

    # Legend
    legend_elements = [Patch(facecolor="#808080", label="No data")]
    for i, date in enumerate(unique_dates):
        legend_elements.append(
            Patch(facecolor=colors[i], label=str(pd.Timestamp(date))[:10])
        )

    ax.legend(
        handles=legend_elements,
        title="Latest Observation Date",
        loc="upper right",
        fontsize=10,
        title_fontsize=11,
        framealpha=0.95,
        edgecolor="black",
        fancybox=False,
    )

    plt.tight_layout()

    # Save with descriptive filename
    density_type = (
        "population"
        if (
            population_raster
            and len(flood_points) > 0
            and "population" in gpd.GeoDataFrame(flood_points).columns
        )
        else "flood"
    )
    output_filename = (
        f"{iso3}_{density_type}_provenance_{target_date.replace('-', '')}.png"
    )
    output_path = f"{output_dir}/{output_filename}"
    save_plot_to_blob(plt, output_path)
    # plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_path}")
    plt.close()

    # ========== GENERATE CHOROPLETH (if population data available) ==========
    if (
        population_raster
        and len(flood_points) > 0
        and "population" in gpd.GeoDataFrame(flood_points).columns
    ):
        print("\n" + "=" * 80)
        print("GENERATING CHOROPLETH MAP")
        print("=" * 80)

        try:
            # Get admin level from country config
            country_config = get_country_config(iso3)
            adm_level = country_config["choropleth_adm_level"]
            gdf_admin = load_fieldmaps_parquet(
                iso3, adm_level=adm_level, admin_source="blob"
            )
            print(f"Using ADM{adm_level} boundaries: {len(gdf_admin)} divisions")

            # Load ADM1 for overlay
            gdf_admin1_overlay = load_fieldmaps_parquet(
                iso3, adm_level=1, admin_source="blob"
            )
            print(f"Loading ADM1 for overlay: {len(gdf_admin1_overlay)} divisions")

            # Create GeoDataFrame from flood points
            gdf_floods = gpd.GeoDataFrame(flood_points, crs="EPSG:4326")

            # Calculate affected population by admin division
            gdf_choropleth = calculate_admin_population(
                gdf_floods, gdf_admin, adm_level
            )

            # Separate unassigned pseudo-divisions for reporting (but exclude from visualization)
            adm_name_col = f"adm{adm_level}_name"
            is_unassigned = gdf_choropleth[adm_name_col].str.contains(
                "Unassigned", na=False
            )
            gdf_unassigned = gdf_choropleth[is_unassigned].copy()
            gdf_choropleth_viz = gdf_choropleth[~is_unassigned].copy()

            # Print summary (including unassigned)
            affected_divs_viz = gdf_choropleth_viz[
                gdf_choropleth_viz["affected_pop"] > 0
            ]
            total_pop = gdf_choropleth["affected_pop"].sum()
            print(
                f"Divisions with affected population: {len(affected_divs_viz)}/{len(gdf_choropleth_viz)}"
            )
            if len(gdf_unassigned) > 0:
                unassigned_pop = gdf_unassigned["affected_pop"].sum()
                print(
                    f"  + {len(gdf_unassigned)} unassigned pseudo-divisions: {unassigned_pop:.0f} people"
                )
            print(f"Total affected population: {total_pop:,.0f}")

            # Create choropleth figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")

            # Find best available name column (fallback if adm_name is None)
            adm_name_col = f"adm{adm_level}_name"

            # Check if the primary name column has values, if not fallback to lower admin levels
            if gdf_choropleth_viz[adm_name_col].isna().all():
                print(
                    f"  Note: {adm_name_col} is empty, falling back to lower admin levels for labels"
                )
                for fallback_level in range(adm_level - 1, 0, -1):
                    fallback_col = f"adm{fallback_level}_name"
                    if (
                        fallback_col in gdf_choropleth_viz.columns
                        and not gdf_choropleth_viz[fallback_col].isna().all()
                    ):
                        print(f"  Using {fallback_col} for labels")
                        adm_name_col = fallback_col
                        break

            # Plot choropleth with improved color scaling
            # Use vmin=0 and vmax=max_pop to ensure full color range is used
            max_pop = gdf_choropleth_viz["affected_pop"].max()

            # For very small values, use a classification scheme instead of continuous
            if max_pop > 0 and max_pop < 100:
                # Use a categorical/binned approach for small values
                # Create bins: 0, 1-5, 5-10, 10-20, 20+
                bins = [0, 1, 5, 10, 20, max_pop + 1]
                norm = BoundaryNorm(bins, ncolors=256)

                # Create custom colormap with white for 0
                colors_list = [
                    "white",
                    "#ffffcc",
                    "#ffeda0",
                    "#fed976",
                    "#feb24c",
                    "#fd8d3c",
                    "#fc4e2a",
                    "#e31a1c",
                    "#bd0026",
                    "#800026",
                ]
                cmap_custom = LinearSegmentedColormap.from_list(
                    "white_ylorrd", colors_list, N=256
                )

                gdf_choropleth_viz.plot(
                    column="affected_pop",
                    ax=ax,
                    cmap=cmap_custom,
                    edgecolor="#c0c0c0",  # Happy medium - darker grey, visible but not distracting
                    linewidth=0.15,
                    legend=True,
                    norm=norm,
                    legend_kwds={
                        "label": "Affected Population",
                        "orientation": "vertical",
                        "shrink": 0.6,
                    },
                )
            else:
                # Use continuous scale with vmin=0 for larger populations
                # Create custom colormap with white for 0
                colors_list = [
                    "white",
                    "#ffffcc",
                    "#ffeda0",
                    "#fed976",
                    "#feb24c",
                    "#fd8d3c",
                    "#fc4e2a",
                    "#e31a1c",
                    "#bd0026",
                    "#800026",
                ]
                cmap_custom = LinearSegmentedColormap.from_list(
                    "white_ylorrd", colors_list, N=256
                )

                gdf_choropleth_viz.plot(
                    column="affected_pop",
                    ax=ax,
                    cmap=cmap_custom,
                    edgecolor="#c0c0c0",  # Happy medium - darker grey, visible but not distracting
                    linewidth=0.15,
                    legend=True,
                    vmin=0,
                    vmax=max_pop,
                    legend_kwds={
                        "label": "Affected Population",
                        "orientation": "vertical",
                        "shrink": 0.6,
                    },
                )

            # Add ADM3 boundaries colored by data provenance (using exactextract - blazingly fast!)
            print("  Extracting modal provenance for ADM3 boundaries (exactextract)...")

            # Map provenance index to colors: oldest=red, newest=green
            colors = generate_rdylgn_colors(len(unique_dates))
            date_to_color = {i: colors[i] for i in range(len(unique_dates))}

            # Write provenance raster to temp file for exactextract
            # exactextract needs a file path, not in-memory array
            # Get provenance as numpy array (handle both xarray and numpy)
            if isinstance(provenance_indexed, xr.DataArray):
                prov_data = provenance_indexed.values.astype("int16")
                x_min, x_max = float(provenance_indexed.x.min()), float(
                    provenance_indexed.x.max()
                )
                y_min, y_max = float(provenance_indexed.y.min()), float(
                    provenance_indexed.y.max()
                )
            else:
                # Already a numpy array (from cache)
                prov_data = provenance_indexed.astype("int16")
                # Get bounds from provenance_target
                x_min, x_max = float(provenance_target.x.min()), float(
                    provenance_target.x.max()
                )
                y_min, y_max = float(provenance_target.y.min()), float(
                    provenance_target.y.max()
                )

            # Create affine transform from coordinates
            height, width = prov_data.shape
            transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

            # Write to temporary file
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp_raster_path = tmp.name

            with rasterio.open(
                tmp_raster_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="int16",
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(prov_data, 1)

            # Use exactextract to get mode of provenance_indexed for each polygon
            # This is MUCH faster than rasterstats!
            modal_prov = exactextract.exact_extract(
                tmp_raster_path,
                gdf_admin,
                ["mode"],
                include_cols=[f"adm{adm_level}_id"],
                output="pandas",
            )

            # Clean up temp file
            os.unlink(tmp_raster_path)

            # Map results to colors
            prov_colors = []
            for _, row in modal_prov.iterrows():
                mode_val = row["mode"]
                if pd.isna(mode_val) or mode_val == -1:
                    prov_colors.append("lightgray")
                else:
                    prov_colors.append(date_to_color.get(int(mode_val), "lightgray"))

            gdf_admin["prov_color"] = prov_colors

            # Dissolve by provenance color to create one polygon per color group
            print("  Dissolving by provenance color...")
            dissolved_provenance = gdf_admin.dissolve(by="prov_color", as_index=False)

            # Create mapping from color back to date for legend
            color_to_date = {
                color: unique_dates[i] for i, color in date_to_color.items()
            }

            # Plot no-data areas with grey fill (70% opacity)
            no_data_polys = dissolved_provenance[
                dissolved_provenance["prov_color"] == "lightgray"
            ]
            if len(no_data_polys) > 0:
                no_data_polys.plot(
                    ax=ax, facecolor="lightgrey", edgecolor="none", alpha=0.7, zorder=1
                )

            # Plot dissolved provenance boundaries (thick colored lines)
            legend_elements = []

            for idx, row in dissolved_provenance.iterrows():
                color = row["prov_color"]
                if color != "lightgray":  # Plot colored provenance zones
                    gpd.GeoSeries([row.geometry]).boundary.plot(
                        ax=ax, edgecolor=color, linewidth=2.5, alpha=0.9, zorder=3
                    )

            # Build legend in chronological order (oldest to newest, top to bottom)
            # Sort dates and add legend entries in that order
            sorted_dates = sorted(unique_dates)
            unique_dates_list = list(
                unique_dates
            )  # Convert numpy array to list for .index()
            for date in sorted_dates:
                # Find which index this date corresponds to
                date_idx = unique_dates_list.index(date)
                color = date_to_color.get(date_idx)
                if color:
                    date_str = str(date)[:10]
                    legend_elements.append(
                        Patch(
                            facecolor="none",
                            edgecolor=color,
                            linewidth=2.5,
                            label=date_str,
                        )
                    )

            # Add no-data to legend
            if len(no_data_polys) > 0:
                legend_elements.append(
                    Patch(
                        facecolor="lightgrey",
                        alpha=0.7,
                        edgecolor="none",
                        label="No data",
                    )
                )

            # Plot all ADM3 boundaries in light grey (happy medium)
            gdf_admin.boundary.plot(
                ax=ax, edgecolor="#c0c0c0", linewidth=0.15, alpha=0.4, zorder=2
            )

            print(f"  Provenance boundaries (dissolved) and ADM3 borders plotted")

            # Add ADM1 boundaries on top (transparent fill, thicker lines)
            gdf_admin1_overlay.plot(
                ax=ax, facecolor="none", edgecolor="black", linewidth=1.5, alpha=0.8
            )

            # Add labels for top affected divisions (using adjustText to avoid overlaps)
            top_n = min(5, len(affected_divs_viz))
            if top_n > 0:
                top_affected = affected_divs_viz.nlargest(top_n, "affected_pop")
                texts = []
                for idx, row in top_affected.iterrows():
                    centroid = row.geometry.centroid
                    text = ax.text(
                        centroid.x,
                        centroid.y,
                        f"{row[adm_name_col]}\n({row['affected_pop']:.0f})",
                        fontsize=8,
                        weight="bold",
                        ha="center",
                        va="center",
                        color="black",
                        path_effects=[
                            withStroke(linewidth=3, foreground="white", alpha=0.7)
                        ],
                    )
                    texts.append(text)

                # Adjust text positions to avoid overlaps (like ggrepel!)
                adjust_text(
                    texts,
                    ax=ax,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5, alpha=0.6),
                )

            # Set axis limits to match provenance target extent (with 2% buffer)
            x_min, x_max = float(provenance_target.x.min()), float(
                provenance_target.x.max()
            )
            y_min, y_max = float(provenance_target.y.min()), float(
                provenance_target.y.max()
            )
            x_buffer = (x_max - x_min) * 0.02
            y_buffer = (y_max - y_min) * 0.02
            ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
            ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

            # Title and labels
            mode_label = "Cumulative" if flood_mode == "cumulative" else "Latest"
            ax.set_title(
                f"Affected Population by Admin{adm_level} Division ({mode_label} Mode) - {target_date}\n"
                f"Total: {total_pop:,.0f} people in {len(affected_divs_viz)} divisions",
                fontsize=14,
                fontweight="bold",
                pad=15,
            )
            # Remove axis labels and tick labels for cleaner map
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)
            ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5, zorder=0)

            # Add provenance legend
            if legend_elements:
                legend = ax.legend(
                    handles=legend_elements,
                    title="Data Provenance",
                    loc=legend_loc,  # From country config: JAM='lower left', HTI='upper left', CUB='lower right'
                    fontsize=9,
                    title_fontsize=10,
                    framealpha=0.85,  # Transparent white background (0=transparent, 1=opaque)
                    facecolor="white",
                    edgecolor="black",
                )

            plt.tight_layout()

            # Save choropleth
            mode_suffix = "cumulative" if flood_mode == "cumulative" else "latest"
            choropleth_filename = f"{iso3}_population_{mode_suffix}_adm{adm_level}_{target_date.replace('-', '')}.png"
            choropleth_path = f"{output_dir}/{choropleth_filename}"
            save_plot_to_blob(plt, choropleth_path)
            # plt.savefig(
            #     choropleth_path, dpi=150, bbox_inches="tight", facecolor="white"
            # )
            print(f"Saved: {choropleth_path}")
            plt.close()

        except Exception as e:
            print(f"WARNING: Could not generate choropleth: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate flood provenance + density maps using N most recent observations"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format (e.g., 2025-10-27 or 2024-07-15)",
    )
    parser.add_argument(
        "--n-latest",
        type=str,
        help="Number of most recent observations to use",
    )
    parser.add_argument(
        "--iso3", type=str, default="JAM", help="ISO3 country code (default: JAM)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Directory for caching processed data (default: data/cache)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Force recompute and bypass cache"
    )
    parser.add_argument(
        "--flood-mode",
        type=str,
        default="latest",
        choices=["latest", "cumulative"],
        help="'latest' = only use flood pixels from latest provenance (default), 'cumulative' = sum across all dates (conservative extent)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/plots",
        help="Directory for output files (default: outputs/plots)",
    )

    args = parser.parse_args()

    main(
        args.end_date,
        int(args.n_latest),
        args.iso3,
        args.cache_dir,
        use_cache=not args.no_cache,
        flood_mode=args.flood_mode,
        output_dir=args.output_dir,
    )
