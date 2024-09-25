import os
import rasterio.features
from shapely.geometry import shape, MultiPolygon
import geopandas as gpd
import csv

INPUT_FOLDER = r'../data/dynamic_world//' # Change this to the folder containing the raster files

wetland_names = [
    "Aloppkolen", "Annsjon", "Askoviken", "Asnen",
    "Dattern", "Dummemosse", "Eman", "Falsterbo", "Farnebofjarden", "Fyllean",
    "Gammelstadsviken", "Getapulien", "Getteron", "Gotlands", "Gullhog",
    "Gustavmurane", "Helge", "Hjalstaviken", "Hornborgasjon", "Hovramomradet",
    "Kallgate", "Kilsviken", "Klingavalsan", "Komosse", "Koppangen", "Kvismaren",
    "Laidaure", "Lundakrabukten", "Maanavuoma", "Mellanljusnan", "Mellerston",
    "Morrumsan", "Mossatrask", "Nittalven", "Nordrealvs", "Olands", "Oldflan",
    "Oset", "Osten", "Ottenby", "Ovresulan", "Paivavouma", "Persofjarden",
    "Pirttimysvuoma", "Rappomyran", "Sikavagarna", "Skalderviken",
    "Stigfjorden", "Storemosse", "Storkolen", "Svartadalen",
    "Svenskundsviken", "Takern", "Tarnsjon", "Tavvovouma", "Tjalmejaure",
    "Tonnersjoheden", "Traslovslage", "Tysoarna", "Vasikkavouma",
    "Vattenan"
]

for wetland_name in wetland_names:

    # Folder containing the input raster files
    input_folder = INPUT_FOLDER + wetland_name

    print(input_folder)

    # Specify the CRS for reprojecting to UTM (replace 'EPSG:32633' with the appropriate UTM zone)
    utm_crs = "EPSG:32633"

    # Create an empty list to store the results
    results = []

    # Iterate over files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):  # Assuming the raster files have the .tif extension
            raster_path = os.path.join(input_folder, filename)

            # Use a context manager to open the raster file
            with rasterio.open(raster_path) as src:
                # Read the first band
                binary_array = src.read(1)

                # Make sure you read the transform inside the context manager
                transform = src.transform

            # Get shapes (polygons)
            shapes = list(rasterio.features.shapes(binary_array, transform=transform))

            # Extract polygons with values 0 and 3 using a list comprehension
            polygons = [shape(geom) for geom, value in shapes if value in [0, 3]]

            # Dissolve the polygons into a single polygon and buffer it by 0 (no buffer)
            dissolved_polygon = MultiPolygon(polygons).buffer(0)

            # Create a GeoDataFrame with the dissolved polygon
            dissolved_gdf = gpd.GeoDataFrame(geometry=[dissolved_polygon])

            # Set the CRS of the GeoDataFrame to WGS84 (EPSG:4326)
            dissolved_gdf.crs = "EPSG:4326"

            # Reproject the GeoDataFrame to UTM (or your desired projected CRS)
            dissolved_gdf = dissolved_gdf.to_crs(utm_crs)

            # skip if no such file or directory
            if not os.path.exists(r'../wetland_shps/' + wetland_name + '.shp'):
                print(f"Skipping processing for {wetland_name} as no wetland boundary exists.")
                continue

            # clip to wetland boundary
            wetland_boundary = gpd.read_file(r'../wetland_shps/' + wetland_name + '.shp')
            wetland_boundary = wetland_boundary.to_crs(utm_crs)
            dissolved_gdf = gpd.clip(dissolved_gdf, wetland_boundary)

            # Calculate the area in square meters (m²) in the projected CRS
            if not dissolved_gdf.empty:
                area_utm = dissolved_gdf.geometry.area.iloc[0]
            else:
                area_utm = 0
                print(f"No valid geometries for {filename}. Setting area to 0.")

            # Test save as shapefile
            #dissolved_gdf.to_file(f'test_shps\dw\{filename}.shp')

            # Optionally, you can print the area for each raster
            print(f"{filename} Area (UTM): {area_utm} m²")

            # Append the results to the list
            results.append((filename, area_utm))


    output_csv = r'../data/dyn_w_' + wetland_name + '.csv'
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Area (metres squared)"])
        for filename, area in results:
            writer.writerow([filename, area])
