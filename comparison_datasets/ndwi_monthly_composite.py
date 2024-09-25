import ee
from utils import get_aoi

# Initialize Earth Engine
ee.Initialize()

# uncomment if wanting to process all sites
#aoi_list = get_aoi.get_all_aois()

# add name in for only one site
aoi_list = get_aoi.get_area_of_interest('Aloppkolen')

# Define the years for which you want to obtain composites.
startYear = 2020
endYear = 2023

# Loop through each AOI.
for aoi_name, roi in aoi_list.items():  # Iterate through dictionary items.

    # Create an empty image collection to store monthly NDWI images
    ndwiMaskCollection = ee.ImageCollection([])

    # Loop through each year
    for year in range(startYear, endYear + 1):
        # Loop through each month
        for month in range(1, 13):
            # Define the start and end dates for the current month and year.
            startDate = ee.Date.fromYMD(year, month, 1)
            endDate = ee.Date.fromYMD(year, month, 28)  # Adjust end date as needed.

            # Filter the Sentinel-2 data by date, cloud cover, and location.
            collection = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                          .filterBounds(roi)
                          .filterDate(startDate, endDate)
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)))  # Adjust cloud cover threshold as needed.

            # Check if there are composites available for the current month.
            compositeCount = collection.size().getInfo()
            if compositeCount == 0:
                print('No composites available for AOI', aoi_name, 'Year', year, 'Month', month)
                continue  # Skip this month and move to the next.

            # Create a composite by reducing the collection to the median value.
            composite = collection.median()

            # Calculate NDWI for the composite image.
            ndwi = composite.normalizedDifference(['B3', 'B8'])

            # Define a threshold for NDWI to identify open water.
            ndwiThreshold = 0  # You can adjust this threshold as needed.

            # Create a binary mask where 1 represents open water and 0 represents background.
            waterMask = ndwi.gt(ndwiThreshold)

            # Clip the water mask to the ROI (polygon).
            waterMaskClipped = waterMask.clip(roi)

            # Add the water mask to the collection only if it has bands.
            if waterMask.bandNames().size().getInfo() > 0:
                ndwiMaskCollection = ndwiMaskCollection.merge(ee.ImageCollection([waterMaskClipped.set('year', year, 'month', month)]))

            # Export the water mask to Google Drive for the current month and year.
            exportWaterMask = waterMaskClipped.unmask(0).byte()  # Convert to byte type
            task = ee.batch.Export.image.toDrive(exportWaterMask,
                description= aoi_name + '_' + str(year) + '_' + str(month),  # Name for the exported file
                scale=10,
                folder='ndwi_mask_' + aoi_name,  # Use the AOI name for the folder
                region=roi.getInfo()['coordinates'],
                maxPixels=1e10
            )
            task.start()
