import os
from osgeo import gdal

def process_tile_file(tile_file_path): # Open the file with gdal and update the NoData value
    raster = gdal.Open(tile_file_path, gdal.GA_Update)
    band = raster.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    raster.FlushCache()
    del band
    del raster

def download_tile(download_url, destination_path, session): # If the file doesn't exist yet, download it
    if (os.path.isfile(destination_path)):
        return f"File {destination_path} already exists"
    
    flightRequest = session.request('get', download_url) # Make a flight request to get a session cookie
    contentRequest = session.get(flightRequest.url)
    if contentRequest.ok:
        file = open(destination_path, 'wb')
        file.write(contentRequest.content)
        file.close()
        process_tile_file(destination_path)
        return f"Downloaded {destination_path}"
    else:
        return f"Error downloading {download_url}: {contentRequest.status_code}, {contentRequest.text}"
