import numpy as np
import os
import xarray as xr

class DataWriter:
    def __init__(self, outputPath):
        self.outputPath = outputPath        
        self.checkExistOrCreate(self.outputPath)

    def checkExistOrCreate(self, outputPath):
        if not os.path.exists(outputPath): 
            print("Path is not exist, created")
            os.makedirs(outputPath)
        else:
            print("Path exists")

    def toNC(self, fname, data, coords, varName=None, dims=None, format=None):
        if dims != None and varName != None:
            xrData = xr.DataArray(data,
                                  coords = coords,
                                  dims = dims,
                                  name = varName)
        else:
            xrData = xr.Dataset(data,
                                coords = coords)
        if format == None:
            xrData.to_netcdf(self.outputPath + fname)
        else:
            xrData.to_netcdf(self.outputPath + fname, format=format)
    
    def toNPY(self, fname, data):
        np.save(self.outputPath + fname, data)
    
    #def toNC(self, fname, data, coords, dims, varName, format=None):
    #    if format == None:
    #        xrData = xr.DataArray(data, 
    #                              coords = coords,
    #                              dims = dims,
    #                              name = varName)
    #        xrData.to_netcdf(self.outputPath + fname)
    #    else:
    #        xrData = xr.DataArray(data,
    #                              coords = coords,
    #                              dims = dims,
    #                              name = varName)
    #        xrData.to_netcdf(self.outputPath + fname, format=format)

    #def toNPY(self, fname, data):
    #    np.save(self.outputPath + fname, data)
