import sys
import os
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
sys.path.insert(0, "../")
import config
from util.vvmLoader import VVMLoader
from util.dataWriter import DataWriter
from util.calculator import *


if __name__ == "__main__":
    iniTimeIdx, endTimeIdx, std = int(sys.argv[1]), int(sys.argv[2]), round(float(sys.argv[3]), 1)
    timeArange = np.arange(iniTimeIdx, endTimeIdx, 1)
    convolMethod = "fft"#chooseConvolMethod(std)

    vvmLoader = VVMLoader(dataDir=config.vvmPath, subName="mjo_std_mg")
    thData = vvmLoader.loadThermoDynamic(0)
    xc, yc, zc = np.array(thData["xc"]), np.array(thData["yc"]), np.array(thData["zc"])

    #outputPath = config.convolveMseAnomalyPath + f"gaussian-{std}/"
    outputPath = config.convolveWPath + f"gaussian-{std}/"
    dataWriter = DataWriter(outputPath=outputPath)

    for tIdx in timeArange:
        print(f"========== {tIdx:06d} ==========")
        dyData = vvmLoader.loadDynamic(tIdx)
        w = np.array(dyData["w"][0])
        wInThetaGrid = np.roll((w + np.roll(w, -1, axis=0)) / 2, 1, axis=0)
        #mseAnomaly = np.array(nc.Dataset(config.mseAnomalyPath + f"mse-{tIdx:06d}.nc")["mse"][0])
        gaussianWeight = getGaussianWeight(len(yc), std=std)
        convMseAnomaly = getGaussianConvolve(wInThetaGrid, gaussianWeight, method=convolMethod)
        convMseAnomaly = convMseAnomaly[np.newaxis, :, :, :]
        dataWriter.toNC(fname=f"w-{tIdx:06d}.nc",
                        data=convMseAnomaly, 
                        coords = {"time": np.array([1]), "zc": zc, "yc": yc, "xc": xc}, 
                        dims = ["time", "zc", "yc", "xc"], 
                        varName = "w")


