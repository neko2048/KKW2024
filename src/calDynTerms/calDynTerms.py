import sys
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib import cm
sys.path.insert(0, "../")
import config
from util.vvmLoader import VVMLoader
from util.dataWriter import DataWriter
#from util.dataPloter import truncate_colormap
#from util.dataPloter import getCmapAndNorm
from util.calculator import getHrzLaplacian

def calBuoyancy(thData):
    th = np.array(thData["th"][0])
    qv = np.array(thData["qv"][0])
    qc = np.array(thData["qc"][0])
    qi = np.array(thData["qi"][0])
    qr = np.array(thData["qr"][0])
    thBar = np.tile(np.mean(th, axis=(1, 2), keepdims=True), reps=(1, len(yc), len(xc)))
    buoyancy = 9.81 * ((th - thBar) / thBar + 0.61 * qv - qc - qi - qr)
    buoyancy = buoyancy - np.tile(np.mean(buoyancy, axis=(1, 2), keepdims=True), reps=(1, len(yc), len(xc)))
    return buoyancy

def calLaplace(data, dx, dy):
    #return (np.roll(data, 1, axis=2) + np.roll(data, -1, axis=2) - 2 * data) / (dx ** 2) + \
    #       (np.roll(data, 1, axis=1) + np.roll(data, -1, axis=1) - 2 * data) / (dy ** 2)
    return getHrzLaplacian(data) / (dx**2)

def partialX(data, dx):
    return (np.roll(data, -1, axis=2) - np.roll(data, 1, axis=2)) / (dx*2)

def partialY(data, dy):
    return (np.roll(data, -1, axis=1) - np.roll(data, 1, axis=1)) / (dy*2)

def partialZ(data, zc):
    partial = np.zeros(shape=data.shape)
    partial[1:-1] = (data[2:] - data[:-2]) / (zc[2:] - zc[:-2])
    partial[0] = (data[1] - data[0]) / (zc[1] - zc[0])
    partial[-1] = (data[-1] - data[-2]) / (zc[-1] - zc[-2])
    return partial

def getDM03(u, v, w, rho, dx, dy, zc):
    termX = partialX(rho * u * w, dx)
    termY = partialY(rho * v * w, dy)
    termZ = partialZ(rho * w * w, zc)
    return - 1 / rho * (termX + termY + termZ)
    
def getDM05(u, v, dx, dy, zc):
    return partialZ(calLaplace((u**2 + v**2) / 2, dx, dy), zc)

def getDM06(u, v, w, dx, dy, zc):
    termU = partialX(w * partialZ(u, zc), dx)
    termV = partialY(w * partialZ(v, zc), dy)
    return partialZ(termU + termV, zc)

def getDM07(u, v, zeta, dx, dy, zc):
    zetaTerm = v * partialX(zeta, dx) - u * partialY(zeta, dy) + zeta ** 2
    return - partialZ(zetaTerm, zc)

if __name__ == "__main__":
    iniTimeIdx, endTimeIdx = int(sys.argv[1]), int(sys.argv[2])
    caseName = "mjo_std_mg"
    vvmLoader = VVMLoader(dataDir=f"{config.vvmPath}/", subName=caseName)
    rho = vvmLoader.loadRHO()[:-1][:, np.newaxis, np.newaxis]
    thData = vvmLoader.loadThermoDynamic(0)
    xc, yc, zc = np.array(thData["xc"]), np.array(thData["yc"]), np.array(thData["zc"])
    zc3D = zc[:, np.newaxis, np.newaxis]
    dx, dy = xc[1] - xc[0], yc[1] - yc[0]
    timeArange = np.arange(iniTimeIdx, endTimeIdx)
    #scale = 1e6
    #levels = [round(x, 1) for x in np.arange(0, 2.1, 0.1)]
    #levels.extend([-x for x in levels])
    #levels = np.sort(np.unique(levels))
    #dynCmap, dynNorm = getCmapAndNorm("RdBu_r", levels, extend="both")
    #cmap = truncate_colormap(plt.get_cmap("Greys"), minval=0.25, maxval=1, n=5)
    dataWriter = DataWriter(outputPath=f"./")
    #dataWriter = DataWriter(outputPath=f"{config.dynTermPath}")
    

    for tIdx in timeArange:
        print(f"========== {tIdx:06d} ==========")
        thData = vvmLoader.loadThermoDynamic(tIdx)
        dm01 = calBuoyancy(thData)
        #qc = np.array(thData["qc"][0])
        #qr = np.array(thData["qr"][0])
        #qi = np.array(thData["qi"][0])
        dm02 = calLaplace(dm01, dx, dy)
        dyData = vvmLoader.loadDynamic(tIdx)
        u = np.array(dyData["u"][0])
        u = (np.roll(u, 1, axis=2) + u) / 2
        v = np.array(dyData["v"][0])
        v = (np.roll(v, 1, axis=1) + v) / 2
        w = np.array(dyData["w"][0])
        w = (np.roll(w, 1, axis=0) + w) / 2
        zeta = np.array(dyData["zeta"][0])
        zeta = (np.roll(zeta, [1, 1], axis=(1, 2)) + np.roll(zeta, [1], axis=(2)) + np.roll(zeta, [1], axis=(1)) + zeta) / 4

        dm03 = getDM03(u, v, w, rho, dx, dy, zc3D)
        dm04 = calLaplace(dm03, dx, dy)
        dm05 = getDM05(u, v, dx, dy, zc3D)
        dm06 = getDM06(u, v, w, dx, dy, zc3D)
        dm07 = getDM07(u, v, zeta, dx, dy, zc3D)

        dyTerms = np.array([dm01, dm02, dm03, dm04, dm05, dm06, dm07])
        dyTermsTitle = ["dm01", "dm02", "dm03", "dm04", "dm05", "dm06", "dm07"]

        dataWriter.toNC(fname = f"diag-{tIdx:06d}.nc", 
                        data = dict(
                                    dm01 = (["time", "zc", "yc", "xc"], dm01[np.newaxis, :, :, :]),
                                    dm02 = (["time", "zc", "yc", "xc"], dm02[np.newaxis, :, :, :]),  
                                    dm03 = (["time", "zc", "yc", "xc"], dm03[np.newaxis, :, :, :]),  
                                    dm04 = (["time", "zc", "yc", "xc"], dm04[np.newaxis, :, :, :]),  
                                    dm05 = (["time", "zc", "yc", "xc"], dm05[np.newaxis, :, :, :]),  
                                    dm06 = (["time", "zc", "yc", "xc"], dm06[np.newaxis, :, :, :]),  
                                    dm07 = (["time", "zc", "yc", "xc"], dm07[np.newaxis, :, :, :])),
                        coords = {'time': np.ones(shape=(1,)), 'zc': zc, 'yc': yc, 'xc': xc}, 
                        format="NETCDF3_64BIT"
                        )

"""
        fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 8))
        plt.sca(axs[0, 0])
        plt.pcolormesh(xc / 1e3, zc/1e3, np.ma.masked_array(qc, qc<=0), cmap=cmap, shading="nearest", vmin=0, vmax=0.005)
        plt.contour(xc/1e3, zc/1e3, qi>=1e-4, levels=[0.9], colors='#008000', linewidths=1.5)
        C = plt.contourf(xc/1e3, zc/1e3, np.ma.masked_array(qr, qr<=1e-4), colors="#00FFFF", alpha=0.3)
        C = plt.contourf(xc/1e3, zc/1e3, np.ma.masked_array(qr, qr<=5e-3), hatches=["||"], colors="None")

        for i in range(1, len(dyTerms)+1):
            plt.sca(axs[i//3, i%3])
            im = plt.pcolormesh(xc / 1e3, zc/1e3, scale*dyTerms[i-1], shading="nearest", cmap=dynCmap, norm=dynNorm)
            plt.contour(xc/1e3, zc/1e3, qc>0, levels=[0.9], colors='black')
            print(dyTerms[i-1].min(), dyTerms[i-1].max())

        for i in range(6):
            plt.sca(axs[i//3, i%3])
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.ylim(0, 12.5)
            plt.xlim(60, 102.4)
            if i >= 1:
                plt.title(dyTermsTitle[i-1], fontsize=15)

        plt.subplots_adjust(left=0.05, right=0.975, bottom=0.15, wspace=0.1)
        plt.suptitle(fr"$10^{int(np.log10(scale))}\times$ Laplace Terms | {caseName} | {tIdx * config.minPerTimeIdx//60:03d} hr {tIdx * config.minPerTimeIdx % 60} min", fontsize=20, y=0.975)
        cbar_ax = fig.add_axes([0.05, 0.06, 0.925, 0.025])
        cb = fig.colorbar(im, cax=cbar_ax, extend="both", orientation='horizontal')#, format='%.0e')
        cb.ax.tick_params(labelsize=10)#5)
        cb.set_ticks(levels, [str(x) for x in levels])
        plt.savefig(f"{caseName}-{tIdx:06d}.jpg", dpi=300)
"""

