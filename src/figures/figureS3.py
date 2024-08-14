import sys
import numpy as np
import netCDF4 as nc
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
sys.path.insert(0, "../")
import config
from util.vvmLoader import VVMLoader

def getIntensiveData(x, y):
    f = interpolate.interp1d(x, y)
    newX = np.linspace(x.min(), x.max(), 10000)
    return newX, f(newX)

if __name__ == "__main__":
    initTimeStep, endTimeStep = 460, 461 #int(sys.argv[1]), int(sys.argv[2])
    coarseReslList =  np.array([0.0])#, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0])
    realSpaceLength = np.array([0.1])#, 0.4, 0.6, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.6, 3.6,  4.8,  7.2,  9.6])

    timeArange = np.arange(initTimeStep, endTimeStep)
    vvmLoader = VVMLoader(dataDir=config.vvmPath, subName="mjo_std_mg")
    thData = vvmLoader.loadThermoDynamic(0)
    xc, yc, zc = np.array(thData["xc"]), np.array(thData["yc"]), np.array(thData["zc"])
    rho = vvmLoader.loadRHO()[:-1]
    zz = vvmLoader.loadZZ()
    #rho = vvmLoader.loadRHO()
    xMin, xMax = np.argmin(np.abs(xc-87500)), np.argmin(np.abs(xc-92500))+1
    #xMin, xMax = np.argmin(np.abs(xc-88000)), np.argmin(np.abs(xc-94000))+1
    #xMin2, xMax2 = np.argmin(np.abs(xc-87500)), np.argmin(np.abs(xc-92500))+1
    yMin, yMax = np.argmin(np.abs(yc-27500)), np.argmin(np.abs(yc-32500))+1
    #yMin, yMax = np.argmin(np.abs(yc-29000)), np.argmin(np.abs(yc-35000))+1
    #yMin2, yMax2 = np.argmin(np.abs(yc-27500)), np.argmin(np.abs(yc-32500))+1

    rbCmap = plt.get_cmap("jet")

    for tIdx in timeArange:
        print(f"========== {tIdx:06d} ==========")
        thData = vvmLoader.loadThermoDynamic(tIdx)
        numRow, numCol = 1, 1
        fig, ax = plt.subplots(numRow, numCol, figsize=(10, 10))
        #ax[0, 0].set_ylabel("Height [km]", fontsize=30)
        print("Buoyancy")
        buoyAcce = np.roll(nc.Dataset(config.rebuildDynPath + f"uniform-1/a-{tIdx:06d}.nc")["a"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax]
        np.mean(np.roll(nc.Dataset(config.rebuildDynPath + f"uniform-1/a-{tIdx:06d}.nc")["a"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax], axis=(1, 2))
        print("DM03")
        dm03Acce = np.roll(nc.Dataset(config.dynTermAccePath + f"uniform-1/a-{tIdx:06d}.nc")["dm04"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax]
        print("DM0567")
        resAcce =  np.roll(nc.Dataset(config.dynTermAccePath + f"uniform-1/a-{tIdx:06d}.nc")["dm05"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax]
        resAcce += np.roll(nc.Dataset(config.dynTermAccePath + f"uniform-1/a-{tIdx:06d}.nc")["dm06"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax]
        resAcce += np.roll(nc.Dataset(config.dynTermAccePath + f"uniform-1/a-{tIdx:06d}.nc")["dm07"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax]
        resAcce = resAcce / 3

        fs = 30
        lw=6
        plt.sca(ax)
        plt.title(r"Tendency Contributions", fontsize=fs, y=1.015)
        plt.plot(1e3*rho * np.mean(buoyAcce, axis=(1, 2)), zc/1e3, c='red', linewidth=lw, label=r"$\rho_0 \overline{a(B)}$")
        plt.plot(1e3*rho * np.mean(dm03Acce, axis=(1, 2)), zc/1e3, c='#4DBEEE', linewidth=lw, label=r"$\rho_0 \overline{a(D_V)}$")
        plt.plot(1e3*rho * np.mean(resAcce, axis=(1, 2)), zc/1e3, c='#77AC30', linewidth=lw, label=r"$\rho_0 \overline{a(D_H)}$")
        plt.plot(1e3*rho * np.mean(resAcce+buoyAcce+dm03Acce, axis=(1, 2)), zc/1e3, c='black', linewidth=lw, label=r"$\rho_0 \overline{a}$")
        plt.xticks([int(x) for x in np.linspace(-8, 8, 9)])
        plt.legend(fontsize=25)


        plt.sca(ax)
        plt.grid(True)
        plt.ylabel("z [km]", fontsize=30)
        plt.xlabel(r"$\rho_0 \overline{a}$" + r" $[\times 10^{-3} kg\cdot m^{-2} s^{-2}]$", fontsize=30)
        plt.ylim(0, 15)
        plt.xticks(fontsize=30)
        plt.yticks(np.arange(0, 15, 2), fontsize=30)
        
        #plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.95, wspace=0.15, hspace=0.15)
        plt.savefig(f"./figureS3.png", dpi=300)
        plt.clf()


