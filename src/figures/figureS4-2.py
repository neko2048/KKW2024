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
    initTimeStep, endTimeStep = 460, 461#int(sys.argv[1]), int(sys.argv[2])
    #coarseReslList =  np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0])
    #realSpaceLength = np.array([0.1, 0.4, 0.6, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.6, 3.6,  4.8,  7.2,  9.6])
    #coarseReslList =  np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0])
    coarseReslList =  np.array([0.0, 1.5, 4.0, 5.0, 7.5, 10.0, 20.0])#, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0])
    #coarseReslList =  np.array([0.0, 4.0])#, 5.0, 7.5, 20.0])
    realSpaceLength = np.array([0.1] + [round(0.6*x, 1) for x in coarseReslList[1:]])
    
    timeArange = np.arange(initTimeStep, endTimeStep)
    vvmLoader = VVMLoader(dataDir=config.vvmPath, subName="mjo_std_mg")
    thData = vvmLoader.loadThermoDynamic(0)
    xc, yc, zc = np.array(thData["xc"]), np.array(thData["yc"]), np.array(thData["zc"])
    rho = vvmLoader.loadRHO()[:-1]
    zz = vvmLoader.loadZZ()
    xMin, xMax = np.argmin(np.abs(xc-73450)), np.argmin(np.abs(xc-74250))+1
    #xMin, xMax = np.argmin(np.abs(xc-88000)), np.argmin(np.abs(xc-94000))+1
    #xMin2, xMax2 = np.argmin(np.abs(xc-87500)), np.argmin(np.abs(xc-92500))+1
    yMin, yMax = np.argmin(np.abs(yc-24250)), np.argmin(np.abs(yc-24950))+1
    #yMin, yMax = np.argmin(np.abs(yc-29000)), np.argmin(np.abs(yc-35000))+1
    #yMin2, yMax2 = np.argmin(np.abs(yc-27500)), np.argmin(np.abs(yc-32500))+1
    colorbarName = "nipy_spectral"
    rbCmap = plt.get_cmap(colorbarName)

    for tIdx in timeArange:
        print(f"========== {tIdx:06d} ==========")
        thData = vvmLoader.loadThermoDynamic(tIdx)
        numRow, numCol = 3, 1
        fig, ax = plt.subplots(numRow, numCol, figsize=(8, 20))
        
        fs = 35
        lw=4
        for stdIdx, std in enumerate(coarseReslList):
            print(std)
            # Buoyancy
            print("Buoyancy")
            plt.sca(ax[0])
            plt.title("(d) " + r"$\rho_0 \overline{a(\widetilde{B}})$ [$\times 10^{-3}  kg\cdot m^{-2}s^{-2}$]", fontsize=fs, y=1.015)
            if (coarseReslList[stdIdx] != 0.0):
                buoy = 1e3*np.mean(np.roll(nc.Dataset(config.rebuildDynPath + f"gaussian-{std:.1f}/a-{tIdx:06d}.nc")["a"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax], axis=(1, 2))
                plt.plot(rho*buoy, zc/1e3, c=rbCmap((stdIdx+1)/(len(coarseReslList)+1)), linewidth=lw)
            else:
                buoy = 1e3*np.mean(np.roll(nc.Dataset(config.rebuildDynPath + f"uniform-1/a-{tIdx:06d}.nc")["a"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax], axis=(1, 2))
                plt.plot(rho*buoy, zc/1e3, c='black', linewidth=lw, zorder=3)
                #plt.plot(buoy, zc/1e3, c='black', linewidth=lw)
            plt.xticks([-3, -2, -1, 0, 1, 2, 3])
            #plt.legend(fontsize=fs/3*2)
            plt.xlim(-2.5, 2.7)

            # DM03
            print("DM03")
            plt.sca(ax[1])
            plt.title("(h) " + r"$\rho_0 \overline{a(\widetilde{D_V}})$ [$\times 10^{-3}  kg\cdot m^{-2}s^{-2}$]", fontsize=fs, y=1.015)
            if (coarseReslList[stdIdx] != 0.0):
                dm03 = 1e3*np.mean(np.roll(nc.Dataset(config.dynTermDirectAccePath + f"gaussian-{std:.1f}/a-{tIdx:06d}.nc")["dm04"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax], axis=(1, 2))
                plt.plot(rho*dm03, zc/1e3, c=rbCmap((stdIdx+1)/(len(coarseReslList)+1)), linewidth=lw, label=f"{realSpaceLength[stdIdx]}km")
            else:
                dm03 = 1e3*np.mean(np.roll(nc.Dataset(config.dynTermDirectAccePath + f"uniform-1/a-{tIdx:06d}.nc")["dm04"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax], axis=(1, 2))
                plt.plot(rho*dm03, zc/1e3, c='black', linewidth=lw, zorder=3, label=f"100m")
            plt.legend(fontsize=fs/3*2, loc='upper right')
            plt.xticks([-1, 0, 1, 2, 3])
            plt.xlim(-1.5, 3.4)


            # DM05 06 07
            print("RES")
            plt.sca(ax[2])
            plt.title("(l) " + r"$\rho_0 \overline{a(\widetilde{D_H}})$ [$\times 10^{-3}  kg\cdot m^{-2}s^{-2}$]", fontsize=fs, y=1.015)
            if (coarseReslList[stdIdx] != 0.0):
                res  = np.roll(nc.Dataset(config.dynTermDirectAccePath + f"gaussian-{std:.1f}/a-{tIdx:06d}.nc")["dm05"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax]
                res += np.roll(nc.Dataset(config.dynTermDirectAccePath + f"gaussian-{std:.1f}/a-{tIdx:06d}.nc")["dm06"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax]
                res += np.roll(nc.Dataset(config.dynTermDirectAccePath + f"gaussian-{std:.1f}/a-{tIdx:06d}.nc")["dm07"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax]
                res /= 3
                plt.plot(rho*1e3*np.mean(res, axis=(1, 2)), zc/1e3, c=rbCmap((stdIdx+1)/(len(coarseReslList)+1)), linewidth=lw)
            else:
                
                res =  np.roll(nc.Dataset(config.dynTermDirectAccePath + f"uniform-1/a-{tIdx:06d}.nc")["dm05"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax]
                res += np.roll(nc.Dataset(config.dynTermDirectAccePath + f"uniform-1/a-{tIdx:06d}.nc")["dm06"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax]
                res += np.roll(nc.Dataset(config.dynTermDirectAccePath + f"uniform-1/a-{tIdx:06d}.nc")["dm07"][0], 0, axis=-1)[:, yMin:yMax, xMin:xMax]
                res /= 3
                plt.plot(rho*1e3*np.mean(res, axis=(1, 2)), zc/1e3, c='black', linewidth=lw, zorder=3)
            plt.xticks([-0.2, -0.1, 0, 0.1, 0.2])
            plt.xlim(-0.21, 0.25)

        for j in range(3):
            plt.sca(ax[j])
            plt.grid(True)
            #plt.ylabel("z [km]", fontsize=30)
            plt.xticks(fontsize=fs)
            plt.yticks(np.arange(0, 15, 2), fontsize=fs)
            plt.ylim(0, 8.5)
        
        plt.subplots_adjust(left=0.1, right=0.92, top=0.95, bottom=0.075, hspace=0.3, wspace=0.025)
        plt.savefig(f"./figureS4-2.png", dpi=300)
        plt.clf()


