import sys
import numpy as np
import netCDF4 as nc
import json
import matplotlib.pyplot as plt
from matplotlib import cm
sys.path.insert(0, "../")
import config
from util.vvmLoader import VVMLoader
from util.dataPloter import getCmapAndNorm


if __name__ == "__main__":
    coarseReslList =  [0.0, 1.5, 4.0]
    realSpaceLength = [0] + [x*6*0.1 for x in coarseReslList]
    convDict = json.load(open("../deepConvection.json"))
    vvmLoader = VVMLoader(dataDir=config.vvmPath, subName="mjo_std_mg")
    thData = vvmLoader.loadThermoDynamic(0)
    xc, yc, zc = np.array(thData["xc"]), np.array(thData["yc"]), np.array(thData["zc"])
    rho = np.tile(vvmLoader.loadRHO()[:-1][:, np.newaxis], reps=(1, len(xc)))
    zz = vvmLoader.loadZZ()
    cmap, norm = getCmapAndNorm("RdBu_r", np.linspace(-0.1, 0.1, 21), "both")
    cmap2, norm2 = getCmapAndNorm("Spectral_r", np.linspace(-0.04, 0.04, 17), "both")

    fs = 35

    titles = [
    r"(a) $\rho_0 a(B)$ [$kg\cdot m^{-2}s^{-2}$]",   r"(b) $\rho_0 a(\widetilde{B})\ s=0.9km $ [$kg\cdot m^{-2}s^{-2}$]",   r"(c) $\rho_0 a(\widetilde{B})\ s=2.4km $ [$kg\cdot m^{-2}s^{-2}$]",
    r"(e) $\rho_0 a(D_V)$ [$kg\cdot m^{-2}s^{-2}$]", r"(f) $\rho_0 a(\widetilde{D_V})\ s=0.9km $ [$kg\cdot m^{-2}s^{-2}$]", r"(g) $\rho_0 a(\widetilde{D_V})\ s=2.4km $ [$kg\cdot m^{-2}s^{-2}$]",
    r"(i) $\rho_0 a(D_H)$ [$kg\cdot m^{-2}s^{-2}$]", r"(j) $\rho_0 a(\widetilde{D_H})\ s=0.9km $ [$kg\cdot m^{-2}s^{-2}$]", r"(k) $\rho_0 a(\widetilde{D_H})\ s=2.4km $ [$kg\cdot m^{-2}s^{-2}$]",
    ]

    for convObj in convDict:
        tIdx = convObj["timestep"]
        if tIdx not in [400]: continue
        print(f"========== {tIdx:06d} ==========")
        thData = vvmLoader.loadThermoDynamic(tIdx)
        for prof in range(len(convObj["xAxis"])):
            
            section = convObj["xAxis"][prof]
            qc = np.roll(thData["qc"][0][:, :, np.argmin(np.abs(xc-section))], -30, axis=-1)
            qi = np.roll(thData["qi"][0][:, :, np.argmin(np.abs(xc-section))], -30, axis=-1)
            buoyOrigin = np.roll(nc.Dataset(config.rebuildDynPath + f"uniform-1/a-{tIdx:06d}.nc")["a"][0][:, :, np.argmin(np.abs(xc-section))], -30, axis=-1)
            dgData = nc.Dataset(f"{config.dynTermDirectAccePath}uniform-1/a-{tIdx:06d}.nc")
            dm03Origin = np.roll(dgData["dm04"][0][:, :, np.argmin(np.abs(xc-section))], -30, axis=-1)
            resOrigin = np.roll(dgData["dm05"][0][:, :, np.argmin(np.abs(xc-section))], -30, axis=-1)
            resOrigin += np.roll(dgData["dm06"][0][:, :, np.argmin(np.abs(xc-section))], -30, axis=-1)
            resOrigin += np.roll(dgData["dm07"][0][:, :, np.argmin(np.abs(xc-section))], -30, axis=-1)

            numRow, numCol = 3, 3#int((len(coarseReslList)+1)/2)
            fig, ax = plt.subplots(numRow, numCol, figsize=(30, 20), sharey=True, gridspec_kw={'width_ratios': [1, 1, 1]})
            for i in range(numRow):
                ax[i, 0].set_ylabel("z [km]", fontsize=fs)
            for i in range(numCol):
                ax[2, i].set_xlabel("x [km]", fontsize=fs)
            for i in range(len(coarseReslList)):
                print(i)
                if coarseReslList[i] != 0.0:
                    coarseBuoy = np.roll(nc.Dataset(config.rebuildDynPath + f"gaussian-{coarseReslList[i]}/a-{tIdx:06d}.nc")["a"][0][:, :, np.argmin(np.abs(xc-section))], -30, axis=-1)
                    dgData = nc.Dataset(f"{config.dynTermDirectAccePath}gaussian-{coarseReslList[i]}/a-{tIdx:06d}.nc")
                    coarseDm03 = np.roll(dgData["dm04"][0][:, :, np.argmin(np.abs(xc-section))], -30, axis=-1)
                    coarseRes =  np.roll(dgData["dm05"][0][:, :, np.argmin(np.abs(xc-section))], -30, axis=-1)
                    coarseRes += np.roll(dgData["dm06"][0][:, :, np.argmin(np.abs(xc-section))], -30, axis=-1)
                    coarseRes += np.roll(dgData["dm07"][0][:, :, np.argmin(np.abs(xc-section))], -30, axis=-1)
                else:
                    coarseBuoy = buoyOrigin
                    coarseDm03 = dm03Origin
                    coarseRes = resOrigin

                plt.sca(ax[0, i % numCol])
                im = plt.pcolormesh(xc / 1e3, zc/1e3, 
                               rho * coarseBuoy, 
                               cmap=cmap, norm=norm, shading="nearest")
                plt.contour(xc / 1e3, zc / 1e3, np.logical_or((qc)>0, qi>1e-4), colors='black', linewidths=3)

                plt.sca(ax[1, i % numCol])
                im2 = plt.pcolormesh(xc / 1e3, zc/1e3, 
                               rho * coarseDm03, 
                               cmap=cmap, norm=norm, shading="nearest")
                plt.contour(xc / 1e3, zc / 1e3, np.logical_or((qc)>0, qi>1e-4), colors='black', linewidths=3)

                plt.sca(ax[2, i % numCol])
                im2 = plt.pcolormesh(xc / 1e3, zc/1e3, 
                               rho * coarseRes, 
                               cmap=cmap, norm=norm, shading="nearest")
                plt.contour(xc / 1e3, zc / 1e3, np.logical_or((qc)>0, qi>1e-4), colors='black', linewidths=3)

            for i in range(numRow*numCol):
                ax[i // numCol, i % numCol].contour(xc / 1e3, zc / 1e3, np.logical_or((qc)>0, qi>1e-4), colors='black', linewidths=3)
                ax[i // numCol, i % numCol].set_xticks(np.arange(0, 110, 2))
                ax[i // numCol, i % numCol].set_xlim(convObj["yRangeMin"][prof]/1e3-0.5, convObj["yRangeMax"][prof]/1e3)
                ax[i // numCol, i % numCol].set_ylim(0, 15)
                ax[i // numCol, i % numCol].set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
                ax[i // numCol, i % numCol].tick_params(axis='both', which='major', labelsize=35)
                ax[i // numCol, i % numCol].set_title(titles[i], fontsize=fs, y=1.015)

            plt.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.075, hspace=0.3, wspace=0.025)
            #cbar_ax = fig.add_axes([0.93, 0.695, 0.015, 0.25])
            cbar_ax = fig.add_axes([0.93, 0.075, 0.0175, 0.875])
            cb = fig.colorbar(im, cax=cbar_ax, extend="both", ticks=[round(x, 2) for x in np.linspace(-0.1, 0.1, 21)[::2]])
            cb.ax.tick_params(labelsize=30)
            #cbar_ax = fig.add_axes([0.93, 0.075, 0.015, 0.395])
            #cb = fig.colorbar(im2, cax=cbar_ax, extend="both", ticks=[round(x, 2) for x in np.linspace(-0.04, 0.04, 9)])
            #cb.ax.tick_params(labelsize=30)
            #fig.tight_layout()
            plt.savefig(f"./figure5-1.png", dpi=300)
            plt.clf()






