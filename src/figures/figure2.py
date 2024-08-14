import sys
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib import cm
sys.path.insert(0, "../")
import config
from util.vvmLoader import VVMLoader
from util.dataPloter import truncate_colormap
from util.dataPloter import getCmapAndNorm

if __name__ == "__main__":
    iniTimeIdx, endTimeIdx = 400, 401 #int(sys.argv[1]), int(sys.argv[2])
    
    vvmLoader = VVMLoader(dataDir=config.vvmPath, subName="mjo_std_mg")
    rho2D = vvmLoader.loadRHO()[1:, np.newaxis]
    thData = vvmLoader.loadThermoDynamic(0)
    xc, yc, zc = np.array(thData["xc"]), np.array(thData["yc"]), np.array(thData["zc"])
    timeArange = np.arange(iniTimeIdx, endTimeIdx)
    cmap = truncate_colormap(plt.get_cmap("Greys"), minval=0.25, maxval=1, n=5)
    #levels = [0, 0.01, 0.1, 1, 10, 100]
    levels = np.array([0, 0.01, 0.25, 0.5, 0.75] + [x for x in np.linspace(1, 5, 9)])
    greyCmap, greyNorm = getCmapAndNorm("binary", levels=levels, extend="max")
    levels = [round(x, 3) for x in np.arange(0, 0.21, 0.02)]
    levels.extend([-x for x in levels])
    levels = np.sort(np.unique(levels))
    print(levels)
    #dynCmap, dynNorm = getCmapAndNorm("RdBu_r", levels, extend="both")
    dynCmap, dynNorm = getCmapAndNorm("Spectral_r", levels, extend="both")
    section = 61000
    xIdx = np.argmin(np.abs(xc-section))
    fs = 30
    lw = 3

    for tIdx in timeArange:
        print(f"========== {tIdx:06d} ==========")
        thData = vvmLoader.loadThermoDynamic(tIdx)
        qc = np.roll(thData["qc"][0, :, :, xIdx], -30, axis=-1)
        qv = np.roll(thData["qv"][0, :, :, xIdx], -30, axis=-1)
        qr = np.roll(thData["qr"][0, :, :, xIdx], -30, axis=-1)
        qi = np.roll(thData["qi"][0, :, :, xIdx], -30, axis=-1)
        

        buoyancy = np.roll(nc.Dataset(f"{config.buoyancyPath}buoyancy-{tIdx:06d}.nc")["buoyancy"][0, :, :, xIdx], -30, axis=-1)
        dgData = nc.Dataset(f"{config.dynTermPath}uniform-1/diag-{tIdx:06d}.nc")
        dm03 = np.roll(dgData["dm03"][0, :, :, xIdx], -30, axis=-1)
        dm04 = np.roll(dgData["dm04"][0, :, :, xIdx], -30, axis=-1)
        dm05 = np.roll(dgData["dm05"][0, :, :, xIdx], -30, axis=-1)
        dm06 = np.roll(dgData["dm06"][0, :, :, xIdx], -30, axis=-1)
        dm07 = np.roll(dgData["dm07"][0, :, :, xIdx], -30, axis=-1)
        dyTerms = [buoyancy, dm03, dm05+dm06+dm07]
        dyTermsTitle = [r"(a) B [$m\cdot s^{-2}$]", r"(b) $D_V$ [$m\cdot s^{-2}$]", r"(c) $D_H$ [$\times 10^{-4} m^{-1}s^{-2}$]",
                        r"(d) $\rho_0$a(B) [$kg \cdot m^{-2} s^{-2}$]", r"(e) $\rho_0$a($D_V$) [$kg \cdot m^{-2} s^{-2}$]", r"(f) $\rho_0$a($D_H$) [$kg \cdot m^{-2} s^{-2}$]"]
        
        fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(25, 13))
        plt.sca(axs[0, 0])
        im = plt.pcolormesh(xc / 1e3, zc/1e3, dyTerms[0], shading="nearest", cmap=dynCmap, norm=dynNorm)
        plt.contour(xc/1e3, zc/1e3, qc>0, levels=[0.9], colors='black', linewidths=lw)
        plt.contour(xc/1e3, zc/1e3, qi>=1e-4, levels=[0.9], colors='#008000', linewidths=lw)

        plt.sca(axs[0, 1])
        im = plt.pcolormesh(xc / 1e3, zc/1e3, dyTerms[1], shading="nearest", cmap=dynCmap, norm=dynNorm)
        plt.contour(xc/1e3, zc/1e3, qc>0, levels=[0.9], colors='black', linewidths=lw)
        plt.contour(xc/1e3, zc/1e3, qi>=1e-4, levels=[0.9], colors='#008000', linewidths=lw)

        plt.sca(axs[0, 2])
        im = plt.pcolormesh(xc / 1e3, zc/1e3, 1e4*dyTerms[2], shading="nearest", cmap=dynCmap, norm=dynNorm)
        plt.contour(xc/1e3, zc/1e3, qc>0, levels=[0.9], colors='black', linewidths=lw)
        plt.contour(xc/1e3, zc/1e3, qi>=1e-4, levels=[0.9], colors='#008000', linewidths=lw)

        buoyancy = np.roll(nc.Dataset(f"{config.rebuildDynPath}uniform-1/a-{tIdx:06d}.nc")["a"][0, :, :, xIdx], -30, axis=-1)
        dgData = nc.Dataset(f"{config.dynTermAccePath}uniform-1/a-{tIdx:06d}.nc")
        #dm03 = np.roll(dgData["dm03"][0, :, :, xIdx], -30, axis=-1)
        dm04 = np.roll(dgData["dm04"][0, :, :, xIdx], -30, axis=-1)
        dm05 = np.roll(dgData["dm05"][0, :, :, xIdx], -30, axis=-1)
        dm06 = np.roll(dgData["dm06"][0, :, :, xIdx], -30, axis=-1)
        dm07 = np.roll(dgData["dm07"][0, :, :, xIdx], -30, axis=-1)
        dyTerms = [rho2D*buoyancy, rho2D*dm04, rho2D*(dm05+dm06+dm07)]
        #dyTermsTitle = [r"(a) Microphysics", r"(b) a(Buoyancy) [$m\cdot s^{-2}$]", r"(c) a(Vertical Momentum Flux) [$m\cdot s^{-2}$]", r"(d) a(Horizontal Momentum Flux) [$m\cdot s^{-2}$]"]
        levels2 = [round(x, 3) for x in np.linspace(0, 0.1, 11)]
        levels2.extend([-x for x in levels2])
        levels2 = np.sort(np.unique(levels2))
        print(levels2)
        dynCmap, dynNorm = getCmapAndNorm("RdBu_r", levels2, extend="both")
        plt.sca(axs[1, 0])
        im2 = plt.pcolormesh(xc / 1e3, zc/1e3, dyTerms[0], shading="nearest", cmap=dynCmap, norm=dynNorm)
        plt.contour(xc/1e3, zc/1e3, qc>0, levels=[0.9], colors='black', linewidths=lw)
        plt.contour(xc/1e3, zc/1e3, qi>=1e-4, levels=[0.9], colors='#008000', linewidths=lw)

        plt.sca(axs[1, 1])
        im2 = plt.pcolormesh(xc / 1e3, zc/1e3, dyTerms[1], shading="nearest", cmap=dynCmap, norm=dynNorm)
        plt.contour(xc/1e3, zc/1e3, qc>0, levels=[0.9], colors='black', linewidths=lw)
        plt.contour(xc/1e3, zc/1e3, qi>=1e-4, levels=[0.9], colors='#008000', linewidths=lw)

        plt.sca(axs[1, 2])
        im2 = plt.pcolormesh(xc / 1e3, zc/1e3, dyTerms[2], shading="nearest", cmap=dynCmap, norm=dynNorm)
        plt.contour(xc/1e3, zc/1e3, qc>0, levels=[0.9], colors='black', linewidths=lw)
        plt.contour(xc/1e3, zc/1e3, qi>=1e-4, levels=[0.9], colors='#008000', linewidths=lw)


        for i in range(2):
            for j in range(3):
                plt.sca(axs[i, j])
                plt.xticks(np.arange(0, 120, 2), fontsize=fs)
                plt.yticks(np.arange(0, 15, 2), fontsize=fs)
                plt.ylim(0, 15)
                plt.xlim(89.5, 102.4)
                plt.title(dyTermsTitle[3*i+j], fontsize=fs, y=1.025)
                if j == 0: plt.ylabel("z [km]", fontsize=fs)
                if i == 1: plt.xlabel("x [km]", fontsize=fs)
        cax = fig.add_axes([0.92, 0.555, 0.02, 0.395])
        cb = fig.colorbar(im, cax=cax, extend="both", ticks=levels[::2])#, format='%.0e')
        cb.ax.tick_params(labelsize=fs)
        cax2 = fig.add_axes([0.92, 0.075, 0.02, 0.395])
        cb2 = fig.colorbar(im2, cax=cax2, extend="both", ticks=levels2[::2])#, format='%.0e')
        cb2.ax.tick_params(labelsize=fs)
        
        plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.075, wspace=0.15, hspace=0.25)
        plt.savefig(f"figure2.png", dpi=300)
