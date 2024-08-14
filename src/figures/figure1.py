import sys
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib import cm
sys.path.insert(0, "../")
from util.vvmLoader import VVMLoader
from util.dataPloter import getCmapAndNorm
import config


if __name__ == "__main__":

    vvmLoader = VVMLoader(dataDir=config.vvmPath, subName="mjo_std_mg")
    thData = vvmLoader.loadThermoDynamic(0)
    xc, yc, zc = np.array(thData["xc"]), np.array(thData["yc"]), np.array(thData["zc"])
    timeArange = [400]
    fs = 30
    level = 5000#1500
    argz = np.argmin(np.abs(zc-level))
    print(zc[argz])
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10), gridspec_kw={'width_ratios': [1, 1, 1.5]})
    for i, tIdx in enumerate(timeArange):
        print(f"========== {tIdx:06d} ==========")
        rdData = vvmLoader.loadRadiation(tIdx)
        olr = np.array(rdData["fulwtoa"][0])
        thData = vvmLoader.loadThermoDynamic(tIdx)
        qc = np.array(thData["qc"][0])
        qi = np.array(thData["qi"][0])
        qr = np.array(thData["qr"][0])
        buoyancy = np.array(nc.Dataset(f"{config.buoyancyPath}buoyancy-{tIdx:06d}.nc")["buoyancy"][0])
        qc = np.roll(qc, -30, axis=-2)
        qi = np.roll(qi, -30, axis=-2)
        qr = np.roll(qr, -30, axis=-2)
        buoyancy = np.roll(buoyancy, -30, axis=-2)
        olr = np.roll(olr, -30, axis=-2)

        plt.sca(axs[0])
        plt.grid(True)
        im1=plt.pcolormesh(xc / 1e3, yc/1e3, olr.transpose(), cmap="jet", shading="nearest", vmin=100, vmax=280)
        cb1=plt.colorbar(im1, extend='both', ticks=np.linspace(100, 280, 7))
        cb1.ax.tick_params(labelsize=25)
        plt.plot([90, 90, 97, 97, 90], [60, 67, 67, 60, 60], linewidth=2.5, color='black')
        plt.xlabel("x [km]", fontsize=fs)
        plt.ylabel("y [km]", fontsize=fs)
        label = np.arange(0, 101, 10)
        shownLabel = np.arange(0, 101, 20)
        plt.xticks(np.arange(0, 101, 10), [], fontsize=fs)
        plt.yticks(np.arange(0, 101, 10), [], fontsize=fs)
        axs[0].set_xticklabels([x if x in shownLabel else "" for x in label], fontsize=fs)
        axs[0].set_yticklabels([x if x in shownLabel else "" for x in label], fontsize=fs)
        #plt.title("t={hr:02d}h{minute:02d}m".format(
        #        hr = tIdx * config.minPerTimeIdx // 60,
        #        minute = tIdx * config.minPerTimeIdx % 60),
        #        loc='right', fontsize=25, y=1.005)
        plt.title(rf"(a) OLR [W/$m^2$]", fontsize=fs, y=1.015)
        #axs[0].set_box_aspect(aspect=1)


        plt.sca(axs[1])
        #plt.title("t={hr:02d}h{minute:02d}m".format(
        #        hr = tIdx * config.minPerTimeIdx // 60,
        #        minute = tIdx * config.minPerTimeIdx % 60),
        #        loc='right', fontsize=25, y=1.005)
        plt.title(r"(b) B [$m/s^{2}$] (z=5km)", fontsize=fs, y=1.015)
        plt.grid(True)
        im2=plt.pcolormesh(xc / 1e3, yc / 1e3, buoyancy[argz, :, :].transpose(), cmap="Spectral_r", vmin=-0.05, vmax=0.05)
        cb2=plt.colorbar(im2, extend='both', ticks=np.linspace(-0.05, 0.05, 11))
        cb2.ax.tick_params(labelsize=25)
        plt.contour(xc / 1e3, yc / 1e3, (qc+qi)[argz].transpose() > 0, colors='black')
        plt.plot([90, 90, 97, 97, 90], [60, 67, 67, 60, 60], linewidth=2.5, color='magenta')
        plt.xlabel("x [km]", fontsize=fs)
        plt.ylabel("y [km]", fontsize=fs)
        plt.xticks(np.arange(0, 101, 10), [], fontsize=fs)
        plt.yticks(np.arange(0, 101, 10), [], fontsize=fs)
        axs[1].set_xticklabels([x if x in shownLabel else "" for x in label], fontsize=fs)
        axs[1].set_yticklabels([x if x in shownLabel else "" for x in label], fontsize=fs)
        #axs[1].set_box_aspect(aspect=1)


        plt.sca(axs[2])
        plt.grid(True)
        levels = np.array([0, 0.01, 0.25, 0.5, 0.75] + [x for x in np.linspace(1, 5, 9)])
        greyCmap, greyNorm = getCmapAndNorm("binary", levels=levels, extend="max")
        qcPlot=plt.pcolormesh(xc/1e3, zc/1e3, qc[:, :, np.argmin(np.abs(xc - 61000))]*1e3, cmap=greyCmap, norm=greyNorm)
        plt.contour(xc/1e3, zc/1e3, qi[:, :, np.argmin(np.abs(xc - 61000))]>1e-4, levels=[0.9], colors='green', linewidths=3)
        plt.contourf(xc/1e3, zc/1e3, np.ma.masked_array(np.ones(qr[:, :, np.argmin(np.abs(xc - 61000))].shape), qr[:, :, np.argmin(np.abs(xc - 61000))]<0.5e-4), colors='cyan', alpha=0.25)
        cs = plt.contourf(xc/1e3, zc/1e3, np.ma.masked_array(np.ones(qr[:, :, np.argmin(np.abs(xc - 61000))].shape), qr[:, :, np.argmin(np.abs(xc - 61000))]<10e-4), hatches="/////", colors="None")
        for i in cs.collections:
            i.set_edgecolor("blue")
        cbar=plt.colorbar(qcPlot, extend="max")
        cbar.ax.tick_params(labelsize=25)
        cbar.set_ticks(np.array([0, 0.01, 0.25, 0.5, 0.75] + [x for x in np.linspace(1, 5, 9)]))
        plt.xticks(np.arange(0, 102.4, 2), fontsize=fs)
        plt.xlim(89.5, 102.4)
        plt.yticks(np.arange(0, 15, 2), fontsize=fs)
        plt.ylim(0, 15)
        plt.xlabel("x [km]", fontsize=fs)
        plt.ylabel("z [km]", fontsize=fs)
        plt.title(r"(c) Cloud Water / Ice / Rain (y=61km)", fontsize=fs, y=1.015)
        #plt.title("t={hr:02d}h{minute:02d}m".format(
        #        hr = tIdx * config.minPerTimeIdx // 60,
        #        minute = tIdx * config.minPerTimeIdx % 60),
        #        loc='right', fontsize=25, y=1.005)

    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.2, top=0.8, wspace=0.2)
    plt.savefig(f"./figureS1.png", dpi=300)


