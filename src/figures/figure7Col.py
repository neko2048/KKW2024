import sys
import numpy as np
import netCDF4 as nc
import json
import matplotlib.pyplot as plt
from matplotlib import cm
sys.path.insert(0, "../")
import config
from util.vvmLoader import VVMLoader


if __name__ == "__main__":
    #initTimeStep, endTimeStep = int(sys.argv[1]), int(sys.argv[2])
    coarseReslList =  np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0])
    #coarseReslList =  np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0])
    realSpaceLength = np.array([0.1] + [round(0.6*x, 1) for x in coarseReslList[1:]])
    drawTicksPos = np.arange(0, len(coarseReslList[1:]))

    timeArange = np.array([400, 460])#np.arange(initTimeStep, endTimeStep)
    vvmLoader = VVMLoader(dataDir=config.vvmPath, subName="mjo_std_mg")
    thData = vvmLoader.loadThermoDynamic(0)
    xc, yc, zc = np.array(thData["xc"]), np.array(thData["yc"]), np.array(thData["zc"])
    rho = vvmLoader.loadRHO()[:-1]
    zz = vvmLoader.loadZZ()
    deltaZZ = zz[1:] - zz[:-1]


    cloudSizeList = np.array([])
    corrDistRecord = np.array([])
    folderPath = "correlation"
    forcings = ["total", "buoy", "dm03", "res"]
    titles = [r"(a) $\rho_0 \overline{a}$", r"(b) $\rho_0 \overline{a(\widetilde{B})}$", r"(c) $\rho_0 \overline{a(\widetilde{D_V})}$", r"(d) $\rho_0 \overline{a(\widetilde{D_H})}$"]
    fig, axs = plt.subplots(ncols=1, nrows=4, figsize=(16, 25))
    fs = 35
    for fIdx, force in enumerate(forcings):
        for tIdx in timeArange[:1]:
            print(f"========== {tIdx:06d} ==========")
            #convectQc = np.array(nc.Dataset(f"{config.convectQcPath}qcLabel-Connect-{tIdx:06d}.nc")["label"][0])
            #qcIndexList = np.unique(convectQc)
            #qcIndexList = qcIndexList[qcIndexList != 0]
            #print(qcIndexList)
            cloudSizeList = np.load(f"./{folderPath}L2/cloudSizeList-{tIdx:06d}-{force}.npy")
            corrDistRecord = np.load(f"./{folderPath}L2/corrDistRecord-{tIdx:06d}-{force}.npy")
        for tIdx in timeArange[1:]:
            cloudSizeList =  np.hstack((np.load(f"./{folderPath}L2/cloudSizeList-{tIdx:06d}-{force}.npy"), cloudSizeList))
            corrDistRecord = np.vstack((np.load(f"./{folderPath}L2/corrDistRecord-{tIdx:06d}-{force}.npy"), corrDistRecord))
        print(len(cloudSizeList))
        pr90, pr10 = np.percentile(cloudSizeList, 90), np.percentile(cloudSizeList, 10)
        print(pr90, pr10)
        plt.sca(axs[fIdx])
        for i in range(cloudSizeList.shape[0]):
            if cloudSizeList[i] <= np.percentile(cloudSizeList, 10):
                plt.plot(np.arange(len(realSpaceLength[1:])), corrDistRecord[i][1:], color='#1C73AC', alpha=0.25)
            if cloudSizeList[i] >= np.percentile(cloudSizeList, 90):
                plt.plot(np.arange(len(realSpaceLength[1:])), corrDistRecord[i][1:], color='#CF2A33', alpha=0.25)
        plt.ylabel("Norm. RMS Diff.", fontsize=fs)
        plt.plot(np.arange(len(realSpaceLength[1:])), np.mean(corrDistRecord[cloudSizeList<= pr10], axis=0)[1:], color='#1C73AC', linewidth=5)
        plt.plot(np.arange(len(realSpaceLength[1:])), np.mean(corrDistRecord[cloudSizeList>= pr90], axis=0)[1:], color='#CF2A33', linewidth=5)
        plt.xticks(np.arange(len(realSpaceLength[1:])), realSpaceLength[1:])
        plt.xlim(0, len(realSpaceLength[1:])-1)
        plt.grid(True)
        plt.ylim(-0.05, 1.4)
        plt.xlabel("s [km]", fontsize=fs)
        plt.title(titles[fIdx], fontsize=fs, y=1.015, loc='left')
        plt.xticks(fontsize=28)
        plt.yticks(np.arange(0, 1.5, 0.1), fontsize=28)
        axs[fIdx].set_yticklabels(["0.0", "", "0.2", "", "0.4", "", "0.6", "", "0.8", "", "1.0", "", "1.2", "", "1.4"], fontsize=30)
        plt.ylim(-0.01, 0.8)
    plt.subplots_adjust(bottom=0.05, top=0.95, wspace=0.1, hspace=0.4)
    plt.savefig("figure7Col.png", dpi=300)



