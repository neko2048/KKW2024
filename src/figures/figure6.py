import sys
import numpy as np
import netCDF4 as nc
import json
import matplotlib.pyplot as plt
from matplotlib import cm
sys.path.insert(0, "../")
import config
from util.vvmLoader import VVMLoader

def getSimilarity(a, b):
    return np.sum(a * b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))


if __name__ == "__main__":
    #initTimeStep, endTimeStep = 400, 401#int(sys.argv[1]), int(sys.argv[2])
    coarseReslList =  np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0])
    realSpaceLength = np.array([0.1] + [round(0.6*x, 1) for x in coarseReslList[1:]])
    drawTicksPos = np.arange(0, len(coarseReslList[1:]))

    timeArange = np.array([400])#np.arange(initTimeStep, endTimeStep)
    vvmLoader = VVMLoader(dataDir=config.vvmPath, subName="mjo_std_mg")
    thData = vvmLoader.loadThermoDynamic(0)
    xc, yc, zc = np.array(thData["xc"]), np.array(thData["yc"]), np.array(thData["zc"])
    rho = vvmLoader.loadRHO()[:-1]
    zz = vvmLoader.loadZZ()
    deltaZZ = zz[1:] - zz[:-1]
    xMin, xMax = np.argmin(np.abs(xc-60000)), np.argmin(np.abs(xc-67000))+1
    yMin, yMax = np.argmin(np.abs(yc-90000)), np.argmin(np.abs(yc-97000))+1
    buoySim = [0.026692423611034192, 0.02793435616307762, 0.029220300141875457, 0.03457605144161183, 0.039093443345510175, 0.04398808997878078, 0.04913802192515681, 0.05989603367601429, 0.08732307132901136, 0.12386885120318498, 0.2419884048639343, 0.37996528884380065]
    vmfSim =  [0.1194235395097247, 0.1287852490743033, 0.1397469647006012, 0.1538909900348023, 0.17137394438693965, 0.19146951549930752, 0.21391395299176916, 0.261739568418034, 0.3658771706641795, 0.44049825708261314, 0.587747026766247, 0.737604327924134]
    hmfSim =  [0.1346619263709357, 0.16688040193301737, 0.20938546030740612, 0.25536797900816177, 0.30107300850117596, 0.3437595464093652, 0.3829371671969503, 0.452151932607757, 0.5993405898689351, 0.7300753772304565, 0.8931661855536532, 0.9541728492415363]
    totSim =  [0.05446106560239769, 0.06151875979818718, 0.06688512873637069, 0.07398078224389529, 0.08202499716315154, 0.09195095803124574, 0.10336721588282796, 0.12867851226656143, 0.19629979438580172, 0.2647803615242672, 0.39735938692485795, 0.5198411023291962]
    
    #for tIdx in timeArange[:1]:
    #    print(f"========== {tIdx:06d} ==========")
    #    origBuoAcce = np.mean(np.roll(nc.Dataset(config.rebuildDynPath + f"uniform-1/a-{tIdx:06d}.nc")["a"][0], -30, axis=-1)[:, yMin:yMax, xMin:xMax], axis=(1, 2))
    #    origVmfAcce = np.mean(np.roll(nc.Dataset(config.dynTermDirectAccePath + f"uniform-1/a-{tIdx:06d}.nc")["dm04"][0], -30, axis=-1)[:, yMin:yMax, xMin:xMax], axis=(1, 2))
    #    origHmfAcce =  np.roll(nc.Dataset(config.dynTermDirectAccePath + f"uniform-1/a-{tIdx:06d}.nc")["dm05"][0], -30, axis=-1)[:, yMin:yMax, xMin:xMax]
    #    origHmfAcce += np.roll(nc.Dataset(config.dynTermDirectAccePath + f"uniform-1/a-{tIdx:06d}.nc")["dm06"][0], -30, axis=-1)[:, yMin:yMax, xMin:xMax]
    #    origHmfAcce += np.roll(nc.Dataset(config.dynTermDirectAccePath + f"uniform-1/a-{tIdx:06d}.nc")["dm07"][0], -30, axis=-1)[:, yMin:yMax, xMin:xMax]
    #    origHmfAcce /= 3
    #    origHmfAcce = np.mean(origHmfAcce, axis=(1, 2))
    #    origAcce = origBuoAcce + origVmfAcce + origHmfAcce
    #    for std in coarseReslList[1:]:
    #        print(std)
    #        convBuoAcce = np.mean(np.roll(nc.Dataset(config.rebuildDynPath + f"gaussian-{std:.1f}/a-{tIdx:06d}.nc")["a"][0], -30, axis=-1)[:, yMin:yMax, xMin:xMax], axis=(1, 2))
    #        convVmfAcce = np.mean(np.roll(nc.Dataset(config.dynTermDirectAccePath + f"gaussian-{std:.1f}/a-{tIdx:06d}.nc")["dm04"][0], -30, axis=-1)[:, yMin:yMax, xMin:xMax], axis=(1, 2))
    #        convHmfAcce = np.roll(nc.Dataset(config.dynTermDirectAccePath + f"gaussian-{std:.1f}/a-{tIdx:06d}.nc")["dm05"][0], -30, axis=-1)[:, yMin:yMax, xMin:xMax]
    #        convHmfAcce += np.roll(nc.Dataset(config.dynTermDirectAccePath + f"gaussian-{std:.1f}/a-{tIdx:06d}.nc")["dm06"][0], -30, axis=-1)[:, yMin:yMax, xMin:xMax]
    #        convHmfAcce += np.roll(nc.Dataset(config.dynTermDirectAccePath + f"gaussian-{std:.1f}/a-{tIdx:06d}.nc")["dm07"][0], -30, axis=-1)[:, yMin:yMax, xMin:xMax]
    #        convHmfAcce /= 3
    #        convHmfAcce = np.mean(convHmfAcce, axis=(1, 2))
    #        convAcce = convBuoAcce + convVmfAcce + convHmfAcce
    #        buoySim.append(getSimilarity(origBuoAcce, convBuoAcce))
    #        vmfSim.append(getSimilarity(origVmfAcce, convVmfAcce))
    #        hmfSim.append(getSimilarity(origHmfAcce, convHmfAcce))
    #        totSim.append(getSimilarity(origAcce, convAcce))
    #np.save("./buoySim.npy", buoySim)
    #print(buoySim)
    #np.save("./vmfSim.npy", vmfSim)
    #print(vmfSim)
    #np.save("./hmfSim.npy", hmfSim)
    #print(hmfSim)
    #np.save("./totSim.npy", totSim)
    #print(totSim)

    fs = 35
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    plt.sca(axs)
    plt.plot(np.arange(len(realSpaceLength[1:])), buoySim, color='red', linewidth=5, label=r"$\rho_0 \overline{a(\widetilde{B})}$")
    plt.plot(np.arange(len(realSpaceLength[1:])), vmfSim, color='#4DBEEE', linewidth=5, label=r"$\rho_0 \overline{a(\widetilde{D_V})}$")
    plt.plot(np.arange(len(realSpaceLength[1:])), hmfSim, color='#77AC30', linewidth=5, label=r"$\rho_0  \overline{a(\widetilde{D_H})}$")
    plt.plot(np.arange(len(realSpaceLength[1:])), totSim, color='black', linewidth=5, label=r"$\rho_0 \overline{a}$")
    plt.legend(fontsize=fs)

    plt.sca(axs)
    plt.xticks(np.arange(len(realSpaceLength[1:])), realSpaceLength[1:])
    plt.xlim(0, len(realSpaceLength[1:])-1)
    plt.grid(True)
    plt.title("Dependence on smoothing scale", fontsize=fs, y=1.015)
    plt.xlabel("s [km]", fontsize=fs)
    plt.ylabel("Norm. RMS Difference", fontsize=fs)
    plt.xticks(np.arange(len(realSpaceLength[1:])), realSpaceLength[1:], fontsize=30)
    plt.yticks(np.arange(0, 0.55, 0.05), [])
    axs.set_yticklabels(["0.0", "", "0.1", "", "0.2", "", "0.3", "", "0.4", "", "0.5"], fontsize=30)
    plt.ylim(-0.01, 0.5)
    plt.subplots_adjust(left=0.10, right=0.95)
    plt.savefig("figure6.png", dpi=300)




