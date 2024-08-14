import numpy as np
from scipy.signal import correlate2d, correlate, gaussian
from scipy.ndimage import laplace
from multiprocessing import Pool

def getPseudoAlbedo(nc, qc, deltaZZ, rho3D):
    colNC = np.sum((nc * deltaZZ)[1:], axis=0)
    LWP = np.sum((rho3D * qc * deltaZZ)[1:], axis=0)
    cloudOptDept = 0.19 * (LWP ** (5 / 6)) * (colNC ** (1 / 3))
    A = cloudOptDept / (6.8 + cloudOptDept)
    return A

def hrzAveCoarsen(data, numFrameY, numFrameX):
    presentPoints = []
    averageData = np.zeros(shape=(data.shape[0], data.shape[1]+1, data.shape[2]+1))
    averageData[:, 0:-1, 0:-1] = data
    for yIdx in np.arange(0, data.shape[1]+1, numFrameY):
        for xIdx in np.arange(0, data.shape[2]+1, numFrameX):
            presentPoints.append([yIdx, xIdx])
    for p in presentPoints:
        averageData[:, p[0]:p[0]+numFrameY, p[1]:p[1]+numFrameX] = np.mean(averageData[:, p[0]:p[0]+numFrameY, p[1]:p[1]+numFrameX], axis=(1, 2), keepdims=True)
    return averageData[:, :-1, :-1]

def vtcAveCoarsen(data, numFrameZ, deltaZZ3D, totalDeltaZZ3D):
    averageData = np.zeros(shape=(data.shape[0]+1, data.shape[1], data.shape[2]))
    averageData[0:-1, 0:, 0:] = data

    for i in range(len(zc)+1)[::5]:
        averageData[i:i+numFrameZ, :, :] = np.sum(averageData[i:i+numFrameZ, :, :] * deltaZZ3D[i:i+numFrameZ, :, :], axis=0, keepdims=True)
    averageData = averageData[:-1, :, :]
    averageData = averageData / totalDeltaZZ3D
    #averageData[len(zc)-numFrameZ:] = np.sum(averageData[len(zc)-numFrameZ:, :, :] * deltaZZ3D[len(zc)-numFrameZ:, :, :], axis=(0), keepdims=True)
    return averageData

def getConvolveWeight(hrzFrameSize):
    """
    INPUT: hrzFrameSize, only positive integer or XX.5 is accepted currently
    """
    integerPart, decimalPart = int(hrzFrameSize), round(hrzFrameSize - int(hrzFrameSize), 1)
    closestOddNum = integerPart if integerPart % 2 == 1 else integerPart - 1
    weightBoundVal = np.round(((integerPart - closestOddNum) + decimalPart) / 2, 2)
    weightSize = closestOddNum + 2 * (weightBoundVal != 0.0)
    weight2D = np.full(fill_value = 1., shape=(weightSize, weightSize))
    
    if (weightBoundVal != 0.0):    
        weight2D[0, :]  *= weightBoundVal
        weight2D[-1, :] *= weightBoundVal
        weight2D[:, 0]  *= weightBoundVal
        weight2D[:, -1] *= weightBoundVal
     
    return weight2D

def getExpandSize(weight):
    if weight.shape[0] % 2 == 1:
        expandSize = (weight.shape[0] - 1) // 2
    else:
        expandSize = (weight.shape[0] - 1) // 2 + 1
    return expandSize

def getExpandData(data, weight):
    expandSize = getExpandSize(weight)
    expandData = np.pad(data, 
                        pad_width=((0, 0), 
                                   (expandSize, expandSize), 
                                   (expandSize, expandSize)), 
                        mode="wrap")
    return expandData

def getConvolve(data, hrzFrameSize, method="auto"):
    """method: direct / fft (error: O(1e-16))"""
    if method == "auto":
        method = chooseConvolMethod(hrzFrameSize)

    weight = getConvolveWeight(hrzFrameSize)
    expandSize = getExpandSize(weight)
    expandData = getExpandData(data, weight)
    convData = np.zeros(shape=data.shape)
    for i in range(data.shape[0]):
        print(i, data.shape[0])
        convData[i] = (correlate(expandData[i], weight, method=method) / (hrzFrameSize**2))[2*expandSize:-2*expandSize, 2*expandSize:-2*expandSize]
    return convData

def chooseConvolMethod(hrzFrameSize, maxThres=10):
    if hrzFrameSize >= maxThres:
        return "fft"
    else:
        return "direct"

def getHrzLaplacian(data):
    laplacianData = np.zeros(shape=data.shape)
    for i in range(data.shape[0]):
        laplace(data[i], output=laplacianData[i], mode='wrap')
    return laplacianData


# ========= multi-processing ========
def convolve(data, weight, hrzFrameSize, expandSize, i, method="direct"):
    if method == "direct":
        convolveData = ((correlate2d(data[i], weight)) / (hrzFrameSize**2))[2*expandSize:-2*expandSize, 2*expandSize:-2*expandSize]
    elif method == "fft":
        convolveData = ((correlate(data[i], weight, method="fft")) / (hrzFrameSize**2))[2*expandSize:-2*expandSize, 2*expandSize:-2*expandSize]
    return convolveData

def partitionDataPool(data, smallArrShape, expandSize):
    inputData = [data[:smallArrShape+expandSize*2, :smallArrShape+expandSize*2], 
                 data[-smallArrShape-expandSize*2:, :smallArrShape+expandSize*2], 
                 data[:smallArrShape+expandSize*2, -smallArrShape-expandSize*2:], 
                 data[-smallArrShape-expandSize*2:, -smallArrShape-expandSize*2:]]
    return inputData

def combinePartitionData(partitionData, smallArrShape, outputData):
    outputData[:smallArrShape, :smallArrShape] = partitionData[0]
    outputData[-smallArrShape:, :smallArrShape] = partitionData[1]
    outputData[:smallArrShape, -smallArrShape:] = partitionData[2]
    outputData[-smallArrShape:, -smallArrShape:] = partitionData[3]
    return outputData

def getConvolveParallel(data, hrzFrameSize, method="fft", partitionSize=2):
    expandSize = getExpandSize(hrzFrameSize)
    smallArrShape = int(data.shape[1] / partitionSize)
    expandData = getExpandData(data, hrzFrameSize)
    weight = getConvolveWeight(hrzFrameSize)
    convData = np.zeros(shape=data.shape)
    for i in range(10):#data.shape[0]):
        print(i, data.shape[0])
        inputData = partitionDataPool(expandData[i], smallArrShape, expandSize)
        #returnData = [np.zeros(shape=(smallArrShape, smallArrShape))] * (partitionSize ** 2)
        with Pool(processes=partitionSize**2) as p:
            returnData = p.starmap(convolve, [
                (inputData, weight, hrzFrameSize, expandSize, 0, method), 
                (inputData, weight, hrzFrameSize, expandSize, 1, method), 
                (inputData, weight, hrzFrameSize, expandSize, 2, method), 
                (inputData, weight, hrzFrameSize, expandSize, 3, method), 
            ])
        convData[i] = combinePartitionData(returnData, smallArrShape, convData[i])
    return convData


# ========== Gaussian Kernel =========
def getGaussianWeight(kernelSize, std=1, normalize=True):
    gaussian1D = gaussian(kernelSize, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalize:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D

def getGaussianConvolve(data, kernel, method="fft"):
    """method: direct / fft (error: O(1e-16))"""
    expandSize = getExpandSize(kernel)
    expandData = np.pad(data, 
                        pad_width=((0, 0), 
                                   (512, 512), 
                                   (512, 512)), 
                        mode="wrap")
    convData = np.zeros(shape=data.shape)

    for i in range(data.shape[0]):
        print(i, data.shape[0])
        convData[i] = correlate(expandData[i], kernel, method=method)[2*expandSize:-2*expandSize+1, 2*expandSize:-2*expandSize+1]
    return convData

# ========== Total Variation Denoise ==========
def getSumOfSquareError(x, y, deltaZZ3D=None):
    if deltaZZ == None:
        return (x - y) ** 2
    else:
        weightZZ3D = deltaZZ3D / np.sum(deltaZZ3D, axis=0)[0, 0]
        return ((x - y) * weightZZ3D) ** 2

def getTotalVariation(data, method="isotropic", deltaXY=100):
    gradientX = np.gradient(data, axis=2)
    gradientX[:, 0, :] = (data[:, 1, :] - data[:, -1, :]) / 2
    gradientX[:, -1, :] = (data[:, 0, :] - data[:, -2, :]) / 2
    gradientX /= deltaXY

    gradientY = np.gradient(data, axis=1)
    gradientY[:, :, 0] = (data[:, :, 1] - data[:, :, -1]) / 2
    gradientY[:, :, -1] =(data[:, :, 0] - data[:, :, -2]) / 2
    gradientY /= deltaXY

    return gradientX, gradientY
    #if method == "isotropic":
    #    return np.sum(np.sqrt(gradientX ** 2 + gradientY ** 2))
    #elif method == "anisotropic":
    #    return np.sum(np.abs(gradientX) + np.abs(gradientY))

def getTaperMask(xc, yc, xMin, xMax, yMin, yMax):
    mask = np.zeros(shape=(len(xc), len(yc)))
    mxc, myc = np.meshgrid(xc, yc)
    mask1 = np.exp(-1 / (mxc - xMin)**2) * np.exp(-1 / (mxc - xMax)** 2)
    mask1[np.logical_or(mxc <= xMin, mxc >= xMax)] = 0
    mask2 = np.exp(-1 / (myc - yMin)**2) * np.exp(-1 / (myc - yMax)** 2)
    mask2[np.logical_or(myc <= yMin, myc >= yMax)] = 0
    mask = mask1 * mask2
    return mask






# ========== get variables ==========
def getTemperature(theta, pBar):
    temp = theta * ((pBar / 100000) ** (287 / 1004))
    return temp

def getMSE(temperature, zc3D, qv):
    mse = 1004 * temperature + 9.81 * zc3D + 2.5e6 * qv
    return mse
