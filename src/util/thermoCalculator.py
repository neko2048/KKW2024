import numpy as np
# ========== get variables ==========
def getTemperature(theta, pBar):
    temp = theta * ((pBar / 100000) ** (287 / 1004))
    return temp

def getMSE(temperature, zc3D, qv):
    mse = 1004 * temperature + 9.8 * zc3D + 2.5e6 * qv
    return mse

# crate pseudo-adiabatic / dry-adiabatic
def cal_saturated_vapor_pressure(T_K, simple=True):
    if simple == True:
        es = 6.11 * np.exp(2.5e6 / 287 * (1 / 273.15 - 1 / T_K))
        return es
    else:
        T_K = np.where(T_K-273.15<-50, -50+273.15, T_K)
        # Goff-Gratch formulation, ICE
        esi_hPa = 10.**(-9.09718*(273.16/T_K-1)\
                        -3.56654*np.log10(273.16/T_K)\
                        +0.876793*(1-T_K/273.16)\
                        +np.log10(6.1071))
        # Goff-Gratch formulation, LIQUID
        es_hPa = 10**(-7.90298*(373.16/T_K-1)\
                     +5.02808*np.log10(373.16/T_K)\
                     -1.3816e-7*(10**(11.344*(1-T_K/373.16))-1)\
                     +8.1328e-3*(10**(-3.49149*(373.16/T_K-1))-1)\
                     +np.log10(1013.246))
        #es_hPa = 6.112 * np.exp(17.67 * (T_K - 273.15)/ (T_K - 29.65))
        return np.where(T_K>=273.15, es_hPa, esi_hPa)

def cal_absolute_humidity(vapor_pressure_hPa,pressure_hPa):
    #dum = np.where((pressure_hPa-vapor_pressure_hPa)<1e-5, 1e-5, pressure_hPa-vapor_pressure_hPa)
    dum = pressure_hPa-vapor_pressure_hPa
    mixing_ratio = 0.622*vapor_pressure_hPa/dum
    return mixing_ratio

def cal_equivalent_potential_temperature(P_hPa, rv_kgkg, t_K):
    #theta_e_K = t_K*(1000/P_hPa)**(287.05/1004)*np.exp(2.5e6*rv_kgkg/1004/t_K)
    theta_e_K = (t_K+2.5e6/1004*rv_kgkg)*(1000/P_hPa)**(287.05/1004)
    return theta_e_K
def cal_potential_temperature(P_hPa, T_K):
    theta_K = T_K*(1000/P_hPa)**(287.05/1004)
    return theta_K
def cal_saturated_rv(P_hPa,T_K):
    es_hPa = cal_saturated_vapor_pressure(T_K, simple=True)
    qv = cal_absolute_humidity(es_hPa, P_hPa)
    #print(es_hPa, qv, rv)
    return qv

def parcel_profile_2d(Temp02d_K,Press1d_hPa, qv02d_kgkg, Height1d_m):
    # conservation of equivalent potential temperature and potential temperature
    # interpolate to conservate theta_e
    #trange=np.arange(-40,50,0.01)+273.15
    trange=np.arange(-60,50,0.05)+273.15
    tt,pp=np.meshgrid(trange,Press1d_hPa)
    es_hPa = cal_saturated_vapor_pressure(tt, simple=True)
    rr = cal_absolute_humidity(es_hPa, pp)
    #rr = np.where(pp<300, 0.0, rr)
    theta_e = cal_equivalent_potential_temperature(pp,rr,tt)-273.15
    theta_e = np.where(theta_e>500,np.nan,theta_e)

    nz, ny, nx = Press1d_hPa.size, Temp02d_K.shape[0], Temp02d_K.shape[1]
  
    temp_dry = (Temp02d_K.reshape(1,ny,nx))*(Press1d_hPa.reshape(nz,1,1)/Press1d_hPa[0])**0.2854
    qv = cal_saturated_rv(Press1d_hPa.reshape(nz,1,1), temp_dry)
    idxLCL = np.argmin(np.abs(qv-qv02d_kgkg.reshape(1,ny,nx)), axis=0)

    parcel3d = np.copy(temp_dry)
    idxLCL = idxLCL.reshape(ny,nx)
    idxY   = np.arange(ny).reshape(ny, 1)
    idxX   = np.arange(nx).reshape(1, nx)

    idxt2d = np.argmin(np.abs(trange.reshape(trange.size,1,1)-parcel3d[idxLCL, idxY, idxX].reshape(1,ny,nx)),axis=0)
    conserve_thetae2d = theta_e[idxLCL,idxt2d]

    idx200 = np.argmin(np.abs(Press1d_hPa-200))
    for idx in range(idx200):
      ind = np.where(idxLCL<=idx)
      if len(ind[0])==0: continue
      refresh = np.interp(conserve_thetae2d[ind], theta_e[idx,:], trange)
      parcel3d[idx*np.ones(ind[0].size,dtype=int),ind[0], ind[1]] = refresh
    for idx in range(idx200, nz):
      parcel3d[idx,:,:] = parcel3d[idx-1,:,:]-0.0098*(Height1d_m[idx]-Height1d_m[idx-1])
    return idxLCL, parcel3d

def cal_CAPE(temp_env_K, temp_parcel_K, Height_1d_m):
    (nz, ny, nx) = temp_env_K.shape
    dhei = np.diff(Height_1d_m).reshape(nz-1,1,1)
    diff = (temp_parcel_K-temp_env_K)/(temp_env_K)
    area = (diff[:-1]+diff[1:])*dhei/2*9.81

    idxEL = area.shape[0] - np.argmax(area[::-1,:,:]>=0, axis=0)
    idxEL3dmask = np.arange(nz-1).reshape(nz-1,1,1)<=idxEL.reshape(1,ny,nx)

    CAPE = np.sum(area, axis=0, where = (area>=0)*(idxEL3dmask))
    #CIN  = np.sum(area, axis=0, where = (area<=0)*(idxEL3dmask))
    return CAPE

def getCAPE(th, qv, zc3D, pZC3D, dz3D):
    tempK = getTemperature(th, pBar=pZC3D[:-1])
    qvs = cal_saturated_rv(pZC3D[:-1]/1e2, tempK)
    envHms = getMSE(tempK, zc3D, qvs)
    parcelHm = getMSE(tempK[0], zc3D[0], qv[0])[np.newaxis, :, :]
    deltaEnergy = 9.8*(((parcelHm - envHms) / 1004) / (th * (1 + 0.608*qv))) * dz3D
    return np.sum(deltaEnergy, axis=0, where=deltaEnergy>=0) 
    
