import numpy as np
import netCDF4 as nc

outputPath = "/data/atmenu10246/convBuoy/dat/vertAcce/"
for i in ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '5.0', '7.5', '10.0', '15.0', '20.0']:
#[780, 760, 750, 740, 730, 720, 710, 690, 680, 670, 650, 590, 580, 570, 460, 440, 420]:
    data = nc.Dataset(outputPath + f"gaussian-{i}/a-000780.nc")["a"][0]
    print(i)
    print(np.max(data), np.min(data))
    print("")
