import pydicom
import numpy as np
import SimpleITK as sitk

def map_hu_to_density(hu_array, mapping='MC'):
    ''' 
    Description
    -----------
    Maps the HU values to Density values. Density mapping are taken from Eclipse.
    Provides options MC Physical Density, Eclipse Relative Electron Density or Acuros XB Physical Density mapping
    
    Parameters
    ----------
    hu_array :      3D Numpy Array (float/int)
                    Array containing HU values

    Returns
    -------
    red_array :     3D Numpy Array (float)
                    Array containing mapped relative electron densities
                    
    density_array : 3D Numpy Array (float)
                    Array containing mapped physical densities
    '''
    
    if mapping == 'AXB':
        # Acuros XB HU-Physical Density Mapping
        hu_array_axb = np.clip(hu_array, -1050, 7700)
        hu_grid_axb = np.array([-1050, -1000, -808, -73, 0, 99, 1503, 2065, 4200, 6000, 6100, 7700])
        pd_axb_grid = np.array([0.001, 0.001, 0.240, 0.95, 1.0, 1.190, 1.933, 2.70, 4.2, 7.85, 8.0, 8.8])
        density_array = np.interp(hu_array_axb, hu_grid_axb, pd_axb_grid)
    else:
        # Monte Carlo HU-Physical Density Mapping (for EGSPhu_arrayNT)
        hu_array_mc = np.clip(hu_array, -1050, 2000)
        hu_grid_mc    = np.array([-1050, -1000, -950,-700,125,2000])
        pd_mc_grid = np.array([0.001, 0.001, 0.044, 0.302, 1.101, 2.088])
        density_array = np.interp(hu_array_mc, hu_grid_mc, pd_mc_grid)

    # Eclipse HU-Relative Electron Density Mapping
    hu_array_red = np.clip(hu_array, -1050, 7700)
    hu_grid_red  = np.array([-1050, -1000, -808, -73, 0, 99, 1503, 2065, 4200, 6000, 7700])
    red_grid = np.array([0.001, 0.001, 0.236, 0.949, 1.0, 1.131, 1.781, 2.560, 3.930, 8.170, 9.280])
    red_array = np.interp(hu_array_red, hu_grid_red, red_grid)
    
    return red_array, density_array
