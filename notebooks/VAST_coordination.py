import numpy as np
from astropy import constants as co

def ra_dec_to_xyz(galaxy_table,
                  distance_metric='redshift',
                  h=0.7,
                  ):
    """
    Convert galaxy coordinates from ra-dec-redshift space into xyz space.
    
    
    Parameters
    ==========
    
    galaxy_table : astropy.table of shape (N,?)
        must contain columns 'ra' and 'dec' in degrees, and either 'Rgal' in who 
        knows what unit if distance_metric is 'comoving' or 'redshift' for 
        everything else
        
    distance_metric : str
        Distance metric to use in calculations.  Options are 'comoving' 
        (default; distance dependent on cosmology) and 'redshift' (distance 
        independent of cosmology).
        
    h : float
        Fractional value of Hubble's constant.  Default value is 1 (where 
        H0 = 100h).
        
        
    Returns
    =======
    
    coords_xyz : numpy.ndarray of shape (N,3)
        values of the galaxies in xyz space
    """
    
    
    if distance_metric == 'comoving':
        r_gal = galaxy_table['Rgal'].values        
    elif distance_metric == 'redshift':        
        r_gal = co.c.to('km/s').value*galaxy_table['redshift'].values/(100*h)
    else:
        raise ValueError (f"Distance metric {distance_metric} not recognized!")
        
        
    ra = galaxy_table['RA'].values
    
    dec = galaxy_table['DEC'].values
    
    ############################################################################
    # Convert from ra-dec-radius space to xyz space
    #---------------------------------------------------------------------------
    ra_radian = np.deg2rad(ra)
    
    dec_radian = np.deg2rad(dec)
    
    x = r_gal*np.cos(ra_radian)*np.cos(dec_radian)
    
    y = r_gal*np.sin(ra_radian)*np.cos(dec_radian)
    
    z = r_gal*np.sin(dec_radian)
    
    num_gal = x.shape[0]
    
    coords_xyz = np.concatenate((x.reshape(num_gal,1),
                                 y.reshape(num_gal,1),
                                 z.reshape(num_gal,1)), axis=1)
    ############################################################################
    
    return coords_xyz



def determine_vflag(x, y, z, voids, mask, mask_resolution, rmin, rmax):
    '''
    Determines whether or not a galaxy is a void, wall, edge, or unclassifiable
    galaxy.


    Parameters:
    ===========

    x : float
        x-position of object in units of Mpc/h

    y : float
        y-position of object in units of Mpc/h

    z : float
        z-position of object in units of Mpc/h

    voids : astropy table
        List of holes defining the void regions.  Columns consist of the center 
        of each hole (x,y,z) in units of Mpc/h, the hole radii, and the void 
        identifier for each hole.

    mask : numpy boolean array of shape (n,m)
        True values correspond to ra,dec coordinates which lie within the 
        survey footprint.

    rmin : float
        Minimum distance over which the voids were found.  Should be the same as 
        that used when running VoidFinder.  Units are Mpc/h.

    rmax : float
        Maximum distance over which the voids were found.  Should be the same as 
        that used when running VoidFinder.  Units are Mpc/h.


    Returns:
    ========

    vflag : integer
        0 = wall galaxy
        1 = void galaxy
        2 = edge galaxy (too close to survey boundary to determine)
        9 = outside survey footprint
    '''


    ############################################################################
    # INTRO CALCULATIONS, INITIALIZATIONS
    #---------------------------------------------------------------------------
    # Distance from galaxy to center of all voids
    distance_to_center = np.sqrt((voids['x'] - x)**2 + (voids['y'] - y)**2 + (voids['z'] - z)**2)
    
    # Boolean to find which void surrounds the galaxy, if any
    boolean = distance_to_center < voids['radius']
    ############################################################################


    ############################################################################
    # VOID GALAXIES
    #---------------------------------------------------------------------------
    if any(boolean):
        # The galaxy resides in at least one void
        vflag = 1
    ############################################################################
   
        
    ############################################################################
    # WALL GALAXIES
    #---------------------------------------------------------------------------
    else:
        # The galaxy does not live in any voids
        #-----------------------------------------------------------------------
        # Is the galaxy outside the survey boundary?
        #-----------------------------------------------------------------------
        coord_array = np.array([[x,y,z]])

        # Check to see if the galaxy is within the survey
        if not_in_mask(coord_array, mask, mask_resolution, rmin, rmax):
            # Galaxy is outside the survey mask
            vflag = 9

        else:
            # Galaxy is within the survey mask, but is not within a void
            vflag = 0
            #-------------------------------------------------------------------
            # Is the galaxy within 10 Mpc/h of the survey boundary?
            #-------------------------------------------------------------------
            # Calculate coordinates that are 10 Mpc/h in each Cartesian 
            # direction of the galaxy
            coord_min = np.array([x,y,z]) - 10
            coord_max = np.array([x,y,z]) + 10

            # Coordinates to check
            x_coords = [coord_min[0], coord_max[0], x, x, x, x]
            y_coords = [y, y, coord_min[1], coord_max[1], y, y]
            z_coords = [z, z, z, z, coord_min[2], coord_max[2]]
            extreme_coords = np.array([x_coords, y_coords, z_coords]).T

            i = 0
            while vflag == 0 and i <= 5:
                # Check to see if any of these are outside the survey
                if not_in_mask(extreme_coords[i].reshape(1,3), mask, mask_resolution, rmin, rmax):
                    # Galaxy is within 10 Mpc/h of the survey edge
                    vflag = 2
                i += 1
            #-------------------------------------------------------------------
        #-----------------------------------------------------------------------
    ############################################################################

    
    return vflag
