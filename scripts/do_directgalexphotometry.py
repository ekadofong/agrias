import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import table, coordinates
from astropy.io import fits
from astropy import units as u

from carpenter import pixels, conventions

from ekfplot import plot as ek
from ekfplot import colors as ec
from ekfparse import query
from ekfphot import photometry as ep

import reader

def load_sourcecatalog ():
    merian = table.Table(fits.getdata('../local_data/inputs/Merian_DR1_photoz_EAZY_v1.2.fits',1))
    mcoords = coordinates.SkyCoord( merian['coord_ra_Merian'], merian['coord_dec_Merian'], unit='deg')
    mcat = reader.merianselect(merian, av=0.3)
    return mcat    
    
def singleton ( row, savedir=None, merdir=None, cutout_size=None, verbose=2):
    if savedir is None:
        savedir = '../local_data/cutouts/galex/'
    if merdir is None:
        merdir = '../local_data/cutouts/merian/'
    if cutout_size is None:
        cutout_size = 30.*u.arcsec  
    #row = mcat.loc[mid]
    mid = row.name
    objname = conventions.produce_merianobjectname(ra=row.RA, dec=row.DEC, )
    
    target = coordinates.SkyCoord(row.RA, row.DEC, unit='deg')
    
    # \\ build AUTO aperture from N708 image
    bbmb = pixels.BBMBImage(galaxy_id=mid, distance=row.z_phot )
    band='N708'
    
    imfits = fits.open(f'{merdir}/N708/image/{objname}_N708_merim.fits')
    psf = fits.open(f'{merdir}/N708/psf/{objname}_N708_merpsf.fits')
    bbmb.add_band(band, target, cutout_size, imfits['IMAGE'], imfits['VARIANCE'], psf[0].data )
    bbmb.match_psfs(refband=band)
    emask, autoparams = bbmb.define_autoaper(band)   
    
    exitcode, manifest, names = query.download_galeximages ( row.RA, row.DEC, mid, savedir=savedir )
    gc_output = query.load_galexcutouts(mid, savedir, target, sw=cutout_size, sh=cutout_size, )
    gi = ep.GalexImaging(gc_output, filter_directory='../local_data/filters/') 
    
    galex_photometry = gi.do_ephotometry( (row.RA, row.DEC), autoparams )
    return galex_photometry

def main (verbose=2, overwrite=False):
    mcat = load_sourcecatalog ()
    savefile = '../local_data/output/galex_photometry.csv'
    if (not overwrite) and os.path.exists(savefile):
        direct_galex = pd.read_csv( savefile, index_col=0)
    else:
        direct_galex = pd.DataFrame ( index=mcat.index, columns = ['flux_fuv', 'u_flux_fuv', 'flux_nuv', 'u_flux_nuv'])
    ncomputed=0
    for name, row in mcat.iterrows ():
        if (not overwrite) and (not np.isnan(direct_galex.loc[name, 'flux_nuv'])):
            continue
        try:                        
            galex_photometry = singleton ( row, verbose=verbose )
            direct_galex.loc[name, 'flux_fuv'] = galex_photometry[0,0]
            direct_galex.loc[name, 'u_flux_fuv'] = galex_photometry[1,0]
            direct_galex.loc[name, 'flux_nuv'] = galex_photometry[0,1]
            direct_galex.loc[name, 'u_flux_nuv'] = galex_photometry[1,1] 
            ncomputed += 1
            if (ncomputed % 100) == 0:
                direct_galex.to_csv(savefile)
        except FileNotFoundError:
            if verbose>1:
                print(f'{name} image not found!')        
    direct_galex.to_csv(savefile)
                
        
if __name__ == '__main__':
    main ()