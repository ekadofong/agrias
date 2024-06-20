import os
if os.path.exists('/tigress/kadofong'):
    dirstem = '/tigress/kadofong/merian/'
    os.environ['MERCONT_HOME'] = '/tigress/kadofong/merian/packages/meriancontinuum/'
else:
    os.environ['MERCONT_HOME'] = '/Users/kadofong/work/projects/merian/meriancontinuum/'

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
from astropy import table
from astropy import coordinates
from astropy import units as u
from astropy.io import fits
import sep 

from carpenter import conventions, pixels

from ekfplot import plot as ek
from ekfstats import imstats as eis
from ekfparse import query
from ekfphot import photometry as ep

from meriancontinuum import fitting_utils
from agrias import photometry, utils
import reader

def read_catalogs():
    catfile = '../local_data/inputs/Merian_DR1_photoz_EAZY_v1.2.fits'
    merian = table.Table(fits.getdata(catfile,1))
    ms = reader.merianselect ( merian )
    
    print(f"<AV>_50 = {np.median(ms['AV']):.3f}")
    
    return ms

def observational_corrections (merian_sources):
    emission_correction   = fitting_utils.correct_N2_S3(
        merian_sources['z_phot'],
        merian_sources['logmass_gaap1p0']
    )**-1

    ge_arr = photometry.uvopt_gecorrection(merian_sources)

    extinction_correction = photometry.uvopt_internalextcorrection(
        merian_sources
    )    
    return emission_correction, ge_arr, extinction_correction

def singleton (
        merian_sources, 
        name, 
        dirname,
        emission_correction, 
        ge_correction, 
        extinction_correction,
        save_cutout=True,        
    ):
    row = merian_sources.loc[name]
    skyobj = coordinates.SkyCoord(row['RA'], row['DEC'], unit='deg')
    objname = conventions.produce_merianobjectname(skycoordobj=skyobj)    

    fnames = [
        #f'{dirname}/hsc/{objname}_HSC-g.fits',
        #f'{dirname}/merian/{objname}_N540_merim.fits',
        f'{dirname}/hsc/hsc_r/image/{objname}_HSC-r.fits',
        f'{dirname}/merian/N708/image/{objname}_N708_merim.fits',
        f'{dirname}/hsc/hsc_i/image/{objname}_HSC-i.fits',
        #f'{dirname}/hsc/{objname}_HSC-z.fits',
        #f'{dirname}/hsc/{objname}_HSC-y.fits',
    ]
    psfnames = [
        #f'{dirname}/hsc/{objname}_HSC-g_psf.fits',
        #f'{dirname}/merian/{objname}_N540_merpsf.fits',
        f'{dirname}/hsc/hsc_r/psf/{objname}_HSC-r_psf.fits',
        f'{dirname}/merian/N708/psf/{objname}_N708_merpsf.fits',
        f'{dirname}/hsc/hsc_i/psf/{objname}_HSC-i_psf.fits',
        #f'{dirname}/hsc/{objname}_HSC-z_psf.fits',
        #f'{dirname}/hsc/{objname}_HSC-y_psf.fits',
    ]
    #bands = ['g','N540','r','N708','i','z','y']
    bands = ['r','N708','i']    
    bbmb = pixels.BBMBImage ()
    
    cutout_size = 42.*u.arcsec
    for idx,band in enumerate(bands):
        imfits = fits.open(fnames[idx])
        hdunames = [ x.name for x in imfits ]
        psf = fits.open(psfnames[idx])
        if 'IMAGE' not in hdunames:
            imgextension = 1
        else:
            imgextension = "IMAGE"
            
        if 'VARIANCE' not in hdunames:
            varextension = 3
        else:
            varextension = "VARIANCE"            
            
        bbmb.add_band(band, skyobj, cutout_size, imfits[imgextension], imfits[varextension], psf[0].data )
    matched_image, matched_psf = bbmb.match_psfs (refband='N708') 
    cat, segmap = sep.extract(
        bbmb.matched_image['i'], 
        thresh=5., 
        var=bbmb.matched_var['i'], 
        segmentation_map=True,
        deblend_nthresh=64,
    )
    esegmap = eis.build_ellipsed_segmentationmap(cat, bbmb.matched_image['i'].shape)
    structures = ndimage.label(esegmap)[0]
    cid = eis.get_centerval(structures)
    mask = (structures == cid )|(structures==0)
    fn = lambda x: np.where(mask, x, np.NaN)
    isrow = np.in1d(merian_sources.index, row.name)

    (haflux, u_haflux), (halum, u_halum), (haew, u_haew) = photometry.mbestimate_halpha(
        fn(matched_image['N708']),
        fn(matched_image['r']),
        fn(matched_image['i']),
        row['z_phot'],
        u_n708data=fn(bbmb.var['N708']**.5),
        u_rdata=fn(bbmb.var['r']**.5),
        u_idata=fn(bbmb.var['i']**.5),
        do_aperturecorrection=False,
        do_extinctioncorrection=True,
        do_gecorrection=True,
        do_linecorrection=True,
        ge_correction=ge_correction[isrow,2],
        ex_correction=extinction_correction[isrow,2],
        ns_correction=emission_correction[isrow],
    )
    emask, catparams = bbmb.define_autoaper('i')
    ihalum = halum[emask].sum()    
    u_ihalum = np.sqrt((u_halum[emask]**2).sum())
    imag = -2.5*np.log10( matched_image['i'][emask].sum()) + 27. # \\ convert to AB mag for flux in HSC units
    n708mag = -2.5*np.log10( matched_image['N708'][emask].sum()) + 27.
    
    if save_cutout:
        imghdu = fits.PrimaryHDU(data=halum.value, header=bbmb.hdu['N708']) # erg/s/pixel
        imghdu.name = 'HALPHASB'
        var = fits.ImageHDU(data=u_halum.value, header=bbmb.hdu['N708'])        
        var.name = 'U_HALPHASB'
        mask = fits.ImageHDU(data=esegmap, header=bbmb.hdu["N708"])
        mask.name = 'SEGMAP'
        hdulist = fits.HDUList([imghdu, mask])
        hdulist.writeto(f'{dirname}/halpha/{objname}.fits', overwrite=True)
    
    return haflux, u_haflux, ihalum, u_ihalum, haew, u_haew, (imag, n708mag)
    
def main (savefile, overwrite=True):
    merian_sources = read_catalogs ()
    emission_correction, ge_correction, extinction_correction = observational_corrections ( merian_sources )
    if os.path.exists('/tigress/kadofong'):
        dirname = '/tigress/kadofong/merian/pixel_excess/local_data/cutouts/galex_MDR1'        
    else:
        dirname = '/Users/kadofong/work/projects/merian/agrias/local_data/cutouts/galex_MDR1'
        
    if not os.path.exists(f'{dirname}/halpha'):
        os.makedirs(f'{dirname}/halpha')
    
    if os.path.exists(savefile) and (not overwrite):
        lha_df =  pd.read_csv( savefile, index_col=0)
    else:
        lha_df = pd.DataFrame ( index=merian_sources.index, columns=['LHa', 'u_LHa', 'FHa','u_FHa','EWHa','u_EWHa','imag','n708mag'])
    
    nprocessed = 0
    for name in merian_sources.index:
        try:    
            haflux, u_haflux, ihalum, u_ihalum, haew, u_haew, (imag, n708mag) = singleton (
                merian_sources,
                name, 
                dirname,
                emission_correction, 
                ge_correction, 
                extinction_correction,
                save_cutout=True,  
            )
        
            lha_df.loc[name, 'FHa'] = haflux.value / 1e-15 # 10^-15 erg / s / cm^2 
            lha_df.loc[name, 'u_FHa'] = u_haflux.value / 1e-15 # 10^-15 erg / s / cm^2 
            lha_df.loc[name, 'LHa'] = ihalum.value / 1e40 # save in units of 10^40 erg / s
            lha_df.loc[name, 'u_LHa'] = u_ihalum.value / 1e40 # save in units of 10^40 erg/s
            lha_df.loc[name, 'EWHa'] = haew.value # save in units of 10^40 erg / s
            lha_df.loc[name, 'u_EWHa'] = u_haew.value # save in units of 10^40 erg/s            
            lha_df.loc[name, 'imag'] = imag
            lha_df.loc[name, 'n708mag'] = n708mag
            nprocessed += 1 
            if (nprocessed % 100) == 0:
                lha_df.to_csv(savefile)            
        except IOError:
            print(f'{name} not found in {dirname}!')

            
    return lha_df

if __name__ == '__main__':
    savefile = '../local_data/output/lha_df.csv'
    lha_df = main (savefile)
    lha_df.to_csv(savefile)