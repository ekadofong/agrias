import os
if os.path.exists('/tigress/kadofong'):
    dirstem = '/tigress/kadofong/merian/'
    os.environ['MERCONT_HOME'] = '/tigress/kadofong/merian/packages/meriancontinuum/'
else:
    os.environ['MERCONT_HOME'] = '/Users/kadofong/work/projects/merian/meriancontinuum/'

import numpy as np
import matplotlib.pyplot as plt
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
from burstiness import photometry, utils
import reader




def read_catalogs():
    catfile = '../local_data/inputs/Merian_DR1_photoz_EAZY_v1.2.fits'
    merian = table.Table(fits.getdata(catfile,1))
    ms = reader.merianselect ( merian )
    _galex = reader.galexcrossmatch ()
    overlap = ms.index.intersection(_galex.index)

    merian_sources = ms.reindex(overlap)

    _galex = _galex.sort_values('fuv_exptime', ascending=False)

    galex = _galex.loc[~_galex.index.duplicated(keep='first')].reindex(overlap).reset_index()
    return merian_sources, galex

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
        extinction_correction
    ):
    row = merian_sources.loc[name]
    skyobj = coordinates.SkyCoord(row['RA'], row['DEC'], unit='deg')
    objname = conventions.produce_merianobjectname(skycoordobj=skyobj)    

    fnames = [
        #f'{dirname}/hsc/{objname}_HSC-g.fits',
        #f'{dirname}/merian/{objname}_N540_merim.fits',
        f'{dirname}/hsc/{objname}_HSC-r.fits',
        f'{dirname}/merian/{objname}_N708_merim.fits',
        f'{dirname}/hsc/{objname}_HSC-i.fits',
        #f'{dirname}/hsc/{objname}_HSC-z.fits',
        #f'{dirname}/hsc/{objname}_HSC-y.fits',
    ]
    psfnames = [
        #f'{dirname}/hsc/{objname}_HSC-g_psf.fits',
        #f'{dirname}/merian/{objname}_N540_merpsf.fits',
        f'{dirname}/hsc/{objname}_HSC-r_psf.fits',
        f'{dirname}/merian/{objname}_N708_merpsf.fits',
        f'{dirname}/hsc/{objname}_HSC-i_psf.fits',
        #f'{dirname}/hsc/{objname}_HSC-z_psf.fits',
        #f'{dirname}/hsc/{objname}_HSC-y_psf.fits',
    ]
    #bands = ['g','N540','r','N708','i','z','y']
    bands = ['r','N708','i']    
    bbmb = pixels.BBMBImage ()
    
    cutout_size = 42.*u.arcsec
    for idx,band in enumerate(bands):
        imfits = fits.open(fnames[idx])
        psf = fits.open(psfnames[idx])
        bbmb.add_band(band, skyobj, cutout_size, imfits['IMAGE'], imfits['VARIANCE'], psf[0].data )
    matched_image, matched_psf = bbmb.match_psfs (refband='N708') 
    
    isrow = np.in1d(merian_sources.index, row.name)

    haflux, u_haflux, halum, u_halum = photometry.mbestimate_halpha(
        matched_image['N708'],
        matched_image['r'],
        matched_image['i'],
        row['z_phot'],
        u_n708data=bbmb.var['N708']**.5,
        u_rdata=bbmb.var['r']**.5,
        u_idata=bbmb.var['i']**.5,
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
    
    return ihalum, u_ihalum 
    
def main ():
    merian_sources, galex = read_catalogs ()
    emission_correction, ge_correction, extinction_correction = observational_corrections ( merian_sources )
    if os.path.exists('/tigress/kadofong'):
        dirname = '/tigress/kadofong/merian/pixel_excess/local_data/cutouts/galex_MDR1'        
    else:
        dirname = '/Users/kadofong/work/projects/merian/agrias/local_data/cutouts/galex_MDR1'
    
    lha_df = pd.DataFrame ( index=merian_sources.index, columns=['LHa', 'u_LHa'])
    for name in merian_sources.index:
        try:    
            ihalum, u_ihalum = singleton (
                merian_sources,
                name, 
                dirname,
                emission_correction, 
                ge_correction, 
                extinction_correction        
            )
        
            lha_df.loc[name, 'LHa'] = ihalum.value / 1e40 # save in units of 10^40 erg / s
            lha_df.loc[name, 'u_LHa'] = u_ihalum.value / 1e40 # save in units of 10^40 erg/s
        except IOError:
            print(f'{name} not found in {dirname}!')
            
    return lha_df

if __name__ == '__main__':
    lha_df = main ()
    lha_df.to_csv('../local_data/output/lha_df.csv')