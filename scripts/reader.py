import numpy as np
import pandas as pd

from astropy import table
from astropy import units as u
from astropy import constants as co
from astropy import cosmology
from astropy.io import fits

from ekfphys import observer

from agrias import utils as bu

cosmo = cosmology.FlatLambdaCDM(70.,0.3)

def merianselect ( merian, zmin=0.07, zmax=0.09, maglim=22., only_use=True, verbose=1, av=None ):
    mertab = merian.copy()#[[bu.merian_id, bu.merian_ra, bu.merian_dec, 'z_phot', 'i_cModelmag_Merian']]
    mertab.rename_column(bu.merian_ra,'RA')
    mertab.rename_column(bu.merian_dec,'DEC')
    inband = mertab['z_phot']>zmin
    inband &= mertab['z_phot']<zmax
    inband &= mertab['i_cModelmag_Merian']<maglim
    if verbose > 0:
        print(f'[merianselect] Only choosing sources at {zmin:.3f}<z_phot<{zmax:.3f}')
        print(f'[merianselect] Only choosing sources with i_cModelmag_Merian < {maglim:.1f}')
    
    
    mertab = mertab[inband].to_pandas ()
    mertab = mertab.set_index(bu.merian_id)
    mertab.index = [ 'M%i' % idx for idx in mertab.index ] 
    
    for band in 'grizy':
        mertab[f'{band}_gaap1p0FluxErr_aperCorr_Merian'] = mertab[f'{band}_gaap1p0FluxErr_Merian'] * mertab[f'{band}_gaap1p0Flux_aperCorr_Merian']/mertab[f'{band}_gaap1p0Flux_Merian']
        
    # \\ apply internal extinction corrections
    kcorr_i = observer.calc_kcor(
        'i',
        mertab['z_phot'],
        'g - i',
        -2.5*np.log10(mertab[bu.photcols['g']]/mertab[bu.photcols['r']])        
    )
    mertab['Mi'] = mertab['i_cModelmag_Merian'] - cosmo.distmod(mertab['z_phot'].values).value - kcorr_i
    gr = -2.5*np.log10(mertab[bu.photcols['g']]/mertab[bu.photcols['r']])
    if av is None:
        av = 0.42 # SAGAbg-A mean
    mertab['AV'] = av
    
    if only_use:
        if verbose > 0:
            print("[merianselect] only choosing sources with cmodel/gaap(i) > 1.3")
        extendedness = mertab['i_cModelFlux_Merian'] / mertab[bu.photcols['i']] > 1.3
        #imagcut = mertab['i_cModelmag_Merian'] < 23.
        use = extendedness#&imagcut
        mertab = mertab.loc[use]
    return mertab

def galexcrossmatch ( filename=None,  ):
    if filename is None:
        filename = '../local_data/inputs/MAST_Crossmatch_GALEX.csv'
        
    crossmatch = table.Table.read(
        filename, 
        format='csv', 
        comment='#', 
    )
    
    crossmatch.rename_column('Column0', bu.merian_id)
    crossmatch.add_index(bu.merian_id)
    crossmatch = crossmatch.to_pandas ()

    galex = crossmatch.loc[crossmatch.sort_values('nuv_exptime', ascending=False).index.duplicated(keep='first')]        
    return galex

def get_meriancrossgalex ():
    merian = table.Table(fits.getdata('../local_data/inputs/Merian_DR1_photoz_EAZY_v1.2.fits',1))
    ms = merianselect ( merian )
    _galex = galexcrossmatch ()
    overlap = ms.index.intersection(_galex.index)

    merian_sources = ms.reindex(overlap)

    _galex = _galex.sort_values('fuv_exptime', ascending=False)

    galex = _galex.loc[~_galex.index.duplicated(keep='first')].reindex(overlap).reset_index()
    return merian_sources, galex     

def galex_luminosities ( galex, redshifts, ge_arr, dust_corr ):
    uv_color = galex['fuv_mag'] - galex['nuv_mag']
    for idx,band in enumerate(['fuv','nuv']):
        uvflux = 10.**(galex['nuv_mag']/-2.5) * 3631. * u.Jy
        u_uvflux = 0.4*np.log(10.)*uvflux*galex['nuv_magerr'] 
        uvflux = uvflux.to(u.erg/u.s/u.cm**2/u.Hz)
        uvflux *= dust_corr[:,idx] # \\ internal extinction corrections
        u_uvflux *= dust_corr[:,idx]
        uvflux *= ge_arr[:,idx] # \\ galactic extinction correction
        u_uvflux *= ge_arr[:,idx]
        # \\ ignoring k-correction because it should be 0.01-0.05 mag in this redshift range
        #uvflux *= observer.calc_kcor(band.upper(), redshifts, 'FUV - NUV', uv_color )
        uvlum = (uvflux * 4.*np.pi * cosmo.luminosity_distance(redshifts).to(u.cm)**2).to(u.erg/u.s/u.Hz)   
        u_uvlum = (u_uvflux * 4.*np.pi * cosmo.luminosity_distance(redshifts).to(u.cm)**2).to(u.erg/u.s/u.Hz) 
        galex[f'{band}_flux_corrected']  = uvflux.value
        galex[f'u_{band}_flux_corrected']  = u_uvflux.value
        
        galex[f'{band}_luminosity'] = uvlum.value        
        galex[f'u_{band}_luminosity'] = u_uvlum.value
    return galex

def load_abbyver (
        merian_sources, 
        galex        
    ):
    linetable = pd.read_csv(
        '/Users/kadofong/work/projects/merian/local_data/cutouts/galex/haew.csv', 
        #units=[u.AA, u.Jy, None]
        index_col='objectId_Merian'
    )
    linetable = linetable.loc[~linetable.index.duplicated()]
    linetable = linetable.reindex(merian_sources.index)
    
    rv = 4.05
    wv_eff = np.array([1548.85, 2303.37, 7080.])
    ge_arr = np.zeros([len(galex),wv_eff.size])
    for idx,(z,av) in enumerate(zip(merian_sources['z_phot'].values, merian_sources['ebv_Merian'].values * rv)):
        ge_arr[idx] = observer.gecorrection ( wv_eff*(z+1.), av, rv, return_magcorr=False)

    z_phot = merian_sources['z_phot'].values
    wl_obs = 6563. * u.AA * ( 1. + z_phot )
    linetable['haflux'] = (linetable['haew'].values * u.AA * linetable['continuum_specflux'].values * u.Jy * co.c / wl_obs**2).to(u.erg/u.s/u.cm**2).value
    linetable['haflux'] = linetable['haflux'] * ge_arr[:,2]
    linetable['halum'] = linetable['haflux'] * 4.*np.pi * cosmo.luminosity_distance(z_phot).to(u.cm).value**2

    linetable['nuvflux'] = 10.**(galex['nuv_mag'].values/-2.5) * 3631. * u.Jy
    linetable['nuvflux'] = (linetable['nuvflux'].values  * ge_arr[:,1]).to(u.erg/u.s/u.cm**2/u.Hz).value
    linetable['nuvlum'] = (linetable['nuvflux'].values * u.erg/u.s/u.cm**2/u.Hz * 4.*np.pi * cosmo.luminosity_distance(z_phot).to(u.cm)**2).to(u.erg/u.s/u.Hz).value
    return linetable  