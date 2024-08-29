import time
import pickle
import numpy as np
import pandas as pd
from astropy import table
from astropy import units as u
from astropy import constants as co
from astropy import cosmology
from astropy.io import fits

from ekfphys import observer
from ekfparse import query 
from ekfstats import fit

from agrias import utils as bu
from agrias import photometry


cosmo = cosmology.FlatLambdaCDM(70.,0.3)

def merianselect ( merian=None, zmin=0.06, zmax=0.1, maglim=22., only_use=True, verbose=1, av=None, zp=31.4, pmin=0.244 ):
    if merian is None:
        merian = table.Table(fits.getdata('../local_data/inputs/Merian_DR1_photoz_EAZY_v1.2.fits',1))
    mertab = merian.copy()#[[bu.merian_id, bu.merian_ra, bu.merian_dec, 'z_phot', 'i_cModelmag_Merian']]
    mertab.rename_column(bu.merian_ra,'RA')
    mertab.rename_column(bu.merian_dec,'DEC')
    inband_probability = merian['pz1']+merian['pz2']+merian['pz3']+merian['pz4']
    inband = inband_probability > pmin
    #inband = z_phot>zmin
    #inband &= z_phot<zmax
    n708mag = -2.5*np.log10(mertab[bu.photcols['N708']]) + zp
    mi = -2.5*np.log10(mertab['i_cModelFlux_Merian']) + zp
    #inband &= n708mag<maglim
    inband &= mi < maglim
    if verbose > 0:
        print(f'[merianselect] Only choosing sources at {zmin:.3f}<z_phot<{zmax:.3f}')
        print(f'[merianselect] Only choosing sources with m_i < {maglim:.1f}')
    
    mertab = mertab[inband].to_pandas ()
    mertab = mertab.set_index(bu.merian_id)
    mertab.index = [ 'M%i' % idx for idx in mertab.index ] 
    z_phot = mertab['z500'].values
    
    for band in 'grizy':
        mertab[f'{band}_gaap1p0FluxErr_aperCorr_Merian'] = mertab[f'{band}_gaap1p0FluxErr_Merian'] * mertab[f'{band}_gaap1p0Flux_aperCorr_Merian']/mertab[f'{band}_gaap1p0Flux_Merian']
        
    # \\ apply internal extinction corrections
    kcorr_g = observer.calc_kcor(
        'g',
        z_phot,
        'g - r',
        -2.5*np.log10(mertab[bu.photcols['g']]/mertab[bu.photcols['r']])        
    )     
    kcorr_r = observer.calc_kcor(
        'r',
        z_phot,
        'g - r',
        -2.5*np.log10(mertab[bu.photcols['g']]/mertab[bu.photcols['r']])        
    )    
    kcorr_i = observer.calc_kcor(
        'i',
        z_phot,
        'g - i',
        -2.5*np.log10(mertab[bu.photcols['g']]/mertab[bu.photcols['i']])        
    )
    mi = -2.5*np.log10(mertab['i_cModelFlux_Merian']) + zp
    mertab['Mi'] = mi - cosmo.distmod(z_phot).value - kcorr_i
    ri = -2.5*np.log10(mertab['r_gaap1p0Flux_aperCorr_Merian']/mertab['i_gaap1p0Flux_aperCorr_Merian'])
    gi = -2.5*np.log10(mertab['g_gaap1p0Flux_aperCorr_Merian']/mertab['i_gaap1p0Flux_aperCorr_Merian'])
    mertab['Mr'] = mertab['Mi'] + (ri - kcorr_r + kcorr_i)
    mertab['Mg'] = mertab['Mi'] + (gi - kcorr_g + kcorr_i)
    
    gr = -2.5*np.log10(mertab[bu.photcols['g']]/mertab[bu.photcols['r']])
    if av is None:
        #av = 0.42 # SAGAbg-A mean
        #saga_gr_av_coeffs = np.array([ 12.79771209, -22.34904904,  11.30434592,   0.33866297, -1.33162037])
        #logmstar = mertab['logmass_gaap1p0']
        #apercorr = mertab['i_cModelFlux_Merian'] / mertab['i_gaap1p0Flux_Merian']        
        #logmstar += np.log10(apercorr)
        #saga_logmstar_coeffs = np.array([ 0.35064268, -3.73081311])
        #av = 10.**np.poly1d(saga_logmstar_coeffs)(logmstar)
        
        saga_coeffs = np.load('../local_data/inputs/SAGA_Mr_gr_to_AV.npy')
        saga_u_coeffs = np.load('../local_data/inputs/SAGA_Mr_gr_to_u_AV.npy')
        n = len(saga_coeffs)
        deg = int((-3 + np.sqrt ( 9 - 4*(2-2*n) )) // 2 )
        
        av = 10.**fit.poly2d(mertab['Mr'], gr, saga_coeffs, deg )
        u_av = fit.poly2d(mertab['Mr'], gr, saga_u_coeffs, deg)
        u_av[av>4] = np.inf
        av[av>4] = np.NaN
    mertab['AV'] = av
    mertab['u_AV'] = u_av
    
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

def get_meriancrossgalex (merian=None):
    if merian is None:
        merian = table.Table(fits.getdata('../local_data/inputs/Merian_DR1_photoz_EAZY_v1.2.fits',1))
        merian = merianselect ( merian )
    _galex = galexcrossmatch ()
    overlap = merian.index.intersection(_galex.index)

    merian_sources = merian.reindex(overlap)

    _galex = _galex.sort_values('fuv_exptime', ascending=False)

    galex = _galex.loc[~_galex.index.duplicated(keep='first')].reindex(overlap)#.reset_index()
    return merian_sources, galex     

def compute_halphacorrections ( mcat, use_dustengine=True, load_from_pickle=True, verbose=1 ):
    '''
    
    '''
    import os
    os.environ['MERCONT_HOME'] = '/Users/kadofong/work/projects/merian/meriancontinuum/'
    from meriancontinuum import fitting_utils
    
    if verbose>0:
        start = time.time ()
    # \\ correct for other emission lines via Mintz+24
    emission_correction = fitting_utils.correct_N2_S3(
        mcat['z_phot'],
        mcat['logmass_gaap1p0']
    )**-1    
    if verbose>0:
        print(f'Computed line contamination in {time.time() - start:.1f} seconds.')
        start = time.time ()
        
    # \\ Galactic extinction correction
    if use_dustengine:
        if load_from_pickle:
            with open('../local_data/output/dustengine.pickle', 'rb') as f:
                dusteng = pickle.load(f)
        else:
            dusteng = query.DustEngine()
        direct_geav = mcat.apply(lambda row: dusteng.get_SandFAV(row['RA'], row['DEC']),axis=1) 
    else:
        rv = 3.1
        direct_geav = mcat['ebv_Merian'] * rv
       
    ge_correction = photometry.uvopt_gecorrection(mcat, av=direct_geav)
    if verbose>0:
        print(f'Computed Galactic extinction in {time.time() - start:.1f} seconds.')
        start = time.time ()    
    
    restwl = np.array([1548.85, 2303.37, 7080.])
    dust_correction = np.zeros((len(emission_correction),3))
    for idx,av in enumerate(mcat['AV']):
        if hasattr(av,'mask'):
            dust_correction[idx] = np.NaN
        else:
            dust_correction[idx] = observer.extinction_correction ( restwl, av, RV=4.05)[0].data
    if verbose>0:
        print(f'Computed internal extinction in {time.time() - start:.1f} seconds.')
        start = time.time ()            
            
    aperture_correction = mcat['i_cModelFlux_Merian'] / mcat['i_gaap1p0Flux_Merian']
    if verbose>0:
        print(f'Computed aperture correction in {time.time() - start:.1f} seconds.')
        #start = time.time ()    
    
    return emission_correction, ge_correction, dust_correction, aperture_correction
    
        

def galex_luminosities ( galex, redshifts, ge_arr=None, dust_corr=None ):
    if not isinstance(galex, table.Table):
        galex = table.Table.from_pandas(galex.reset_index())
        galex.add_index('index')
        
    uv_color = galex['fuv_mag'] - galex['nuv_mag']
    for idx,band in enumerate(['fuv','nuv']):
        uvflux = galex[f'flux_{band}'] * u.Jy
        u_uvflux = galex[f'u_flux_{band}'] * u.Jy
        uvflux = uvflux.to(u.erg/u.s/u.cm**2/u.Hz)
        if dust_corr is not None:
            uvflux *= dust_corr[:,idx] # \\ internal extinction corrections
            u_uvflux *= dust_corr[:,idx]
        if ge_arr is not None:
            uvflux *= ge_arr[:,idx] # \\ galactic extinction correction
            u_uvflux *= ge_arr[:,idx]
        else:
            print('Not doing GE Correction!')
        # \\ ignoring k-correction because it should be 0.01-0.05 mag in this redshift range
        #uvflux *= observer.calc_kcor(band.upper(), redshifts, 'FUV - NUV', uv_color )
        uvlum = (uvflux * 4.*np.pi * cosmo.luminosity_distance(redshifts).to(u.cm)**2).to(u.erg/u.s/u.Hz)   
        u_uvlum = (u_uvflux * 4.*np.pi * cosmo.luminosity_distance(redshifts).to(u.cm)**2).to(u.erg/u.s/u.Hz) 
        galex[f'flux_{band}_corrected']  = uvflux.value
        galex[f'u_flux_{band}_corrected']  = u_uvflux.value
        
        galex[f'L{band.upper()}'] = uvlum.value        
        galex[f'u_L{band.upper()}'] = u_uvlum.value
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