import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import cosmology
from astropy import table
from astropy import units as u
from astropy import constants as co
from astropy.io import fits

from ekfplot import plot as ek
from ekfplot import colors as ec
from ekfphys import observer

from . import utils

cosmo = cosmology.FlatLambdaCDM(70.,0.3)
harestwl = 6563. * u.AA

def load_transmission (fname=None):
    if fname is None:
        fname = f"../local_data/filters/mer_n708.txt"    
    transmission = table.Table.read(
        fname,
        comment='#',
        format='ascii.basic',
        names=['wv','transmission'],
        units=[u.AA,None]    
    )[::-1]
    transmission['freq'] = (co.c/transmission['wv']).to(u.Hz)
    return transmission

def mbestimate_halpha (
        n708data, 
        rdata, 
        idata, 
        redshift,
        u_n708data,
        u_rdata = 0.,
        u_idata = 0.,
        do_aperturecorrection=True, 
        do_extinctioncorrection=True,
        do_gecorrection=True, 
        do_linecorrection=True,
        apercorr=1.,
        ex_correction=1.,
        ge_correction=1.,
        ns_correction=1.,
        specflux_unit = None,
        filter_curve_file = None,
    ):
    if specflux_unit is None:
        # -2.5 log10(X/3631 Jy) = 27
        # log10(X/3631 Jy) = 27/-2.5
        # X/3631 Jy = 10^(27/-2.5)
        # X = 10^(27./-2.5) * 3631 Jy
        specflux_unit = 10.**(27./-2.5) * 3631. * u.Jy
        
    transmission = load_transmission (filter_curve_file)
        
    bandspecflux_continuum = (rdata + idata )/2. * specflux_unit
    bandspecflux_line = n708data*specflux_unit - bandspecflux_continuum
    u_bandspecflux_line = np.sqrt((u_n708data*specflux_unit)**2 + 0.25 * (u_rdata*specflux_unit)**2 + 0.25 * (u_idata*specflux_unit)**2)

    tc_integrated = np.trapz(transmission['transmission'][::-1]/(co.h*transmission['freq'][::-1]), transmission['freq'][::-1])
    trans_atline = np.interp(harestwl*(1.+redshift), transmission['wv'], transmission['transmission'])

    haenergy = (co.h*co.c/(harestwl*(1.+redshift))).to(u.erg)
    haflux = (bandspecflux_line * tc_integrated / ( trans_atline / haenergy )).to(u.erg/u.s/u.cm**2)
    u_haflux = (u_bandspecflux_line * tc_integrated / ( trans_atline / haenergy )).to(u.erg/u.s/u.cm**2)
    
    haflux_forew = haflux.copy ()
    u_haflux_forew = u_haflux.copy()
    
    # \\ 1'' aperture -> total flux correction,
    # \\ right now just approximated from i_cmodel / i_gaap1p0
    if do_aperturecorrection:
        #totcorr = merian_sources['i_cModelFlux_Merian'] / merian_sources['i_gaap1p0Flux_Merian']
        haflux *= apercorr
        u_haflux *= apercorr        
        
    # \\ rough internal extinction correction assuming AV=0.5
    if do_extinctioncorrection:        
        haflux *= ex_correction
        u_haflux *= ex_correction
    
    # \\ galactic extinction correction
    if do_gecorrection:
        haflux *= ge_correction
        u_haflux *= ex_correction

    # \\ apply Abby's other line corrections
    if do_linecorrection:
        haflux *= ns_correction
        u_haflux *= ns_correction
        
        haflux_forew *= ns_correction
        u_haflux_forew *= ns_correction
    
    wv_eff = 7080*u.AA
    bandspecflux_continuum_wl = co.c * bandspecflux_continuum / wv_eff**2
    haew = (haflux_forew / bandspecflux_continuum_wl).to(u.AA)
    u_haew = (u_haflux_forew / bandspecflux_continuum_wl).to(u.AA)
    
    dlum = cosmo.luminosity_distance(redshift).to(u.cm)
    u_dlum = ((cosmo.luminosity_distance(0.09) - cosmo.luminosity_distance(0.07))/2.).to(u.cm) # XXX need to generalize this
    distance_factor = 4.*np.pi*dlum**2
    u_distance_factor = 8.*np.pi*dlum*u_dlum
    halum = haflux * distance_factor
    u_halum = np.sqrt((u_haflux * distance_factor)**2 + (haflux * u_distance_factor)**2)
    
    return (haflux, u_haflux), (halum, u_halum), (haew, u_haew)

def uvopt_gecorrection (merian_sources, av=None, rv=3.1):
    if av is None:
        av = merian_sources['ebv_Merian'] * rv
    wv_eff = np.array([1548.85, 2303.37, 0.])
    ge_arr = np.zeros([len(merian_sources),wv_eff.size])
    for idx,(z,c_av) in enumerate(zip(merian_sources['z_phot'], av)):
        wv_eff[2] = harestwl.value * (1. + z)
        ge_arr[idx] = observer.gecorrection ( wv_eff/(1. + z), c_av, rv, return_magcorr=False)
    return ge_arr

def uvopt_internalextcorrection (merian_sources):
    restwl = np.array([1548.85, 2303.37, 7080.])

    dust_corr = np.zeros((len(merian_sources),3))
    for idx,av in enumerate(merian_sources['AV']):
        if hasattr(av,'mask'):
            dust_corr[idx] = np.NaN
        else:
            dust_corr[idx] = observer.extinction_correction ( restwl, av )[0].data
    return dust_corr

def naive_halpha_fromcatalog ( merian, galex, z_col='z_phot' ):        
    # \\ bone-headed halpha flux measurements
    merian_sources = table.Table.from_pandas(merian)
    
    #bandspecflux_continuum = (merian_sources[utils.photcols['r']] + merian_sources[utils.photcols['i']] )/2. * specflux_unit
    #bandspecflux_line = merian_sources[utils.photcols['N708']]*specflux_unit - bandspecflux_continuum

    #tc_integrated = np.trapz(transmission['transmission'][::-1]/(co.h*transmission['freq'][::-1]), transmission['freq'][::-1])
    #trans_atline = np.interp(harestwl*(1.+merian_sources[z_col]), transmission['wv'], transmission['transmission'])

    # \\ estimate observed Halpha flux (i.e. no extinction correction, redshifting, etc. etc.)
    haflux = mbestimate_halpha(
        merian_sources[utils.photcols['N708']],
        merian_sources[utils.photcols['r']],
        merian_sources[utils.photcols['i']],
        merian_sources[z_col]
    ) 
    return merian_sources, haflux