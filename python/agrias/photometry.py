import os
import importlib.resources
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from astropy import cosmology
from astropy import table
from astropy import units as u
from astropy import constants as co
from astropy.io import fits

from ekfplot import plot as ek
from ekfplot import colors as ec
from ekfphys import observer
from ekfstats import math, fit

from . import utils

cosmo = cosmology.FlatLambdaCDM(70.,0.3)
#band_restwl = 6563. * u.AA

def load_transmission (fname=None, band=None):
    if fname is None:
        if band is None:
            band = 'n708'
        fname = importlib.resources.files("agrias").joinpath(f"data/mer_{band}.txt")
    transmission = table.Table.read(
        fname,
        comment='#',
        format='ascii.basic',
        names=['wv','transmission_lambda'],
        units=[u.AA,None]    
    )#[::-1]
    transmission['freq'] = (co.c/transmission['wv']).to(u.Hz)
    transmission = transmission[np.argsort(transmission['wv'])]
    transmission['transmission_nu'] = transmission['transmission_lambda']/transmission['freq']**2
    return transmission

def mbestimate_halpha (
        n708data, 
        gdata,
        rdata, 
        idata,
        zdata, 
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
        u_ex_correction=0.,
        ge_correction=1.,
        ns_correction=1.,
        specflux_unit = None,
        filter_curve_file = None,
        zp = 31.4,
        band='n708',
        #wv_eff_mb = 7080.,
        #wv_rest_mb = 6563., 
        ctype='powerlaw',
        plawbands='griz',
        continuum_adjust=None
    ):
    if specflux_unit is None:
        # -2.5 log10(X/3631 Jy) = 27
        # log10(X/3631 Jy) = 27/-2.5
        # X/3631 Jy = 10^(27/-2.5)
        # X = 10^(27./-2.5) * 3631 Jy
        
        specflux_unit = 10.**(zp/-2.5) * 3631. * u.Jy
    
    transmission = load_transmission (filter_curve_file, band=band)
    wv_eff_mb = {'n708':7080., 'n540':5400.}[band]
    wv_rest_mb = {'n708':6563., 'n540':5007.}[band]
    line_restwl = wv_rest_mb * u.AA   
    
    if ctype == 'ri_avg':
        bandspecflux_continuum = (rdata + idata )/2. * specflux_unit
        #bandspecflux_continuum += bandspecflux_continuum*0.03 # XXX
    elif ctype == 'cubic_spline':
        #wv_eff = np.array([4809., 6229., 7703., 8906.]) # g, r, i, z effective wavelength from transmission curves
        wv_eff = np.array([6229., 7703., 8906.])
        #hscphot = np.array([gdata,rdata,idata,zdata]).T
        hscphot = np.array([rdata,idata,zdata]).T
        fy = interpolate.CubicSpline(
            wv_eff,
            hscphot,
            axis=1
        )
        bandspecflux_continuum = fy(wv_eff_mb) * specflux_unit 
    elif ctype == 'linear':
        wdict = {'g':4809.,'r':6229.,'i':7703.,'z':8906.}
        fdict = {'g':gdata,'r':rdata,'i':idata,'z':zdata}
        wv_eff = np.array( [ wdict[band] for band in plawbands] )
        lsq_x = wv_eff
        lsq_y = np.array([ fdict[band] for band in plawbands ])
        lsq_coeffs = fit.closedform_leastsq(lsq_x, lsq_y)
        #bandspecflux_continuum = 10.**(lsq_coeffs[0]+lsq_coeffs[1]*np.log10(7080.)).flatten() * specflux_unit
        bandspecflux_continuum = (lsq_coeffs[0]+lsq_coeffs[1]*wv_eff_mb).flatten() * specflux_unit        
    elif ctype == 'powerlaw':
        # \\ done via vectorized least squares
        # \\ i.e. Ax = b for A \in R^mxn, x \in n R^n, b \in R^mxp 
        # \\ which deviates from normal to compute all b
        # \\ x = (A^TA)^-1A^Tb
        #wv_eff = np.array([4809., 6229., 7703., 8906.]) # g, r, i, z effective wavelength from transmission curves
        wdict = {'g':4809.,'r':6229.,'i':7703.,'z':8906.}
        fdict = {'g':gdata,'r':rdata,'i':idata,'z':zdata}
        wv_eff = np.array( [ wdict[band] for band in plawbands] )
        lsq_x = np.log10(wv_eff)
        lsq_y = np.log10(np.array([ fdict[band] for band in plawbands ]))
        lsq_coeffs = fit.closedform_leastsq(lsq_x, lsq_y)
        bandspecflux_continuum = 10.**(lsq_coeffs[0]+lsq_coeffs[1]*np.log10(wv_eff_mb)).flatten() * specflux_unit
        
        #bandspecflux_continuum = (lsq_coeffs[0]+lsq_coeffs[1]*7080.).flatten() * specflux_unit
        #w = np.log10(wv_eff)   
        ##w_zp = w[0]     
        ##w = w- w_zp # \\ to begin at origin and satisfy Ax=b
        #w = w.reshape(-1,1)
        #
        #b = np.log10(np.array([
        #        gdata,
        #        rdata,
        #        idata,
        #        zdata
        #]))
        ##b_zp = b[0]
        ##b = b- b_zp      
        #        
        #leastsquares_soln = np.matmul(np.matmul(np.linalg.inv(np.matmul(w.T,w)),w.T),b)
        #
        #w_mer = np.log10(wv_eff_ha) #- w_zp
        ## 10.**(leastsquares_soln*(np.log10(7080.) - w_zp) + b_zp)
        #b_mer = w_mer*leastsquares_soln #+ b_zp
        #c = np.median(b- np.matmul(w,leastsquares_soln), axis=0)
        ##c_mer = np.median(b - b_mer,axis=0)
        #
        #bandspecflux_continuum = 10.**(b_mer+c).flatten() * specflux_unit
    
    if continuum_adjust is not None:
        print(f'[photometry.mbestimate_halpha] Ad-hoc adjustment to continuum fluxes of {continuum_adjust} mag!')
        adjust = 10.**(-0.4*continuum_adjust)
        bandspecflux_continuum *= adjust
        
    

    bandspecflux_line = n708data*specflux_unit - bandspecflux_continuum
    u_bandspecflux_line = np.sqrt((u_n708data*specflux_unit)**2 + 0.25 * (u_rdata*specflux_unit)**2 + 0.25 * (u_idata*specflux_unit)**2)
    
    bsf_lambda = observer.fnu_to_flambda( wv_eff_mb*u.AA, bandspecflux_line )
    u_bsf_lambda = observer.fnu_to_flambda( wv_eff_mb*u.AA, u_bandspecflux_line )
    #tc_integrated = math.trapz(transmission['transmission_nu']/(co.h*transmission['freq']), transmission['freq'])
    # = \int Tr(v)/(hv) dv
    tc_integrated = math.trapz(transmission['transmission_lambda'], transmission['wv'].value ) * transmission['wv'].unit
    trans_atline = np.interp(line_restwl*(1.+redshift), transmission['wv'], transmission['transmission_lambda'])

    haenergy = (co.h*co.c/(line_restwl*(1.+redshift))).to(u.erg)
    
    haflux = (bsf_lambda * tc_integrated / trans_atline ).to(u.erg/u.s/u.cm**2)
    u_haflux = (u_bsf_lambda * tc_integrated / trans_atline).to(u.erg/u.s/u.cm**2)
    #haflux = (bandspecflux_line * tc_integrated / ( trans_atline / haenergy )).to(u.erg/u.s/u.cm**2)
    #u_haflux = (u_bandspecflux_line * tc_integrated / ( trans_atline / haenergy )).to(u.erg/u.s/u.cm**2)
    
    haflux_forew = haflux.copy ()    
    u_haflux_forew = u_haflux.copy()
    
    # \\ 1'' aperture -> total flux correction,
    # \\ right now just approximated from i_cmodel / i_gaap1p0
    if do_aperturecorrection:
        #totcorr = merian_sources['i_cModelFlux_Merian'] / merian_sources['i_gaap1p0Flux_Merian']
        haflux *= apercorr
        u_haflux *= apercorr        
        fcontinuum_ac = bandspecflux_continuum * apercorr
    else:
        fcontinuum_ac = bandspecflux_continuum
        
    # \\ rough internal extinction correction assuming AV=0.5
    if do_extinctioncorrection:        
        # Fintr = c*Fobs
        # sigma^2(Fintr) = c**2 * sigma(Fobs)**2 + sigma(c)**2 * Fobs**2        
        u_haflux = np.sqrt((ex_correction*u_haflux)**2 + (u_ex_correction * haflux)**2)
        haflux *= ex_correction
        fcontinuum_ac *= ex_correction
        
    
    # \\ galactic extinction correction
    if do_gecorrection:
        haflux *= ge_correction
        u_haflux *= ge_correction
        fcontinuum_ac *= ge_correction

    # \\ apply Abby's other line corrections
    if do_linecorrection:
        haflux *= ns_correction
        u_haflux *= ns_correction
        
        haflux_forew *= ns_correction
        u_haflux_forew *= ns_correction
    
    wv_eff = wv_eff_mb*u.AA
    bandspecflux_continuum_wl = co.c * bandspecflux_continuum / wv_eff**2
    haew = (haflux_forew / bandspecflux_continuum_wl).to(u.AA)
    u_haew = (u_haflux_forew / bandspecflux_continuum_wl).to(u.AA)
    
    dlum = cosmo.luminosity_distance(redshift).to(u.cm)
    u_dlum = ((cosmo.luminosity_distance(0.09) - cosmo.luminosity_distance(0.07))/2.).to(u.cm) # XXX need to generalize this
    distance_factor = 4.*np.pi*dlum**2
    u_distance_factor = 8.*np.pi*dlum*u_dlum
    halum = haflux * distance_factor
    u_halum = np.sqrt((u_haflux * distance_factor)**2 + (haflux * u_distance_factor)**2)
    
    return (haflux, u_haflux), (halum, u_halum), (haew, u_haew), fcontinuum_ac.to(u.nJy)

def uvopt_gecorrection (merian_sources, av=None, rv=3.1, zphot='z500'):
    if av is None:
        av = merian_sources['ebv_Merian'] * rv
    wv_eff = np.array([1548.85, 2303.37, 7080., 5400.])
    ge_arr = np.zeros([len(merian_sources),wv_eff.size])
    for idx,(z,c_av) in enumerate(zip(merian_sources[zphot], av)):
        #wv_eff[2] = wv_eff.value * (1. + z)
        ge_arr[idx] = observer.gecorrection ( wv_eff, c_av, rv, return_magcorr=False)
    return ge_arr

def uvopt_internalextcorrection (merian_sources):
    restwl = np.array([1548.85, 2303.37, 7080., 5400.])

    dust_corr = np.zeros((len(merian_sources),3))
    for idx,av in enumerate(merian_sources['AV']):
        if hasattr(av,'mask'):
            dust_corr[idx] = np.NaN
        else:
            dust_corr[idx] = observer.extinction_correction ( restwl, av )[0].data
    return dust_corr

def naive_halpha_fromcatalog ( merian, galex, z_col='z500' ):        
    # \\ bone-headed halpha flux measurements
    merian_sources = table.Table.from_pandas(merian)
    
    #bandspecflux_continuum = (merian_sources[utils.photcols['r']] + merian_sources[utils.photcols['i']] )/2. * specflux_unit
    #bandspecflux_line = merian_sources[utils.photcols['N708']]*specflux_unit - bandspecflux_continuum

    #tc_integrated = np.trapz(transmission['transmission'][::-1]/(co.h*transmission['freq'][::-1]), transmission['freq'][::-1])
    #trans_atline = np.interp(line_restwl*(1.+merian_sources[z_col]), transmission['wv'], transmission['transmission'])

    # \\ estimate observed Halpha flux (i.e. no extinction correction, redshifting, etc. etc.)
    haflux = mbestimate_halpha(
        merian_sources[utils.photcols['N708']],
        merian_sources[utils.photcols['r']],
        merian_sources[utils.photcols['i']],
        merian_sources[z_col]
    ) 
    return merian_sources, haflux