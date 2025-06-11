
import numpy as np
from astropy import cosmology
from astropy import units as u
from agrias import utils, photometry

import reader

cosmo = cosmology.FlatLambdaCDM(70.,0.3)    

def do_mb_photometry (merian_source_catalog):
    ######### Various corrections #########
    valid = (merian_source_catalog['r_cModelFlux_Merian'] >0.)&(merian_source_catalog['Mr']<-16.)
    ms = merian_source_catalog[valid]
    zcorr = np.log10((cosmo.luminosity_distance(ms['z_spec'])/cosmo.luminosity_distance(ms['z500']))**2)


    fha_corrections = reader.compute_halphacorrections(ms, load_from_pickle=False, estimated_av=ms['AV_est'])
    emission_correction, ge_correction, extinction_correction, aperture_correction = fha_corrections

    ms['logmass_corrected'] = ms['logmass_gaap'] + zcorr + np.log10(aperture_correction)    
    
    ######### MB Photometry #########
    n708_fluxes, n708_luminosities, n708_eqws, n708_fcont = photometry.mbestimate_halpha(
        ms[utils.photcols['N708']].values,
        ms[utils.photcols['g']].values,
        ms[utils.photcols['r']].values,
        ms[utils.photcols['i']].values,
        ms[utils.photcols['z']].values,
        ms['z_spec'].values,
        ms[utils.u_photcols['N708']].values,
        0.,
        0.,
        apercorr=aperture_correction.values,
        ge_correction=ge_correction[:,2],
        ex_correction=extinction_correction[:,2],
        ns_correction=emission_correction[:],
        do_aperturecorrection=True,
        do_gecorrection=True,
        do_extinctioncorrection=True,
        do_linecorrection=True,
        specflux_unit=u.nJy,
        ctype='powerlaw',
        plawbands='riz'
    )
    
    return ms