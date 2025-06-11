import pickle 

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd

from astropy import table
from astropy import wcs
from astropy import units as u
from astropy.io import fits
from astropy import cosmology
from astropy.modeling.functional_models import Sersic2D

import progressbar
import sep

from ekfplot import plot as ek
from ekfplot import colors as ec
from ekfstats import sampling, fit, functions

from agrias import photometry, utils

import reader

cosmo = cosmology.FlatLambdaCDM(70.,0.3)


def inject ( cutout, amplitude, r_eff, n=1.,  return_im=False ):
    Y,X = np.mgrid[:cutout.shape[0],:cutout.shape[1]]
    x_0 = cutout.shape[1]/2.
    y_0 = cutout.shape[0]/2.
    source_model = Sersic2D(
        amplitude=amplitude,
        r_eff=r_eff, 
        n=n,
        x_0 = x_0,
        y_0 = y_0,
    )
    source = source_model(X,Y)
    true_flux = source.sum()
    observed = source + np.random.choice(cutout.flatten(), cutout.shape)
    err = np.random.choice(cutout.flatten(), cutout.shape)
    if return_im:
        return observed
    
    cat = table.Table(sep.extract( observed, 5, err=cutout, minarea=10))
    if len(cat) < 1:
        return (np.NaN,true_flux),(np.NaN,x_0),(np.NaN,y_0)
    elif len(cat)>1:
        #return cat, observed
        cat = cat[np.argmax(cat['flux'])]
    else:
        cat = cat[0]
    
    obs_flux = cat['flux']
    obs_x = cat['x']
    obs_y = cat['y']

    return (obs_flux,true_flux),(obs_x, x_0),(obs_y,y_0)

def double_sigmoid (x,a0,bound0,k0,a1,bound1,k1):
    return 1. - (a0*functions.sigmoid(x,bound0,k0) + a1*functions.sigmoid(x,bound1,k1))    

def do_test ( ):
    ######### LOAD image #########
    im = fits.open('/Users/kadofong/Downloads/cutout_159.0697_-1.4446.fits')
    image = im[0].data
    imwcs = wcs.WCS(im[0])
    pixscale = (imwcs.pixel_scale_matrix[1,1]*u.deg).to(u.arcsec) / u.pix

    raw_std = (image[200:250,:50]).std()
    cutout = image[200:250,:50].byteswap().newbyteorder()

    Y,X = np.mgrid[:cutout.shape[0],:cutout.shape[1]]

    
    seeing = 1.1*u.arcsec
    pixseeing = seeing/ pixscale
    area_seeing = np.pi*pixseeing.value**2 * u.pix
    zp = 27. # ???
    
    cdf = pd.DataFrame(columns=['amplitude','r_eff','obs_flux','true_flux','x_obs','y_obs'], dtype=float)
    idx = 0 

    r_eff = pixseeing.value
    npull = 500
    namp = 40

    pbar = progressbar.ProgressBar(maxval=npull*namp)
    pbar.start()
    for amp in np.logspace(np.log10(0.003),np.log10(0.3),namp):
        for _ in range(npull):
            out = inject(cutout, amp, r_eff)
            cdf.loc[idx,'amplitude'] = amp
            cdf.loc[idx,'r_eff'] = r_eff
            cdf.loc[idx,'obs_flux'] = out[0][0]
            cdf.loc[idx,'true_flux'] = out[0][1]
            cdf.loc[idx,'x_obs'] = out[1][0]
            cdf.loc[idx,'y_obs'] = out[2][0]
            idx+=1
            pbar.update(idx)

    cdf['r_sep'] = np.sqrt((cdf['x_obs']-24.5)**2 + (cdf['y_obs']-24.5)**2)*pixscale.value
    cdf['re_flux'] = (cdf['obs_flux']-cdf['true_flux'])/cdf['true_flux']
    miss = (abs(cdf['r_sep'])>r_eff)|(abs(cdf['re_flux'])>2.)
    cdf.loc[miss,['obs_flux','x_obs','y_obs','re_flux']] = np.NaN

    agroups = cdf.groupby('amplitude')

    akeys = list(agroups.groups.keys())
    summary = pd.DataFrame(index=akeys, columns=['n708_mag', 'f_detect','re_flux_16','re_flux_50','re_flux_84'], dtype=float)
    for akey in akeys:
        grp = agroups.get_group(akey)    
        summary.loc[akey, 'f_detect'] = np.isfinite(grp['obs_flux']).sum()/len(grp)
        summary.loc[akey, ['re_flux_16','re_flux_50','re_flux_84']] = np.nanquantile(grp['re_flux'],[.16,.5,.84])
        summary.loc[akey,'n708_mag'] = -2.5*np.log10(grp['true_flux'].values[0]) + zp
        ci = sampling.proportion_confint(np.isfinite(grp['obs_flux']).sum(),len(grp), alpha=0.32, method='jeffreys')
        summary.loc[akey,'u_f_detect'] = ci[1]-ci[0]
        
    return summary

def qafig(summary, bi):
    ek.errorbar(
        summary['n708_mag'],
        summary['f_detect'],
        yerr=summary['u_f_detect']
    )
    ms = np.linspace(23., 28.)
    bi.plot_uncertainties(ms)

    plt.xlabel(r'$m_{N708}$')
    plt.ylabel(r'$f_{\rm detect}$') 
    plt.tight_layout()
    plt.savefig('../figures/crude_photcomplete.png')   

def main ():
    summary = do_test ()
    
    bi = fit.BaseInferer()
    bi.set_predict ( double_sigmoid )
    lnP = bi.define_gaussianlikelihood(bi.predict, with_intrinsic_dispersion=False)
    bi.set_loglikelihood(lnP)
    bi.set_uniformprior([[0.,1.],[23.,27.],[0.,10.],[0.,1.],[23.,27],[0.,10.]])
    data = (summary['n708_mag'].values, summary['f_detect'].values, summary['u_f_detect'].values, None)    
    bi.run(data,progress=False)
    qafig(summary, bi)
    
    np.savetxt('../local_data/output/pd_given_mn708.dat', bi.get_param_estimates()[1])
    #with open('../local_data/output/pd_given_mn708.pkl', 'wb') as f:
    #    pickle.dump(bi, f)

if __name__ == '__main__':
    main ()