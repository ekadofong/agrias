import pickle
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
from astropy import cosmology

from ekfphys import observer
from ekfstats import sampling
from ekfplot import plot as ek
from ekfplot import colors as ec
from ekfplot import colorlists
from ekfparse import query

from agrias import utils, photometry
import sys
sys.path.append('../scripts')
import reader

cosmo = cosmology.FlatLambdaCDM(70.,0.3)

def main(ctype, read_from_scratch=True):
    ms = reader.merianselect()
    
    gama = query.load_gamacatalogs ()
    gama = gama.loc[gama['HA_EW']/gama['HA_EW_ERR'] > 1.]
    gama = gama.loc[gama['HA_EW']>0.]
    

    if read_from_scratch:
        with open('../local_data/fha_corrections.pickle', 'rb') as f:
            fha_corrections = pickle.load(f)
    else:
        fha_corrections = reader.compute_haelphacorrections(ms, load_from_pickle=True)
        with open('../local_data/fha_corrections.pickle','wb') as f:
            pickle.dump(fha_corrections, f)

    emission_correction, ge_correction, extinction_correction, aperture_correction = fha_corrections    
    
    zgrid = np.arange(0.06,0.1,0.005)
    emission_correction = np.zeros([len(zgrid), len(ms)])
    for idx,z in enumerate(zgrid):
        emission_correction[idx] = reader.correct_N2_S3(
                np.ones(len(ms))*z,
                ms['logmass_gaap1p0'] + np.log10(aperture_correction)
        )**-1 
    emission_correction = np.mean(emission_correction,axis=0)    
    
    redshifts = ms['z500'].values
    
    for key,do_linecorrection in zip(('ha','hablend'),(True,False)):
        fluxes, luminosities, eqws, fcont = photometry.mbestimate_halpha(
            ms[utils.photcols['N708']].values,
            ms[utils.photcols['g']].values,
            ms[utils.photcols['r']].values,
            ms[utils.photcols['i']].values,
            ms[utils.photcols['z']].values,
            redshifts,
            #np.ones_like(ms['z500'].values)*0.08,#.value.data,
            ms[utils.u_photcols['N708']].values,
            ms[utils.u_photcols['r']].values,
            ms[utils.u_photcols['i']].values,    
            apercorr=aperture_correction.values,
            ge_correction=ge_correction[:,2],
            ex_correction=extinction_correction[:,2],
            ns_correction=emission_correction[:],
            do_aperturecorrection=True,
            do_gecorrection=False,
            do_extinctioncorrection=False,
            do_linecorrection=do_linecorrection,
            specflux_unit=u.nJy,
            ctype=ctype
        )
        haflux, u_haflux = fluxes 
        halum, u_halum = luminosities
        haew, u_haew = eqws  
        
        ms[f'{key}ew'] = haew
        ms[f'u_{key}ew'] = u_haew
        ms[f'{key}flux'] = haflux#/1e-17
        ms[f'u_{key}flux'] = u_haflux#/1e-17
        ms['fcont'] = fcont.value#/1e-17
        ms['apercorr'] = aperture_correction
        
    lm = (ms['z500']>0.06)&(ms['z500']<0.1)
    ms_match, gama_match = query.match_catalogs(ms.loc[lm], gama, coordkeysA=['RA','DEC'] )



    gama_match['HA_FLUX'] = gama_match['HA_FLUX']*1e-17
    gama_match['HA_FLUX_ERR'] = gama_match['HA_FLUX_ERR']*1e-17


    gama_match['fcont'] = gama_match['HA_FLUX'] / gama_match['HA_EW'] 
    gama_match['HABLEND_FLUX'] = gama_match['HA_FLUX'] + gama_match['NIIR_FLUX']*1e-17 + gama_match['NIIB_FLUX']*1e-17
    gama_match['HABLEND_FLUX_ERR'] = np.sqrt(gama_match['HA_FLUX_ERR']**2 +\
                                            (gama_match['NIIR_FLUX_ERR']*1e-17)**2 +\
                                            (gama_match['NIIB_FLUX_ERR']*1e-17)**2)

    gama_match['HABLEND_EW'] = gama_match['HABLEND_FLUX'] / gama_match['fcont']
    gama_match['HABLEND_EW_ERR'] = gama_match['HABLEND_FLUX_ERR'] / gama_match['fcont']


    survey_recode = np.zeros(gama_match.shape[0])
    survey_recode[gama_match['SURVEY_CODE']==1] = 1
    survey_recode[gama_match['SURVEY_CODE']==5] = 2

    ##### FIGURE
    fig, axarr = plt.subplots(1,2,figsize=(10,4))
    cms = ec.colormap_from_list([colorlists.slides['red'], colorlists.slides['yellow'],colorlists.slides['blue']])

    ratio = np.log10(observer.flambda_to_fnu(np.ones(len(gama_match))*7080.*u.AA, gama_match['fcont'].values*u.erg/u.s/u.cm**2/u.AA).to(u.nJy).value)
    ratio -= np.log10(ms_match.loc[:,'fcont'])
    ratio = ratio.values
    ax = axarr[0]
    mask = survey_recode > 0 
    x = observer.flambda_to_fnu(np.ones(len(gama_match))*7080.*u.AA, gama_match['fcont'].values*u.erg/u.s/u.cm**2/u.AA).to(u.nJy)[mask]
    y = ms_match.loc[mask,'fcont']

    im = ax.scatter(
        x,y,
        s=3,
        c=ratio[mask],
        vmax=0.5,
        vmin=-0.5,
        cmap = cms
    )
    xs = np.logspace(4,7)
    ax.plot(xs,xs,color='k')
    fcratio = np.log10(x/y)


    plt.colorbar(im ,ax=ax,label=r'$m_{r,\rm GAMA} - m_{r,\rm Mer}^{\rm apercorr}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(rf'$F_{{\nu}}(\lambda_{{\rm H\alpha}}$), GAMA (nJy)')
    ax.set_ylabel(rf'$F_{{\nu}}(\lambda_{{\rm H\alpha}}$), Merian (nJy)')

    bx = axarr[1]
    sclipped = stats.sigmaclip(fcratio, low=2.5, high=2.5, )
    ax.plot(xs, 10.**sclipped.upper * xs,  color='k', ls=':' )
    ax.plot(xs, 10.**sclipped.lower * xs, color='k', ls=':' )

    _,(_,bins,_) = ek.hist(fcratio, histtype='step', color=ec.ColorBase(colorlists.slides['blue']).modulate(-0.1).base, ax=bx, lw=3, bins=30)
    ek.hist(sclipped.clipped, color=colorlists.slides['blue'], ax=bx, bins=bins)

    for idx,pct in enumerate([0.05, 0.16, 0.5, 0.84, 0.95]):
        vx = np.nanquantile( sclipped.clipped, pct )
        vy = (bx.get_ylim()[1] - bx.get_ylim()[0])*0.05 + bx.get_ylim()[0]
        #bx.vlines ( vx, 0., 0.5, color=maincolor, lw = [3,2,1][abs(2-idx)], ls='--')
        ek.text(
            0.975,
            0.975 - 0.075*idx,
            fr'$\langle  \mathcal{{R}} \rangle_{{{pct:.2f}}}={vx:.2f}$',
            ax=bx,
            color=colorlists.slides['blue'],
            
        )

    plt.tight_layout ()
    plt.savefig(f'../figures/fnucontinuum_{ctype}.png')    
    
    
    ##### FIGURE 2
    fig, axarr = plt.subplots(2,4, figsize=(18,8))

    linelabels = {'ha':r'H$\alpha$','hablend':r'H$\alpha$+[NII]'}
    metriclabels=['EW','F']
    metricbounds=[(1.,300),(1e-16,1e-13)]

    for cindex,name in enumerate(['ha','hablend']):
        colnames = [(f'{name.upper()}_EW',f'{name.lower()}ew'),(f'{name.upper()}_FLUX',f'{name.lower()}flux')]
        maincolor = [colorlists.hcbold['blue'], colorlists.hcbold['red']][cindex]
        for mindex in range(2):
            ax_oo = axarr[cindex,mindex*2]
            ax_hist = axarr[cindex,mindex*2+1]

            if mindex==1: # \\ other surveys not flux-calibrated in GAMA
                mask = (survey_recode>0)&(abs(ratio)<0.25)
            else:
                mask = abs(ratio)<0.25

            true = gama_match.loc[mask,colnames[mindex][0]].values
            obs = ms_match.loc[mask,colnames[mindex][1]].values

            # \\ one:one scatter
            ax_oo.scatter(
                true,
                obs,
                color=maincolor,
                s=3
            )
            ax_oo.set_xlabel(rf'{metriclabels[mindex]}({linelabels[name]})$_{{\rm GAMA}}$')
            ax_oo.set_ylabel(rf'{metriclabels[mindex]}({linelabels[name]})$_{{\rm Merian}}$')
            ek.loglog(ax=ax_oo)

            xs = np.logspace(*np.log10(metricbounds[mindex]))
            ax_oo.plot(xs,xs,color='k')
            # \\ bestfit
            ax_oo.set_xlim(*metricbounds[mindex])
            ax_oo.set_ylim(*metricbounds[mindex])
            # \\ histogram

            fractional_err = (obs-true)/true
            rbins = np.linspace(-2.,2.,40)
            ek.hist(fractional_err, ax=ax_hist, alpha=0.4, lw=3, bins=rbins, cumulative=False, density=True, color=maincolor)

            for idx,pct in enumerate([0.05, 0.16, 0.5, 0.84, 0.95]):
                vx = np.nanquantile( sampling.fmasker(fractional_err), pct )
                vy = (ax_hist.get_ylim()[1] - ax_hist.get_ylim()[0])*0.05 + ax_hist.get_ylim()[0]
                ax_hist.vlines ( vx, 0., 0.5, color=maincolor, lw = [3,2,1][abs(2-idx)], ls='--')
                ek.text(
                    0.975,
                    0.975 - 0.075*idx,
                    fr'$\langle  \mathcal{{R}} \rangle_{{{pct:.2f}}}={vx:.2f}$',
                    ax=ax_hist,
                    color=maincolor,
                    
                )
            dt = np.nanquantile( sampling.fmasker(fractional_err), .84 ) - np.nanquantile( sampling.fmasker(fractional_err), 0.16 )
            ek.text(
                0.025,
                0.975,
                rf'$\delta={dt:.2f}$',
                ax=ax_hist,
                color=maincolor
            )
                
                
            ax_hist.set_xlabel(f'FE[{metriclabels[mindex]}({linelabels[name]})]')

    ek.text(0.025,0.975,ctype,ax=axarr[0,0])
    plt.tight_layout ()
    plt.savefig(f'../figures/lineestimates_{ctype}.png')
    
    plt.close('all')
    
if __name__ == '__main__':
    print('powerlaw...')
    main('powerlaw')
    print('cubic spline...')
    main('cubic_spline')
    print('r-i average...')
    main('ri_avg')