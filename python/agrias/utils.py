import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec,patches
from astropy.io import fits
from astropy import units as u
from astropy import constants as co
from astropy import cosmology
from astropy import table
from sklearn.neighbors import NearestNeighbors

import sep
from astrodendro import Dendrogram

from ekfstats import sampling, functions, imstats, fit
from ekfplot import plot as ek
from ekfphys import observer 
from ekfplot import colors as ec


from SAGAbg.utils.calc_kcor import calc_kcor



cosmo = cosmology.FlatLambdaCDM(70.,0.3)

photcols = {
    'N708':'N708_gaap1p0Flux_Merian',
    'N540':'N540_gaap1p0Flux_Merian',
    'g':'g_gaap1p0Flux_aperCorr_Merian',
    'r':'r_gaap1p0Flux_aperCorr_Merian',
    'i':'i_gaap1p0Flux_aperCorr_Merian',
    'z':'z_gaap1p0Flux_aperCorr_Merian',
}
u_photcols={
    'N708':'N708_gaap1p0FluxErr_Merian',
    'N540':'N540_gaap1p0FluxErr_Merian',
    'g':'g_gaap1p0FluxErr_aperCorr_Merian', # \\ added on in post-processing
    'r':'r_gaap1p0FluxErr_aperCorr_Merian',
    'i':'i_gaap1p0FluxErr_aperCorr_Merian',
    'z':'z_gaap1p0FluxErr_aperCorr_Merian',
}
merian_ra = 'coord_ra_Merian'
merian_dec = 'coord_dec_Merian'
merian_id = 'objectId_Merian'


# fit from Kado-Fong+2022
logml = lambda gr: 1.65*gr - 0.66
def CM_msun ( Mg, Mr, zp_g = 5.11 ):
    loglum_g = (Mg-zp_g)/-2.5
    logsmass = logml(Mg-Mr) + loglum_g
    return logsmass

def estimate_stellarmass ( gmag, rmag, z ):
    distmod = cosmo.distmod ( z ).value
    gminusr = gmag - rmag
    kcorrect_g = calc_kcor ( 'g', z, 'gr', gminusr )
    kcorrect_r = calc_kcor ( 'r', z, 'gr', gminusr )
    Mg = gmag - distmod - kcorrect_g
    Mr = rmag - distmod - kcorrect_r
    logmstar = CM_msun ( Mg, Mr )
    return logmstar, Mr
    

def mf_2pcf (results, bins, labels=None):
    fig = plt.figure ( figsize=(12,5) )
    grid = gridspec.GridSpec ( len(results)-2, 2, figure=fig )
    ax = fig.add_subplot ( grid[:,0], )
    bxarr = [ fig.add_subplot(grid[idx,1]) for idx in range(len(results)-2)]

    cmap = ec.ColorBase('tab:red').sequential_cmap("tab:blue", reverse=False)
    clist = [ ec.ColorBase(cmap(x)).base for x in np.linspace(0.,1.,len(results)-1)] + ['k']
    for idx in range(1, len(results)):
        corr, u_corr, _ = results[idx]
        is_subset = idx < (len(results)-1)
        if labels is not None:
            label = labels[idx-1]
            #if is_subset:
            #    label = labels[idx#f'{lhabins[idx]-1:.2f}<logLHa<{lhabins[idx]:.2f}'
            #else:
            #    label = 'all'
        else:
            label = None
                
        #ax = axarr[0]
        ax.plot ( 
            sampling.midpts(bins), 
            corr,
            color = clist[idx],
            label = label
            #yerr = u_corr,
        )
        ax.fill_between (
            sampling.midpts(bins),
            corr - u_corr,
            corr + u_corr,
            alpha=0.1,
            color = clist[idx]
        )

        
        if is_subset:
            bx = bxarr[idx-1]

            ek.errorbar ( 
                sampling.midpts(bins) + idx*0.01,
                corr / results[-1][0],
                color = clist[idx],
                yerr = u_corr / results[-1][0],
                capsize = 3,
                ax=bx            
            )

            bx.axhline(1.,color=clist[idx],ls='--')
            #bx.set_ylim(0.5,2.)
        bx.set_xscale('log')

    ax.legend ()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel ( r'$\theta$ [deg]')
    ax.set_ylabel ( r'$\hat{w}(\theta)$')
    return fig, [ax] + bxarr

def get_catalog_lines ( cat, filtername  ):
    rfilter = {'N708':'i','N540':'r'}[filtername]
    bfilter = {'N708':'r','N540':'g'}[filtername]
    
    hamb = cat[f'{filtername}_gaap1p0Flux_Merian']
    avgcont = (cat[f'{rfilter}_gaap1p0Flux_aperCorr_Merian']+cat[f'{bfilter}_gaap1p0Flux_aperCorr_Merian'])/2.
    
    u_hamb = cat[f'{filtername}_gaap1p0FluxErr_Merian']
    u_avgcont = np.sqrt(cat[f'{rfilter}_gaap1p0FluxErr_Merian']**2 + cat[f'{bfilter}_gaap1p0FluxErr_Merian']**2)/2.

    return (hamb, u_hamb), (avgcont, u_avgcont)

def get_halpha_phys ( halpha_flux ):
    distance_factor = 4.*np.pi*cosmo.luminosity_distance(0.07).to(u.cm)**2
    halpha_luminosity = (halpha_flux * distance_factor).to(u.erg/u.s)
    halpha_sfr = halpha_luminosity.value * 5.5e-42
    return halpha_luminosity, halpha_sfr

def get_filterdescription (filtername):
    ldict = {'N708':7080., 'N540':5040.}
    wdict = {'N708':275., 'N540':210.}
    lambda_c = ldict[filtername] * u.AA
    dlambda = wdict[filtername] * u.AA
    wv2nu = lambda wv: (co.c/wv).to(u.Hz)
    nu_c = wv2nu(lambda_c)

    lambda_edges = (lambda_c + np.array([1.,-1.])*dlambda)
    nu_edges = wv2nu(lambda_edges)
    return nu_edges 
   
def merian_facecolor ( idx, xbins ):
    inband = (xbins[idx]>0.055)&(xbins[idx]<0.11)    
    inaux  = (xbins[idx]>0.4)&(xbins[idx]<0.45) 
    if inband:
        return ec.ColorBase('tab:red').modulate(0.2).base
    #elif inaux:
    #    return ec.ColorBase('tab:blue').modulate(0.2).base
    else:
        return 'lightgrey'
    
def merian_edgecolor ( idx, xbins ):
    inband = (xbins[idx]>0.055)&(xbins[idx]<0.11)    
    inaux  = (xbins[idx]>0.4)&(xbins[idx]<0.45)     
    if inband:
        return ec.ColorBase('tab:red').modulate(-0.1,0.3).base
    #elif inaux:
    #    return ec.ColorBase('tab:blue').modulate(-0.1, 0.3).base
    else:
        return 'grey'
    
def load_merianclean ( merian ):
    imag = -2.5*np.log10(merian['i_cModelFlux_Merian']) + 31.4
    mer_clean = merian[(imag>18.)&(imag<24.)] 
    inband_zphot = mer_clean['z_phot'] < 0.15
    inband_zspec = mer_clean['z_spec'] < 0.15
    mer_clean['z_is_good'] = inband_zphot&inband_zspec
    gminusr = -2.5*np.log10(mer_clean['g_gaap1p0Flux_aperCorr_Merian']/mer_clean['r_gaap1p0Flux_aperCorr_Merian'])
    mer_clean['gminusr'] = gminusr
    return mer_clean

def mk_photozperformance ( mer_clean, show_index = 0 ): 
    fig, axarr = plt.subplots(1,2,figsize=(9,4.5))
    lims = np.array([0.01,0.35])
    xbins = np.arange(*lims, 0.015)
    ybins = np.linspace(0.0, 0.4, 100)

    ax = axarr[0]
    _=ek.histstack (
        *functions.finite_masker([mer_clean['z_spec'], mer_clean['z_phot']], inplace=True), 
        xbins=xbins, 
        ybins=ybins,
        facecolor = lambda x: merian_facecolor(x,xbins),
        edgecolor = lambda x: merian_edgecolor(x,xbins),
        show_quantile=False,
        quantile_kwargs={'color':'k', 's':1},
        ax=ax
    )


    ax.plot( lims, lims, color='k', ls=':', zorder=-1 )
    ax.set_xlabel('spec-z')
    ax.set_ylabel('phot-z')
    ax.set_xlim(lims)
    ax.set_ylim(ybins[0], ybins[-1])

    # \\ show outlier fraction
    assns = np.digitize ( mer_clean['z_spec'], xbins )
    zstats = np.zeros([len(xbins)-1, 3])
    dev = (mer_clean['z_phot'] - mer_clean['z_spec'])/(1.+mer_clean['z_spec'])
    for bin_index in np.arange(1, xbins.size):
        mask = assns == bin_index
        zstats[bin_index-1,0] = np.subtract(*np.quantile(dev[mask], [0.84, 0.16]))
        zstats[bin_index-1,2] = np.median(abs(dev[mask]))
        zstats[bin_index-1,1] = np.sum(abs(dev[mask])>0.025) / mask.sum()
        

    bx = axarr[1]
    
    labels = [
        r'$\Delta z/(1+z)$, central 68$^{\rm th}$ interval',
        'outlier fraction',
        r'$\langle \Delta z/(1+z) \rangle_{50}$'
    ]
    bx.bar ( 
        sampling.midpts(xbins), 
        zstats[:,0],
        width=np.diff(xbins),
        color=[merian_facecolor(idx, xbins) for idx in range(1, len(xbins))] 
    )
    
    ek.text ( 
        0.95,
        0.05, 
        r'$18 < m_i < 24$', #r'$18 < m_i < 24$',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'),
        ax=ax,
        zorder=100
    )
    if show_index == 1:
        ek.text ( 0.95, 0.95, r'$f_{\rm out} = \frac{N(|\Delta z/(1+z)|>0.025)}{N_{\rm total}}$', ax=bx )
    bx.set_xlabel(r'$z_{\rm spec}$')
    bx.set_ylabel(labels[show_index])

    plt.tight_layout ()


def merian_to_legacycsv ( cat ):
    tab = cat['objectId_Merian	coord_ra_Merian	coord_dec_Merian'.split()]
    tab.rename_columns(tab.colnames, ['name','RA','DEC'])
    tab.write('/Users/kadofong/Desktop/merian_legacysurveyinput.csv',format='ascii.csv',overwrite=True)    
    
    
# \\\ PROPOSAL

def compute_puritycompleteness ( cat, bins, discriminator='gminusr' ):
    is_inband = (cat['z_spec']>0.0)&(cat['z_spec']<0.15)
    #assns = np.digitize(cat[discriminator], bins)
    bmidpts = sampling.midpts(bins)
    arr = np.zeros([len(bmidpts),4])
    for bindex in range(1, bins.size):
        #section = assns == bindex
        targets = cat[discriminator] < bmidpts[bindex-1]
        completeness = targets[is_inband].sum()/is_inband.sum()
        purity = targets[is_inband].sum() / targets.sum()
        tp_rate = (targets&is_inband).sum () / is_inband.sum()
        fp_rate = (targets&~is_inband).sum() / targets.sum()
        arr[bindex-1,0] = completeness
        arr[bindex-1,1] = purity
        arr[bindex-1,2] = tp_rate
        arr[bindex-1,3] = fp_rate
    return arr


def mk_gminusrcut (catalog, cutcolumn='gminusr',show_roc = False):
    fig, axarr=plt.subplots(1,2,figsize=(10,5))
    bins = np.linspace(*np.nanquantile(catalog[cutcolumn], [0.01,.99]),20)
    bmidpts = sampling.midpts(bins)

    labels = ['in band', 'out of band']
    color_list = ['tab:blue','tab:red']
    for mindex,mask in enumerate([catalog['z_is_good'],~catalog['z_is_good']]):
        ek.hist(
            catalog[mask][cutcolumn], 
            bins=bins, 
            histtype='bar', 
            lw=2, 
            label=labels[mindex], 
            alpha=.5, 
            color=color_list[mindex],
            ax=axarr[0]
        )

    axarr[0].legend (loc='upper left', frameon=False)
    axarr[0].set_xlabel(r'$g-r$')
    axarr[0].set_ylabel('N')

    bmidpts = sampling.midpts(bins)
    pc_stats = compute_puritycompleteness ( catalog, bins )
    
    if not show_roc:
        axarr[1].plot (
            bmidpts, 
            pc_stats[:,0],
            lw=3,
            color='k',
            label='completeness'
        )
        axarr[1].plot (
            bmidpts, 
            pc_stats[:,1],
            ls='--',
            lw=3,
            color='k',
            label='purity'
        )
        #axarr[1].axvline(0., color='C1', lw=2)
    else:
        axarr[1].scatter(
            pc_stats[:,3],
            pc_stats[:,2],
            lw=3,
            c=bmidpts,
            cmap='coolwarm'
        )

    axarr[1].set_xlabel(r'maximum allowed $g-r$')
    axarr[1].set_ylabel('metric')
    axarr[1].legend()
    plt.tight_layout()    
    return fig, axarr
    
def flux2mag ( flux ):
    # \\\ Merian catalogs are in nJy, such that
    # \\ mAB = -2.5*np.log10(F_merian) + 31.4
    # \\ since -2.5*np.log10(gAB * 1e9 / Jy) = 31.4
    return -2.5*np.log10(flux) + 31.4

def qa_models (ctim, ctmap, haim, hamap, ltab, radii, segmentator, pad, x_0=None, y_0=None, ellip=0., pa=0.):
    colors = ['r','lime','cyan']
    fig, axarr = plt.subplots(2,3,figsize=(12,8))
    cmap='Greys'
    ek.imshow(ctim, ax=axarr[0,1], cmap=cmap, q=0.005)
    ek.imshow(segmentator(ctmap), ax=axarr[0,0], cmap=cmap, q=0.005)
    ek.imshow(segmentator(ctmap)-ctim,ax=axarr[0,2], cmap='coolwarm')
    
    ek.imshow(haim, ax=axarr[1,1], cmap=cmap, q=0.005)
    ek.imshow(segmentator(hamap), ax=axarr[1,0], cmap=cmap, q=0.005)
    ek.imshow(segmentator(hamap)-haim,ax=axarr[1,2], cmap='coolwarm')    

    if len(ltab) > 0:
        for ax in axarr[1]:
            ax.scatter (ltab['xc']-pad,ltab['yc']-pad, marker='x', color='cyan')    
    for ax in axarr.flatten():
        if x_0 is None:
            center = imstats.get_center(ctim)
        else:
            center = (x_0,y_0)
        for idx, radius in enumerate(radii):        
            ellipse = patches.Ellipse(
                center,
                radius*2.,
                (1.-ellip)*radius*2.,
                np.rad2deg(pa),
                facecolor="None",
                edgecolor=colors[idx]
            )
            ax.add_patch(ellipse)
            
        for idx, radius in enumerate(ltab['a']):
            ellipse = patches.Ellipse(
                center,
                2.*radius,#/(1.-ellip),
                2.*radius*(1.-ellip),
                np.rad2deg(pa),
                facecolor="None",
                edgecolor=colors[-1],
                ls=':'
            )
            #ellipse = patches.Circle(
            #    center,
            #    radius,
            #    edgecolor=colors[-1],
            #    facecolor="None",
            #    ls=':'
            #)
            ax.add_patch(ellipse)     


    plt.tight_layout ()


def compute_halpha_maps (catalog, imdir, pixscale=0.168, savedir=None, verbose=False):
    ovlarr = np.zeros([len(catalog),8])
    mpix = (0.5/pixscale)**2 * np.pi
    skytophys = cosmo.kpc_proper_per_arcmin ( catalog['z_hi'] ).to(u.kpc/u.arcsec).value

    rstats = []
    for index, name in enumerate(catalog[merian_id]):
        if verbose:
            print(f'Running {name}...')
        hamap = np.load (f'{imdir}/{name}_hamap.npy')
        ctmap = np.load (f'{imdir}/{name}_contmap.npy')

        # \\ SOURCE DETECTION
        c_rms = np.std(ctmap[325:,325:])
        sepcat, segmap = sep.extract(ctmap,thresh=3., err=c_rms, segmentation_map=True, deblend_cont=0.01)
        cindex = imstats.get_centerval(segmap)
        
        # \\ DENDROGRAM SF REGION DETECTION AND CATALOG CREATION
        rms = np.std(hamap[325:,325:])
        dendro = Dendrogram.compute ( np.where((segmap==cindex)|(segmap==0),hamap,0.), min_value=3.*rms, min_delta=rms, min_npix=mpix )
        
        arr = np.zeros([len(dendro.leaves),4])
        for idx,leaf in enumerate(dendro.leaves):
            coord, val = leaf.get_peak()
            arr[idx,0] = coord[1]
            arr[idx,1] = coord[0]
            arr[idx,2] = val
            arr[idx,3] = leaf.get_npix ()
            
        ltab = table.Table( arr, names=['xc','yc','peakval', 'npix'], dtype=[int,int,float,int])
        xcenter,ycenter = imstats.get_center(hamap)
        ltab['r'] = np.sqrt((ltab['xc']-xcenter)**2+(ltab['yc']-ycenter)**2)
        rphys = ltab['r'] * pixscale * skytophys[index]
        ltab = ltab[(rphys<20.)]
        ltab.write(f"{imdir}/{name}_dendro.csv", overwrite=True)

    
        # \\ MORPHOLOGY STATISTICS
        ovlarr[index,0] = len(ltab)        
        if ovlarr[index,0]>1:
            X = ltab[['xc','yc']].to_pandas().values
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)
            ovlarr[index,1] = np.nanmedian(distances[:,1])* pixscale * skytophys[index] # pix * arcsec/pix * kpc/arcsec
        else:
            ovlarr[index,1] = np.NaN

        ltab['r'] = np.sqrt((ltab['xc']-xcenter)**2+(ltab['yc']-ycenter)**2)
        rhii = np.sum ( ltab['peakval'] * ltab['r'] ) / np.sum(ltab['peakval'] )    
        ovlarr[index,5] = rhii*pixscale * skytophys[index]

        if np.isnan(rhii):
            rmax = 75
        else:
            rmax = max(50,rhii*3)


        Y,X = np.mgrid[:hamap.shape[0],:hamap.shape[1]]
        R = np.sqrt( (X-xcenter)**2 + (Y-ycenter)**2 )
        ovlarr[index,4] = np.nansum(np.where(R<rmax, hamap,0.))

        if cindex>0:            
            row = sepcat[cindex-1]
            
            # \\ SERSIC FITTING            
            pad = abs(int(xcenter - rmax))
            
            fn = lambda x: np.where((segmap==cindex)|(segmap==0), x, 0.)[pad:-pad,pad:-pad]

            ellip = 1.-row['b'] / row['a']
            reff = np.sqrt(row['a']*row['b'])
            ctmodel, ctim = fit.fit_sersic_2d ( fn(ctmap), 0.5, init_r_eff=reff, init_ellip=ellip, init_theta=row['theta'])

            cx = ctmodel.x_0.value
            cy = ctmodel.y_0.value
            hamodel, haim = fit.fit_sersic_2d (
                fn(hamap), 
                0.5, 
                init_r_eff=reff, 
                init_ellip=ellip, 
                init_theta=row['theta'], 
                init_x_0=cx,
                init_y_0=cy,
                fixed_parameters=['x_0','y_0']
            ) 
            ovlarr[index, 2] = ctmodel.r_eff * pixscale * skytophys[index]
            ovlarr[index, 3] = hamodel.r_eff * pixscale * skytophys[index]
            
            ovlarr[index, 6] = ctmodel.n.value
            ovlarr[index, 7] = hamodel.n.value
            
            # \\ remeasure SF region radial positions based on distance from continuum Sersic center
            ltab['r_c'] = np.sqrt((ltab['xc']-cx-pad)**2+(ltab['yc']-cy-pad)**2)
            ltab['a'] = fit.ellipse_from_point(ltab['xc']-cx-pad, ltab['yc']-cy-pad, ellip=ctmodel.ellip, theta=ctmodel.theta)
            ltab = ltab[(ltab['a']/ctmodel.r_eff) < 4] # MAX a/Reff
            rhii = np.sum ( ltab['peakval'] * ltab['a'] ) / np.sum(ltab['peakval'] )    
            ovlarr[index,5] = rhii * pixscale * skytophys[index]


            qa_models (
                ctim, 
                ctmap, 
                haim, 
                hamap, 
                ltab, 
                radii=[ctmodel.r_eff, hamodel.r_eff, rhii], 
                x_0=cx, 
                y_0=cy,
                ellip=ctmodel.ellip, 
                pa=ctmodel.theta ,
                segmentator = fn, 
                pad=pad
            )
            
            if savedir is not None:
                plt.savefig(f'{savedir}/{name}_mightee.png')
                plt.close()

            if len(ltab)>0:
                # save region stats
                ltab[merian_id] = name
                ltab['a'] = ltab['a'] * pixscale * skytophys[index]
                ltab['logmhi'] = catalog.loc[name]['logmhi']
                ltab['logmstar'] = catalog.loc[name]['logmstar']
                ltab['reff'] = ovlarr[index,2]
                ltab['rhii'] = ovlarr[index,5]
                ltab['reff_ha'] = ovlarr[index,3]
                rstats.append(ltab)


    gtable = table.Table( ovlarr, names=['nregions','nndist','reff','reff_ha','totflux_ha', 'rhii', 'nsersic_cont','nsersic_ha'])
    gtable[merian_id] = catalog[merian_id]
    gtable.add_index ( merian_id )

    rstats = table.vstack(rstats)    
    return gtable, rstats


def load_merianxmightee ():
    from SAGAbg import calibrations
    matches = table.Table.read('./merian_cross_mightee.csv')

    matches['use'] = True
    matches[merian_id] = [ f'M{x}' for x in matches[merian_id] ]
    matches.add_index(merian_id)
    bad_det = [ f'M{x}' for x in [3036503670242974680,3036547650708063020,3036785145219690318,3036890698335947376,3037233745963806791,3496059948234726771,3036899494428974599] ]

    for bd in bad_det:
        matches.loc[bd]['use'] = False
    #haew = table.Table.read('/Users/kadofong/work/projects/merian/local_data/cutouts/mightee/halpha_maps/haew.csv')

    #halum = calibrations.LHa_from_EW(haew['haew'], matches['Mr'])
    #hasfr = calibrations.LHa2SFR(halum)
    #ssfr = (hasfr / (10.**matches['logmstar']*u.Msun)).to(u.yr**-1)
    #tdep = (10.**matches['logmhi']*u.Msun/ hasfr ).to(u.Gyr)
    #sfe = hasfr / (10.**matches['logmhi']*u.Msun)
    #fhi = 10.**matches['logmhi'] / (10.**matches['logmstar']+10.**matches['logmhi']) 
    #matches['fhi'] = fhi
    #matches['ssfr'] = ssfr
    #matches['tdep'] = tdep
    #matches['hasfr'] = hasfr
    #matches.add_index(merian_id)
    
    return matches 


def mk_stackedradialprofiles (
        matches, 
        gtable, 
        rstats, 
        discriminator='logmhi', 
        quantiles=None,
        bins=None, 
        sf_relation=None, 
        density=False, 
        fit_plaw=True, 
        return_normalization=True,
        axarr=None,
    ):
    if quantiles is None:
        quantiles = np.linspace(0,1,4)
    
    if axarr is None:
        fig, axarr = plt.subplots(1,2,figsize=(10,4))
    else:
        fig = None
    
    disctitle = {
        'reff_ha':r'$R_{\rm eff, H\alpha}$',
        'logmstar':ek.common_labels['logmstar'],
        'logmhi':ek.common_labels['logmhi'],
        'nregions':r'$N_{\rm reg}$',
        'z_hi':r'$z_{\rm HI}$',
        'coord_ra_Merian':'RA',
        'rhii':r'$R_{\rm HII}$',
    }[discriminator]
    dunit = {
        'reff_ha':'kpc',
        'logmhi':r'$M_\odot$',
        'logmstar':r'$M_\odot$',
        'nregions':None,
        'z_hi':None,
        'coord_ra_Merian':'deg',
        'rhii':'kpc',
    }[discriminator]
    dnames = [', N708c',r', H\alpha']
    linestyles = ['-','--',':']

    #radii = ['reff','reff_ha']
    if bins is None:
        bins = np.linspace(0.,np.nanquantile(np.concatenate([rstats['a']/rstats[r_base] for r_base in radii]),0.99),20)#/rstats[r_base]
    dr = np.diff(bins)[0]
    mask = matches['use'] 

    r_base = 'reff'
    dindex = 0
    #for dindex, r_base in enumerate(radii[:1]):    
    rnrml = rstats['a']/rstats[r_base]
    
    if discriminator in matches.colnames:
        df = matches
    elif discriminator in gtable.colnames:
        df = gtable
    else:
        raise ValueError (f"{discriminator} not in matches or gtable!")

    pdf = df.to_pandas()
    if pdf.index.name != merian_id:
        pdf = pdf.set_index(merian_id)
    qts = np.nanquantile(df[mask][discriminator], quantiles)
    
    lf_estimates = []
    normalization = np.zeros(len(qts)-1)
    ring_areas = lambda r: 2.*np.pi*r                
    for idx in range(len(qts)-1):
        ax = axarr[0]
        rqs = pdf.reindex(rstats[merian_id])
        massbin = (rqs[discriminator]>qts[idx])&(rqs[discriminator]<=qts[idx+1])
        if dunit is None:
            dunit_formatted = ''
        else:
            dunit_formatted = ' [' + dunit + ']'
        title = r'$%.2f<$%s$\leq%.2f$%s' % (qts[idx], disctitle, qts[idx+1], dunit_formatted)       
    
        _r, _w = functions.fmasker(rnrml[massbin], 1./ring_areas(rnrml[massbin]))

        imhist = np.histogram(
            _r,
            bins=bins,                
            weights=_w,
            density=density
        )
        normalization[idx] = np.trapz(imhist[0], sampling.midpts(imhist[1]) )
        
        # \\\ FIT PLAW
        if fit_plaw:
            lf = fit.LFitter (mmin=-10.,mmax=10.,bmin=-10.,bmax=10.,smin=0.05, smax=1.)
            
        strapped = sampling.bootstrap_histcounts(_r,bins,5000,w=_w, density=density)
        ylow = np.nanquantile(strapped,0.16,axis=0)
        yhigh = np.nanquantile(strapped,0.84,axis=0)
        
        is_ul = ylow <= 0 
        ek.errorbar(
            sampling.midpts(imhist[1])[~is_ul],
            imhist[0][~is_ul],
            ylow=ylow[~is_ul],
            yhigh=yhigh[~is_ul],
            ax=ax,
            color=f'C{idx}'
        )
        ax.scatter(
            sampling.midpts(imhist[1])[is_ul],
            yhigh[is_ul],
            color=f'C{idx}',
            marker='v'
        )
        yerr = (np.log(10.)*imhist[0])**-1*np.nanstd(strapped, axis=0)
        x,y,yerr = functions.fmasker(sampling.midpts(imhist[1]),np.log10(imhist[0]),yerr)
        
        if fit_plaw:
            lf.run(
                x,
                y,
                yerr,
                initial=[-1.,1.5,0.1]
            )
            
            pest = lf.get_param_estimates ()
            lf_estimates.append(pest)
            ek.errorbar(
                0.5*(qts[idx]+qts[idx+1]),
                pest[1,0],
                xlow=qts[idx],
                xhigh=qts[idx+1],
                ylow=pest[0,0],
                yhigh=pest[2,0],
                ax=axarr[-1],
                color=f'C{idx}', 
                capsize=3,
            )

        ax.set_yscale('log')
        ax.set_xlabel(r'$r_{\rm SFreg}/R$')
        ax.set_ylabel(r'$dN/d(r/R)$')            
        ax.set_ylim(0.01,ax.get_ylim()[1])

    lf_estimates = np.asarray(lf_estimates)
    
    if sf_relation is not None:
        ms = np.linspace(qts[0], qts[-1], 10)
        axarr[-1].plot ( ms, sf_relation(ms) )
    axarr[-1].set_xlabel(r'%s %s' % (disctitle, dunit))
    axarr[-1].set_ylabel('power-law index')
    plt.tight_layout ()    
    
    if return_normalization:
        return fig, axarr, normalization, lf_estimates
    else:
        return fig, axarr, lf_estimates
    
def load_merianxgalex(
        merian_filename=None,
        galex_filename=None,
        save_merian_for_crossmatch=False,
        save_crossmatch=False,
    ):
    if merian_filename is None:
        merian_filename = '/Users/kadofong/Downloads/Merian_DR1_photoz_EAZY_v1.2.fits'
    if galex_filename is None:
        galex_filename = '/Users/kadofong/Downloads/MAST_Crossmatch_GALEX.csv'
        
    # \\ load MERIAN
    merian = table.Table(fits.getdata(merian_filename,1))    
    merian.rename_column(merian_ra,'RA')
    merian.rename_column(merian_dec,'DEC')
    inband = merian['z_phot']>0.06
    inband &= merian['z_phot']<0.1
    inband &= merian['i_cModelmag_Merian']<22.
    merian = merian[inband].to_pandas ()
    merian = merian.set_index(merian_id)
    merian.index = [ 'M%i' % idx for idx in merian.index ]     
    if save_merian_for_crossmatch:
        merian.to_csv("~/Desktop/merian_inband.csv",)
        return 0

    # \\ load GALEX
    crossmatch = table.Table.read(
        galex_filename,
        format='csv', 
        comment='#', 
        #dtype=[(analysis.merian_id,int)]
    )
    crossmatch.rename_column('Column0', merian_id)
    crossmatch.add_index(merian_id)
    crossmatch = crossmatch.to_pandas ()

    galex = crossmatch.loc[crossmatch.sort_values('nuv_exptime', ascending=False).index.duplicated(keep='first')]    
    if save_crossmatch:
        merian.reindex(galex.index).to_csv('/Users/kadofong/Desktop/merian_cross_galex.csv')
        return 0
    
    return merian.reindex(galex.index), galex

def uvha_galacticextinction ( merian, rv=4.05, z_col='z_phot' ):    
    ge_arr = np.zeros([len(merian),wv_eff.size])
    for idx,(z,av) in enumerate(zip(merian[z_col], merian['ebv_Merian'] * rv)):
        wv_eff = np.array([1548.85, 2303.37, 0.])/(1.+z)
        wv_eff[2] = 6563.
        ge_arr[idx] = observer.gecorrection ( wv_eff, av, rv, return_magcorr=False)
    return ge_arr  



def naive_luminosities ( merian, galex ):
    merian_sources, haflux = naive_halpha(merian, galex)    
    # \\ 1'' aperture -> total flux correction,
    # \\ right now just approximated from i_cmodel / i_gaap1p0
    totcorr = merian_sources['i_cModelFlux_Merian'] / merian_sources['i_gaap1p0Flux_Merian']
    # \\ rough internal extinction correction assuming AV=0.5
    z = 0.078
    wv_eff =  np.array([1548.85/(1.+z), 2303.37/(1.+z), 6563.])
    dust_corr = observer.extinction_correction ( wv_eff, 0.5 )[0]
    # \\ galactic extinction correction
    ge_arr = uvha_galacticextinction ( merian_sources, z_col=z_col )

    
    haflux *= ge_arr[:,2]
    emission_correction = fitting_utils.correct_N2_S3(
        merian_sources[z_col],
        merian_sources['logmass_gaap1p0']
    )
    haflux /= emission_correction
    haflux *= totcorr
    haflux *= dust_corr[2]
    halum = haflux * 4.*np.pi*cosmo.luminosity_distance(merian_sources[z_col]).to(u.cm)**2

    nuvflux = 10.**(galex['nuv_mag'].values/-2.5) * 3631. * u.Jy
    nuvflux = (nuvflux * ge_arr[:,1]).to(u.erg/u.s/u.cm**2/u.Hz)
    nuvflux *= dust_corr[1]
    nuvlum = (nuvflux * 4.*np.pi * cosmo.luminosity_distance(merian_sources[z_col]).to(u.cm)**2).to(u.erg/u.s/u.Hz)    
    
    return halum, nuvlum
        
def load_abbylines (merian_sources, galex):    
    linetable = table.Table.read(
        '/Users/kadofong/work/projects/merian/local_data/cutouts/galex/haew.csv', 
        units=[u.AA, u.Jy, None]
    )

    rv = 4.05
    wv_eff = np.array([1548.85, 2303.37, 7080.])
    ge_arr = np.zeros([galex.index.size,wv_eff.size])
    for idx,(z,av) in enumerate(zip(merian_sources['z_phot'].value, merian_sources['ebv_Merian'].value * rv)):
        ge_arr[idx] = observer.gecorrection ( wv_eff*(z+1.), av, rv, return_magcorr=False)


    z_phot = merian_sources['z_phot'].values
    wl_obs = 6563. * u.AA * ( 1. + z_phot )
    linetable['haflux'] = (linetable['haew'].quantity * linetable['continuum_specflux'].quantity * co.c / wl_obs**2).to(u.erg/u.s/u.cm**2)
    linetable['haflux'] = linetable['haflux'] * ge_arr[:,2]
    linetable['halum'] = linetable['haflux'] * 4.*np.pi * cosmo.luminosity_distance(z_phot).to(u.cm)**2

    uvmeas = galex
    linetable['nuvflux'] = 10.**(uvmeas['nuv_mag'].values/-2.5) * 3631. * u.Jy
    linetable['nuvflux'] = (linetable['nuvflux'] * ge_arr[:,1]).to(u.erg/u.s/u.cm**2/u.Hz)
    linetable['nuvlum'] = (linetable['nuvflux'] * 4.*np.pi * cosmo.luminosity_distance(z_phot).to(u.cm)**2).to(u.erg/u.s/u.Hz)    
    return linetable