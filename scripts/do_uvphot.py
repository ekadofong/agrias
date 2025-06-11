import os
import glob
import numpy as np

# \\ parallelization
import concurrent.futures
from tqdm import tqdm

# \\ plotting
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
from astropy import cosmology
from astropy import coordinates

import sep

#from ekfphys import observer, calibrations
from ekfplot import plot as ek
#from ekfplot import colors as ec
#from ekfplot import colorlists
from ekfphot import photometry
from ekfparse import query
from ekfstats import sampling

import reader

cosmo = cosmology.FlatLambdaCDM(70.,0.3)

savedir = './galex_images/'
savedir = os.path.abspath(savedir)
master_filename = f'{savedir}/master_keys.txt'

def load_catalog ():
    ms3 = reader.merianselect(pmin=0., version=3)

    pzmin=0.26
    ms = reader.merianselect(pmin=pzmin, version=2)
    #lm = (ms['z500']>0.06)&(ms['z500']<0.1)

    znew = ms3.reindex(ms.index)['z_spec']
    zold = ms['z_spec']
    zcombined = np.where(np.isfinite(znew), znew, zold)
    ms['z_spec'] = zcombined    
    return ms

def pull_cutouts (row):
    topull, names = query.get_galexobs ( row.RA, row.DEC, )
    
    mfile = pd.read_csv(master_filename, delim_whitespace=True, index_col=0)
    cutout_dir = f'{savedir}/{row.name}'

    for idx,band in enumerate(['fuv','nuv']):
        match = mfile.query(f'{band}_obsid=="{names[idx]}"')
        
        if len(match)==0:            
            continue

        parent_row = match.iloc[0]
        cfile = query.hotfix_galex_naming(parent_row[f'{band}_obsid'])

        if not os.path.exists(cutout_dir):
            os.makedirs(cutout_dir)

        for file in glob.glob(f'{savedir}/{parent_row.name}/{cfile}*'):   
            target = f'{savedir}/{row.name}/{os.path.basename(file)}'
            if not os.path.exists(target):   
                print(f'Symlinking {os.path.basename(file)} from {parent_row.name}.')                 
                os.symlink(
                    file,
                    target,
                )
    exitcode, manifest, _ = query.download_galeximages(row['RA'], row['DEC'], row.name, savedir=savedir, obsout=(topull,names))
    
    if not os.path.exists(master_filename):
        with open(master_filename,'w') as f:
            print('merian_id fuv_obsid nuv_obsid', file=f)
    with open(master_filename, 'a') as f:
        print(f'{row.name} {names[0]} {names[1]}', file=f)

def do_photometry (row, make_figure=True):    
    gcutouts = query.load_galexcutouts(
        row.name,
        savedir,
        coordinates.SkyCoord(row.RA, row.DEC, unit='deg'),
        1.*u.arcmin,
        1.*u.arcmin,
    )
    
    data = gcutouts['nd']['PRIMARY'].data.byteswap().newbyteorder().astype(np.float64)
    var = gcutouts['nd']['VARIANCE'].data.byteswap().newbyteorder().astype(np.float64)
    catalog, segmap = sep.extract(
        data - np.median(data),
        #3.,
        3.*sampling.sigmaclipped_std(data, low=5., high=3.),
        #var=var,
        segmentation_map=True,
    )  
    
    gi = photometry.GalexImaging(gcutouts, filter_directory='/Users/kadofong/work/theory/sfr_calibrators/local_data/filters/')
    
    ul_phot = gi.do_upperlimitphotometry((row.RA, row.DEC))
    ul_emask = gi.emask.copy()
    ell_phot = gi.do_ephotometry((row.RA,row.DEC), catalog[segmap[data.shape[0]//2, data.shape[1]//2]-1], cat_pixscale=gi.pixscale)
    ell_emask = gi.emask.copy()
    
    np.save(f'{savedir}/{row.name}/photometry.npy', np.array([ul_phot, ell_phot]))
    
    if make_figure:
        fig, axarr = plt.subplots(1,2,figsize=(10,4))
        labels = ['FUV','NUV']
        for idx,band in enumerate(['fd','nd']):
            data = gcutouts[band]['PRIMARY'].data.byteswap().newbyteorder().astype(np.float64)
            ek.imshow(data,cmap='Greys', ax=axarr[idx])
        
            axarr[idx].contour(ul_emask, colors='C1')
            axarr[idx].contour(ell_emask, colors='C0')
            ek.text(
                0.975,
                0.025,
                labels[idx],
                fontsize=20,
                color='k',
                bordercolor='w',
                borderwidth=5,
                ax=axarr[idx]
            )
            ek.text(
                0.025,
                0.975,
                rf'''$m_{{{labels[idx]}, ul}} = {-2.5*np.log10(ul_phot[0,idx]/3631.):.1f}$
'$m_{{{labels[idx]}, \epsilon}} = {-2.5*np.log10(ell_phot[0,idx]/3631.):.1f}$''',
                ax=axarr[idx],
                bordercolor='w',
                borderwidth=5
            )
        plt.tight_layout()
        plt.savefig(f'{savedir}/{row.name}/{row.name}_nuvdet.png')
        
def process_row(row):
    if os.path.exists(f'{savedir}/{row.name}/photometry.npy'):
        return row.name
    else:
        raise ValueError
    
    pull_cutouts(row)
    do_photometry(row)
    plt.close()
    return row.name
        
if __name__ == '__main__':
    ms = load_catalog()
    rows = [row for _, row in ms.iterrows()]
    
    debug = False
    if debug:
        process_row(rows[-1])
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_row, row) for row in rows]

            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
                pass
