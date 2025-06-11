from astropy.io import fits
from carpenter import pixels

def load_merian ( galaxy_id, filter_name ):
    filename = f'magellan_spec_{galaxy_id}_{filter_name}_deepCoadd_calexp.fits'
    psfname = f'magellan_spec_{galaxy_id}_{filter_name}_psf.fits'
    dirname = path_d[galaxy_id]
    path = dirname + filename
    img = fits.getdata(path, 1)
    var = fits.getdata(path, 3)
    psf = fits.getdata(dirname + psfname, 0)
    return img, var, psf

def load_hsc ( galaxy_id, filter_name ):
    filename = f's18a_wide_{galaxy_id}_{filter_name}.fits'
    psfname = f's18a_wide_{galaxy_id}_{filter_name}_psf.fits'
    dirname = path_d[galaxy_id]
    path = dirname + filename
    img = fits.getdata(path, 1)
    var = fits.getdata(path, 3)
    psf = fits.getdata(dirname + psfname, 0)
    return img, var, psf

def load_image ( galaxy_id, filter_name ):
    if filter_name in ['N708','N540']:
        return load_merian ( galaxy_id, filter_name )
    else:
        return load_hsc ( galaxy_id, filter_name )
    
def load_bbmb ( gid, **kwargs ):
    bbmb = pixels.BBMBImage ( )
    for band in ['g','N540','r','N708','i','z','y']:
        bbmb.add_band ( band, *load_image(gid, band) )

    fwhm_a, _ = bbmb.measure_psfsizes()
    mim, mpsf = bbmb.match_psfs ( np.argmax(fwhm_a), cbell_alpha=1., **kwargs )
    return bbmb