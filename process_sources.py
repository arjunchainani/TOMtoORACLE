import tqdm
from astropy.table import Table
from astroOracle.pretrained_models import ORACLE, ORACLE_lite
from astroOracle.LSST_Source import LSST_Source
from load_tom import load_oracle_features_from_TOM

# maps from the TOM feature names to the ORACLE feature names
ts_key_map = { # need to figure out solution for MJD and PHOTFLAG features
    'filtername': 'BAND',
    'psflux': 'FLUXCAL',
    'psfluxerr': 'FLUXCALERR',
}

static_key_map = {
    'ra': 'RA',
    'decl': 'DEC',
    'mwebv': 'MWEBV',
    'mwebv_err': 'MWEBV_ERR',
    'z_final': 'REDSHIFT_HELIO',
    'z_final_err': 'REDSHIFT_HELIO_ERR',
    'hostgal_zphot': 'HOSTGAL_PHOTOZ',
    'hostgal_zphot_err': 'HOSTGAL_PHOTOZ_ERR',
    'hostgal_zspec': 'HOSTGAL_SPECZ',
    'hostgal_zspec_err': 'HOSTGAL_SPECZ_ERR',
    'hostgal_ra': 'HOSTGAL_RA',
    'hostgal_dec': 'HOSTGAL_DEC',
    'hostgal_snsep': 'HOSTGAL_SNSEP',
    'hostgal_ellipticity': 'HOSTGAL_ELLIPTICITY',
    'hostgal_mag_u': 'HOSTGAL_MAG_u',
    'hostgal_mag_g': 'HOSTGAL_MAG_g',
    'hostgal_mag_r': 'HOSTGAL_MAG_r',
    'hostgal_mag_i': 'HOSTGAL_MAG_i',
    'hostgal_mag_z': 'HOSTGAL_MAG_z',
    'hostgal_mag_y': 'HOSTGAL_MAG_Y',
}

# wrapper for the LSST_Source class that can take in manual input rather than just from parquet
class TOM_Source(LSST_Source):
    def __init__(self, snid):
        setattr(self, 'SNID', snid)
        # add setattr for elasticc_class after converting it from gentype