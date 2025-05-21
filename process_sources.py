import numpy as np
import polars as pl
from astroOracle.LSST_Source import LSST_Source

# maps from the TOM feature names to the ORACLE feature names
ts_key_map = { # need to figure out solution for MJD and PHOTFLAG features
    'filtername': 'BAND',
    'psflux': 'FLUXCAL',
    'psfluxerr': 'FLUXCALERR',
}

static_key_map = {
    'diaobject_id': 'SNID',
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

gentype_to_type = {
    10: 'SNIa', 25: 'SNIb/c', 37: 'SNII', 12: 'SNIax', 11: 'SN91bg', 50: 'KN', 82: 'M-dwarf Flare', 84: 'Dwarf Novae', 88: 'uLens', 
    40: 'SLSN', 42: 'TDE', 45: 'ILOT', 46: 'CART', 59: 'PISN', 90: 'Cepheid', 80: 'RR Lyrae', 91: 'Delta Scuti', 83: 'EB', 60: 'AGN',
    32: 'SNII', 31: 'SNII', 35: 'SNII', 36: 'SNII', 21: 'SNIb/c', 20: 'SNIb/c', 72: 'SLSN', 27: 'SNIb/c', 26: 'SNIb/c'
}

# # wrapper for the LSST_Source class that can take in manual input rather than just from parquet
class TOM_Source(LSST_Source):
    other_features = ['RA', 'DEC', 'MWEBV', 'MWEBV_ERR', 'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR', 'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR', 'HOSTGAL_SPECZ', 'HOSTGAL_SPECZ_ERR', 'HOSTGAL_RA', 'HOSTGAL_DEC', 'HOSTGAL_SNSEP', 'HOSTGAL_DDLR', 'HOSTGAL_LOGMASS', 'HOSTGAL_LOGMASS_ERR', 'HOSTGAL_LOGSFR', 'HOSTGAL_LOGSFR_ERR', 'HOSTGAL_LOGsSFR', 'HOSTGAL_LOGsSFR_ERR', 'HOSTGAL_COLOR', 'HOSTGAL_COLOR_ERR', 'HOSTGAL_ELLIPTICITY', 'HOSTGAL_MAG_u', 'HOSTGAL_MAG_g', 'HOSTGAL_MAG_r', 'HOSTGAL_MAG_i', 'HOSTGAL_MAG_z', 'HOSTGAL_MAG_Y', 'HOSTGAL_MAGERR_u', 'HOSTGAL_MAGERR_g', 'HOSTGAL_MAGERR_r', 'HOSTGAL_MAGERR_i', 'HOSTGAL_MAGERR_z', 'HOSTGAL_MAGERR_Y']
    
    def __init__(self, data):
        setattr(self, 'astrophysical_class', gentype_to_type[data[3][0]['gentype']])
        
        # loads in and normalizes the MJD timestamps for each observation
        ts_data = data[2]
        
        mjd = np.array([])
        flux = np.array([])
        flux_err = np.array([])
        filters = np.array([])
        
        for obs in ts_data:
            mjd = np.append(mjd, obs['midpointtai'])
            flux = np.append(flux, obs['psflux'])
            flux_err = np.append(flux_err, obs['psfluxerr'])
            filters = np.append(filters, obs['filtername'])
        
        self.MJD = mjd
        self.FLUXCAL = flux
        self.FLUXCALERR = flux_err
        self.BAND = filters
        
        self.PHOTFLAG = pl.Series([0] * len(self.MJD)) # temporary until a better solution is found
        
        # loading in static data
        static_data = data[1]
        
        for feature in static_data.keys():
            setattr(self, static_key_map[feature], static_data[feature])
        
        # extra LC processing
        self.process_lightcurve()
        self.compute_custom_features()
    
    def process_lightcurve(self):
        saturation_mask =  (self.PHOTFLAG & 1024) == 0 

        # Alter time series data to remove saturations
        for time_series_feature in self.time_series_features:
            if time_series_feature != 'PHOTFLAG':
                setattr(self, time_series_feature, getattr(self, time_series_feature)[saturation_mask])
            else:
                setattr(self, time_series_feature, getattr(self, time_series_feature).filter(saturation_mask))