import numpy as np
import polars as pl
import pandas as pd
import networkx as nx
from tqdm import tqdm
from typing import List
from astroOracle.LSST_Source import LSST_Source
from astroOracle.dataloader import static_feature_list, ts_length, ts_flag_value, get_static_features
from astroOracle.pretrained_models import ORACLE
from astroOracle.interpret_results import get_conditional_probabilites
from astroOracle.taxonomy import source_node_label
from keras.utils import pad_sequences
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
pd.set_option('display.max_columns', None)

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

# wrapper for the LSST_Source class that can take in manual input rather than just from parquet
class TOM_Source(LSST_Source):
    # other_features = ['RA', 'DEC', 'MWEBV', 'MWEBV_ERR', 'REDSHIFT_HELIO', 'REDSHIFT_HELIO_ERR', 'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR', 'HOSTGAL_SPECZ', 'HOSTGAL_SPECZ_ERR', 'HOSTGAL_RA', 'HOSTGAL_DEC', 'HOSTGAL_SNSEP', 'HOSTGAL_DDLR', 'HOSTGAL_LOGMASS', 'HOSTGAL_LOGMASS_ERR', 'HOSTGAL_LOGSFR', 'HOSTGAL_LOGSFR_ERR', 'HOSTGAL_LOGsSFR', 'HOSTGAL_LOGsSFR_ERR', 'HOSTGAL_COLOR', 'HOSTGAL_COLOR_ERR', 'HOSTGAL_ELLIPTICITY', 'HOSTGAL_MAG_u', 'HOSTGAL_MAG_g', 'HOSTGAL_MAG_r', 'HOSTGAL_MAG_i', 'HOSTGAL_MAG_z', 'HOSTGAL_MAG_Y', 'HOSTGAL_MAGERR_u', 'HOSTGAL_MAGERR_g', 'HOSTGAL_MAGERR_r', 'HOSTGAL_MAGERR_i', 'HOSTGAL_MAGERR_z', 'HOSTGAL_MAGERR_Y']
    other_features = static_feature_list
        
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
        print(f'MJD Range: {self.MJD[0]} - {self.MJD[-1]}')
        self.FLUXCAL = flux
        self.FLUXCALERR = flux_err
        self.BAND = filters
        
        self.PHOTFLAG = pl.Series([4096 if abs(self.FLUXCAL[i] / self.FLUXCALERR[i]) >= 5.0 else 0 for i in range(len(self.FLUXCAL))]) # detection flag if sigma >= 5
        try:
            self.PHOTFLAG[int(np.where(self.PHOTFLAG == 4096)[0][0])] = 6144
        except IndexError:
            pass
        
        # loading in static data
        static_data = data[1]
        
        for feature in static_data.keys():
            setattr(self, static_key_map[feature], static_data[feature])
        
        # extra LC processing
        self.process_lightcurve()
        self.compute_custom_features()
        
        # visualization
        self.ELASTICC_class = self.astrophysical_class
        self.plot_flux_curve(f'./TOM_LC_{self.SNID}.png')
    
    def process_lightcurve(self):
        saturation_mask =  (self.PHOTFLAG & 1024) == 0 

        # Alter time series data to remove saturations
        for time_series_feature in self.time_series_features:
            if time_series_feature != 'PHOTFLAG':
                setattr(self, time_series_feature, getattr(self, time_series_feature)[saturation_mask])
            else:
                setattr(self, time_series_feature, getattr(self, time_series_feature).filter(saturation_mask))
    
    def plot_flux_curve(self, save_path=None) -> None:
        """Plot the SNANA calibrated flux vs time plot for all the data in the processed time series. All detections are marked with a star while non detections are marked with dots. Observations are color codded by their passband. This function is fundamentally a visualization tool and is not intended for making plots for papers.
        """

        # Colorize the data
        c = [self.colors[band] for band in self.BAND]
        patches = [mpatches.Patch(color=self.colors[band], label=band, linewidth=1) for band in self.colors]
        fmts = np.where((self.PHOTFLAG & 4096) != 0, '*', '.')
        alpha = np.where((fmts == '.'), 0.1, 1)

        # Plot flux time series
        for i in range(len(self.MJD)):
            plt.errorbar(x=self.MJD[i], y=self.FLUXCAL[i], yerr=self.FLUXCALERR[i], color=c[i], fmt=fmts[i], alpha=alpha[i], markersize = '10')

        # Labels
        plt.title(f"SNID: {self.SNID} | CLASS: {self.ELASTICC_class} | z = {self.REDSHIFT_HELIO} | ra = {self.RA} | dec = {self.DEC}", wrap=True)
        plt.xlabel('Time (MJD)')
        plt.ylabel('Calibrated Flux')
        plt.legend(handles=patches)

        if save_path:
            plt.savefig(save_path)

        plt.show()


class preppedORACLE(ORACLE):
    def prep_dataframes(self, x_ts_list:List[pd.DataFrame]):

        # Assert that columns names are correct

        augmented_arrays = []

        for ind in tqdm(range(len(x_ts_list)), desc ="TS Processing: "):

            df = x_ts_list[ind]

            # # Scale the flux and flux error values
            # df['scaled_FLUXCAL'] = df['FLUXCAL'] / flux_scaling_const
            # df['scaled_FLUXCALERR'] = df['FLUXCALERR']/ flux_scaling_const

            # # Subtract off the time of first observation and divide by scale factor
            # df['scaled_time_since_first_obs'] = df['MJD'] / time_scaling_const

            # # Remove saturations
            # saturation_mask = np.where((df['PHOTFLAG'] & 1024) == 0)[0]
            # df = df.iloc[saturation_mask].copy()

            # # 1 if it was a detection, zero otherwise
            # df.loc[:,'detection_flag'] = np.where((df['PHOTFLAG'] & 4096 != 0), 1, 0)

            # # Encode pass band information correctly 
            # df['band_label'] = [pb_wavelengths[pb] for pb in df['BAND']]
            
            df = df[['scaled_time_since_first_obs', 'detection_flag', 'scaled_FLUXCAL', 'scaled_FLUXCALERR', 'band_label']]
            
            # Truncate array if too long
            arr = df.to_numpy()
            if arr.shape[0]>ts_length:
                arr = arr[:ts_length, :]

            augmented_arrays.append(arr)
            
        augmented_arrays = pad_sequences(augmented_arrays, maxlen=ts_length,  dtype='float32', padding='post', value=ts_flag_value)

        return augmented_arrays
    
    def prep_static_features(self, x_static_list:List[pd.DataFrame]):
        
        for i in tqdm(range(len(x_static_list)), desc ="Static Processing: "):        
            x_static_list[i] = get_static_features(x_static_list[i])
        
        if len(x_static_list) == 1:
            x_static_list = np.expand_dims(x_static_list[0], axis=0)
        else:
            x_static_list = np.squeeze(x_static_list)
        
        return x_static_list
    
    def predict(self, x_ts_list, x_static_list):
        x_ts_tensors = self.prep_dataframes(x_ts_list)
        x_static_tensors = self.prep_static_features(x_static_list)
    
        # Make predictions
        logits = self.model.predict([x_ts_tensors, x_static_tensors], verbose=0)
        _, pseudo_conditional_probabilities = get_conditional_probabilites(logits, self.tree)

        level_order_nodes = nx.bfs_tree(self.tree, source=source_node_label).nodes()
        columns_names =  list(level_order_nodes)

        df = pd.DataFrame(pseudo_conditional_probabilities, columns=columns_names)
        
        return df