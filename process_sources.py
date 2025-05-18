import tqdm
from astropy.table import Table
from astroOracle.pretrained_models import ORACLE, ORACLE_lite
from astroOracle.LSST_Source import LSST_Source
from load_tom import load_oracle_features_from_TOM

# a map from the TOM feature names to the ORACLE feature names
key_map = {
    ''
}

# wrapper for the LSST_Source class that can take in manual input rather than just from parquet
class TOM_Source(LSST_Source):
    def __init__(self):
        pass