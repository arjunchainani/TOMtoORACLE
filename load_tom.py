import pandas as pd
import numpy as np
import requests

class TomClient:
    """A thin class that supports sending requests via "requests" to the DESC tom.

    Usage: initialize one of these, giving it the url, your TOM
    username, and either your TOM password, or a file that has your TOM
    password in it:

      tc = TomClient( username='rknop', passwordfile='/home/raknop/secrets/tom_rknop_passwd' )

    (You can give it a url with url=; it defaults to https://desc-tom.lbl.gov.)

    Thereafter, just do something like

      res = tc.request( "POST", "elasticc2/ppdbdiaobject/55772173" )

    and res will come back with a string that you can load into JSON
    that will have all the fun data about PPDBDiaObject number 55772173.

    tc.request is just a thin front-end to python requests.request.  The
    only reason to use this client rather than the python requests
    module directly is that this class takes care of the stupid fiddly
    bits of getting some headers that django demands set up right in the
    request object when you log in.

    """

    def __init__( self, url="https://desc-tom.lbl.gov", username=None, password=None, passwordfile=None, connect=True ):
        self._url = url
        self._username = username
        self._password = password
        self._rqs = None
        if self._password is None:
            if passwordfile is None:
                raise RuntimeError( "Must give either password or passwordfile. " )
            with open( passwordfile ) as ifp:
                self._password = ifp.readline().strip()

        if connect:
            self.connect()

    def connect( self ):
        self._rqs = requests.session()
        res = self._rqs.get( f'{self._url}/accounts/login/' )
        if res.status_code != 200:
            raise RuntimeError( f"Got status {res.status_code} from first attempt to connect to {self._url}" )
        res = self._rqs.post( f'{self._url}/accounts/login/',
                              data={ 'username': self._username,
                                     'password': self._password,
                                     'csrfmiddlewaretoken': self._rqs.cookies['csrftoken'] } )
        if res.status_code != 200:
            raise RuntimeError( f"Failed to log in; http status: {res.status_code}" )
        if 'Please enter a correct' in res.text:
            # This is a very cheesy attempt at checking if the login failed.
            # I haven't found clean documentation on how to log into a django site
            # from an app like this using standard authentication stuff.  So, for
            # now, I'm counting on the HTML that happened to come back when
            # I ran it with a failed login one time.  One of these days I'll actually
            # figure out how Django auth works and make a version of /accounts/login/
            # designed for use in API scripts like this one, rather than desgined
            # for interactive users.
            raise RuntimeError( "Failed to log in.  I think.  Put in a debug break and look at res.text" )
        self._rqs.headers.update( { 'X-CSRFToken': self._rqs.cookies['csrftoken'] } )

    def request( self, method="GET", page=None, **kwargs ):
        """Send a request to the TOM

        method : a string with the HTTP method ("GET", "POST", etc.)

        page : the page to get; this is the URL but with the url you
          passed to the constructor removed.  So, if you wanted to get
          https://desc-tom.lbl.gov/elasticc, you'd pass just "elasticc"
          here.

        **kwargs : additional keyword arguments are passed on to
          requests.request

        """
        return self._rqs.request( method=method, url=f"{self._url}/{page}", **kwargs )
        
    def post( self, page=None, **kwargs ):
        """Shortand for TomClient.request( "POST", ... )"""
        return self.request( "POST", page, **kwargs )

    def get( self, page=None, **kwargs ):
        """Shortand for TomClient.request( "GET", ... )"""
        return self.request( "GET", page, **kwargs )

    def put( self, page=None, **kwargs ):
        """Shortand for TomClient.request( "PUT", ... )"""
        return self.request( "PUT", page, **kwargs )

def load_oracle_features_from_TOM(
    num_objects: int,
    username: str, 
    passwordfile: str,
    detected_in_last_days: float,
    mjd_now: float,
    detected_since_mjd: float = None,
    cheat_gentypes: list = None,
    ):
    
    tom = TomClient(url = "https://desc-tom-2.lbl.gov", username = username, passwordfile = passwordfile)
    dic = {
        'detected_in_last_days': detected_in_last_days,
        'mjd_now': mjd_now
    }
    
    if detected_since_mjd is not None:
        dic['detected_since_mjd'] = detected_since_mjd
    
    if cheat_gentypes is not None:
        dic['cheat_gentypes'] = cheat_gentypes
    
    res = tom.post('elasticc2/gethottransients', json=dic)
    data = res.json() if res.status_code == 200 else res.status_code
    print('=> Fetched hot transients...')
    
    ids = [obj['objectid'] for obj in data['diaobject']]
    ids = ids[:num_objects] if len(ids) > num_objects else ids
    
    # using these object ids, load in static and time series data for ORACLE
    static = tom.post('db/runsqlquery',
                      json={'query': '''SELECT diaobject_id, ra, decl, mwebv, mwebv_err, z_final, z_final_err, hostgal_zphot, hostgal_zphot_err,
                  hostgal_zspec, hostgal_zspec_err, hostgal_ra, hostgal_dec, hostgal_snsep, hostgal_ellipticity, hostgal_mag_u,
                  hostgal_mag_g, hostgal_mag_r, hostgal_mag_i, hostgal_mag_z, hostgal_mag_y FROM elasticc2_ppdbdiaobject WHERE diaobject_id IN (%s) ORDER BY diaobject_id;''' % (', '.join(str(id) for id in ids)),
                       'subdict': {}})
    static_data = static.json() if static.status_code == 200 else static.status_code
    print('=> Loaded static data...')
    
    ts = tom.post('db/runsqlquery/',
                 json={'query': 'SELECT filtername, psflux, psfluxerr FROM elasticc2_ppdbdiasource WHERE diaobject_id IN (%s) ORDER BY diaobject_id;' % (', '.join(str(id) for id in ids)),
                      'subdict': {}})
    ts_data = ts.json() if ts.status_code == 200 else ts.status_code
    print('=> Loaded time-series data...')
    
    return type(static_data), type(ts_data)

if __name__ == '__main__':
    res = load_oracle_features_from_TOM(
        num_objects=5,
        username='arjun',
        passwordfile='./../oracle/passwordfile',
        detected_in_last_days=1,
        mjd_now=60800
    )
    
    print(res)