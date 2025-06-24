from load_tom import load_oracle_features_from_TOM
from process_sources import TOM_Source, preppedORACLE
import argparse

parser = argparse.ArgumentParser(description='Fetch latest transients from TOM and classify them through ORACLE')
parser.add_argument('-n', '--num_objects', type=int, metavar='', default=5, help='Number of transients to classify')
parser.add_argument('-u', '--username', type=str, metavar='', required=True, help='TOM Username')
parser.add_argument('-p', '--passwordfile', type=str, metavar='', required=True, help='TOM Password File Path')
parser.add_argument('-d', '--detected_in_last_days', type=int, metavar='', default=1, help='How many nights to look back in time for finding transients')
parser.add_argument('-m', '--mjd_now', type=int, metavar='', default=60800, help='Current MJD')
parser.add_argument('-mp', '--model_path', type=str, metavar='', required=True, help='Path to ORACLE model weights file')
args = parser.parse_args()

if __name__ == '__main__':
    tom_data = load_oracle_features_from_TOM(
        num_objects=args.num_objects,
        username=args.username,
        passwordfile=args.passwordfile,
        detected_in_last_days=args.detected_in_last_days, 
        mjd_now=args.mjd_now
    )
    
    model = preppedORACLE(args.model_path)
    
    for object in tom_data:
        source = TOM_Source(object)
        table = source.get_event_table()
        print(f'\n\n--------------------SNID: {source.SNID}------------------------\n')
        table.pprint_all()
        pred = model.predict([table.to_pandas()], [table.meta])
        print(f'\n\nTrue Class: {source.astrophysical_class}')
        print(pred)
        print(f'Predicted Class: {model.predict_classes([table.to_pandas()], [table.meta])}')