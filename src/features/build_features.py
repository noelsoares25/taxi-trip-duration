import pathlib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from feature_definitions import feature_build


def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)


if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    train_path = home_dir.as_posix() + '/data/raw/train.csv' 
    test_path = home_dir.as_posix() + '/data/raw/test.csv'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    output_path = home_dir.as_posix() + '/data/processed'
    
    train_data = feature_build(train_data, 'train-data')
    test_data = feature_build(test_data, 'test-data')
    
    do_not_use_for_training = ['id', 'pickup_datetime', 'dropoff_datetime', 'check_trip_duration', 'pickup_date', 'avg_speed_h', 'avg_speed_m',
                               'pickup_lat_bin', 'pickup_long_bin', 'center_lat_bin', 'center_long_bin', 'pickup_dt_bin', 'pickup_datetime_group']
    
    feature_names = [f for f in train_data.columns if f not in do_not_use_for_training]
    print("We have %i features." % len(feature_names))
    
    train_data = train_data[feature_names]
    test_data = test_data[feature_names]
    
    save_data(train_data, test_data, output_path)