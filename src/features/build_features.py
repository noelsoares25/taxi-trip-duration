import pathlib
import pandas as pd
import numpy as np

from feature_definitions import feature_build

if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    train_path = home_dir.as_posix() + '/data/raw/train.csv' 
    test_path = home_dir.as_posix() + '/data/raw/test.csv'
    train_date = pd.read_csv(train_path)
    test_date = pd.read_csv(test_path)