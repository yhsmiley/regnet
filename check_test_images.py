import os
import pandas as pd

test_dir = '/dataset/test/'

test_csv = '/dataset/test.csv'
test_df = pd.read_csv(test_csv)
csv_filenames = test_df['filename'].tolist()

for filename in os.listdir(test_dir):
    if filename not in csv_filenames:
        print(filename)
