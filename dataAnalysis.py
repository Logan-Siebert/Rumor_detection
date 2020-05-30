import pandas as pd
import numpy as np

pd.set_option('expand_frame_repr', True)
data = pd.read_csv("expData.csv", sep = " ")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(data)
