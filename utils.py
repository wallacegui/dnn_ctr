import pandas as pd
import numpy as np

max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x)+1)