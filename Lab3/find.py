import numpy as np
import pandas as pd

data = pd.read_pickle("fmri-labels.pkl")

# Check the type and shape
print("Type:", type(data))
print("Shape:", data.shape)

# View the first 5 elements
print("First 5 items:\n", data)

# If it's a 2D array, print more
if data.ndim == 2:
    print("Sample row:\n", data[0])