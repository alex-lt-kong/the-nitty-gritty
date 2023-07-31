import numpy as np
import pandas as pd

print('Loading df1...')
df1 = pd.read_csv('./np.csv.out', header=None)
print('Done\nLoading df2...')
df2 = pd.read_csv('./openblas.csv.out', header=None)

breakpoint()

print('Comparing two DataFrames...')
tolerance = 1e-2
diff_count = 0
for index, diff in np.ndenumerate(np.abs(df1 - df2)):
    if diff >= tolerance:
        print(f"Difference found at index {index} with diff {diff:03f}")
        diff_count += 1
        if diff_count >= 10:
            break
if diff_count == 0:
    print('The dataframes are identical within the tolerance level')
