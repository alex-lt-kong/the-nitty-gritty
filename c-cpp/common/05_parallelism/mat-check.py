import numpy as np
import pandas as pd
import argparse


def main() -> None:
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process two strings and a float')

    # Add the arguments
    parser.add_argument('--csv1', '-c1', type=str, help='Path of 1st csv')
    parser.add_argument('--csv2', '-c2', type=str, help='Path of 2nd csv')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-3, help='tolerance')

    # Parse the arguments
    args = parser.parse_args()

    print(f'Loading df1 from [{args.csv1}]...')
    df1 = pd.read_csv(args.csv1, header=None)
    print(f'Done\nLoading df2 from [{args.csv2}]...')
    df2 = pd.read_csv(args.csv2, header=None)
    print('Done')

    print('Comparing two DataFrames...')
    assert df1.shape == df2.shape, "The shape of df1 and df2 are different!"

    diff_count = 0
    for index, diff in np.ndenumerate(np.abs(df1 - df2)):
        if diff >= args.tolerance:
            print(f"Difference found at index {index} with diff {diff:05f}")
            diff_count += 1
            if diff_count >= 10:
                print('mat-check exited due to too many mismatches')
                print('A breakpoint is triggered and you can examine them manually')
                breakpoint()
                break
    if diff_count == 0:
        print('The dataframes are identical within the tolerance level')


if __name__ == '__main__':
    main()
