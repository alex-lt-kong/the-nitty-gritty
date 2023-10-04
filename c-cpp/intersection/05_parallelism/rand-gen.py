import argparse
import numpy as np

# Define the command line argument parser
parser = argparse.ArgumentParser(description='Generate random doubles and save them to a file.')
parser.add_argument('--save-to', '-s', type=str, default='random_doubles.txt',
                    help='File name to save the random doubles.')
parser.add_argument('--num-ele', '-n', type=int, default=10, help='Number of doubles to generate')
args = parser.parse_args()

random_nums = np.random.uniform(low=-10, high=10, size=args.num_ele)

# Save the random numbers to a file specified by the command line argument
with open(args.save_to, 'w') as f:
    for num in random_nums:
        f.write(str(num) + '\n')

print('Random doubles saved to file:', args.save_to)
