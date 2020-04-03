import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Random Samples')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1)
    parser.add_argument('--sample_size', type=int, default=1)
    args = parser.parse_args()

    rand_list = np.random.randint(low=args.start, high=args.end+1, size=args.sample_size)

    subval_file = open('subval_argo.txt', 'w')
    for idx in rand_list:
        print(idx)
        subval_file.write(str(idx).zfill(6))
        subval_file.write('\n')

    subval_file.close()
    print('Done')