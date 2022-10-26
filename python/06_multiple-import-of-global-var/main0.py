import lib
import main2
import main1
import multiprocessing as mp
import sys
import time

print(f'main0.py, var is {lib.get_var()}')

def print_func(continent='Asia'):
    print('The name of continent is : ', sys.argv)
    time.sleep(60)
    print('done!')


if __name__ == "__main__":  # confirms that the code is under main function
    names = ['America', 'Europe', 'Africa']
    procs = []
    proc = mp.Process(target=print_func)  # instantiating without any argument
    procs.append(proc)
    proc.start()

    # instantiating process with arguments
    for name in names:
        # print(name)
        proc = mp.Process(target=print_func, args=(name,))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
