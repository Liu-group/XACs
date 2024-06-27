from multiprocessing import Process, Pool, cpu_count
import os

def info(title):
    print(title)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(i, j):
    info('function f')
    return {'a':i}, {'b':j}


if __name__ == '__main__':
    info('main line')
    nproc = cpu_count()
    print('cpu count:', nproc)
    pool = Pool(nproc)
    asyncresults = []
    for i in range(45):
        for j in range(i + 1, 45):
            asyncresults.append([i, j] + [pool.apply_async(f, args=(i,j)).get()])
    for asyncresult in asyncresults:
        a, b, (c, d) = asyncresult
        print(a, b, c, d)