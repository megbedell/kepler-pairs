"""
Trim the output for find_pairs.py down and generate a table of only the best matches.
"""
import numpy as np
from astropy.table import Table, unique
from astropy.io import fits
import pandas as pd
from schwimmbad import SerialPool, MultiPool, MPIPool
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from find_pairs_mpi import make_x, make_cov
import sys

def calc_chisq_nonzero(star):
    """
    Chisquared-like metric to diagnose how different from zero the proper motions
    Does NOT take parallax into account
    """
    x = make_x(star)[1:]
    cov = make_cov(star)[1:,1:]
    return np.dot(x, np.linalg.solve(cov, x))

def worker(data):
    """
    Wrapper function for parallelization
    """
    j,(i1,i2) = data
    chisq_nonzero1 = calc_chisq_nonzero(gaia_src_tbl.iloc[i1])
    chisq_nonzero2 = calc_chisq_nonzero(gaia_src_tbl.iloc[i2])
    if (chisq_nonzero1 > chisq_nonzero_limit) and (chisq_nonzero2 > chisq_nonzero_limit):
        return j, True
    else:
        return j, False

def callback(results):
    j, b = results
    if (j%1e6) == 0:
        print("{0}th pair calculated".format(j))
    if b:
        with h5py.File(filename, 'r+') as f:
            f['mask'][j] = b
        
def main(pool):
    """
    Main function for MPIPool
    see example: https://schwimmbad.readthedocs.io/en/latest/examples/index.html#using-mpipool
    """
    print("starting the pool...")
    
    with h5py.File('matches_chisqlt5.hdf5', 'r') as f:
        pairs_ind1s = np.copy(f['pairs_ind1s'])
        pairs_ind2s = np.copy(f['pairs_ind2s'])
        chisqs = np.copy(f['chisqs'])
    
    # make the output file
    with h5py.File(filename, 'w') as f:
        f.create_dataset('mask', data=np.zeros(len(pairs_ind1s), dtype=bool))

    tasks = list(enumerate(zip(pairs_ind1s, pairs_ind2s)))
    print("tasks constructed")
    
    # run
    results = pool.map(worker, tasks, callback=callback)
    pool.close()
    
    # apply mask
    with h5py.File(filename, 'r+') as f:
        mask = np.copy(f['mask'])
        
    print("{0} pairs meet these criteria. saving outputs...".format(np.sum(mask)))

    pairs_ind1s = pairs_ind1s[mask]
    pairs_ind2s = pairs_ind2s[mask]
    chisqs = chisqs[mask]
    with h5py.File('good_pairs.hdf5', 'w') as f:
        f.create_dataset('pairs_ind1s', data=pairs_ind1s)
        f.create_dataset('pairs_ind2s', data=pairs_ind2s)
        f.create_dataset('chisqs', data=chisqs)

    inds = np.unique(np.append(pairs_ind1s, pairs_ind2s))
    
    gaia_table_file = '../data/gaia-kepler-dustin.fits' # idk why this needs to be reloaded??
    hdul = fits.open(gaia_table_file)
    gaia_src_tbl = Table(hdul[1].data)
    gaia_src_tbl = gaia_src_tbl.to_pandas()
    
    gaia_to_save = gaia_src_tbl.iloc[inds]
    gaia_to_save.to_pickle('../code/good_gaia_sources.pkl')



if __name__ == '__main__':
    if False:
        print("loading up the pairs...")
        pairs_file = '../data/matched-pairs-dustin.fits'
        hdul = fits.open(pairs_file)
        pairs = hdul[0].data
    
        with h5py.File('chisqs.hdf5') as f:
            chisqs = np.copy(f['chisqs'])
    
        chisq_limit = 5.    
        print("loaded. masking down to chisq < {0:.1f}...".format(chisq_limit))
    
        plt.hist(chisqs[(chisqs > 0.) & (chisqs < 100.)], bins=100)
        plt.xlabel('$\chi^2$', fontsize=16)
        plt.ylabel('# Pairs', fontsize=16)
        plt.yscale('log')
        plt.savefig('chisq_keplerpairs.png')
    
        matches_mask = (chisqs > 0) & (chisqs < chisq_limit)
        print("{0} pairs fit these criteria".format(np.sum(matches_mask)))
    
        print("saving indices of best matches...")
        len_inds, len_matches = np.shape(pairs)
        pairs_inds = np.array([np.arange(len_inds),]*len_matches).transpose()
        pairs_ind1s = pairs_inds[matches_mask]
        pairs_ind2s = pairs[matches_mask]
        chisqs = chisqs[matches_mask]
        with h5py.File('matches_chisqlt{0}.hdf5'.format(int(chisq_limit)), 'w') as f:
            f.create_dataset('pairs_ind1s', data=pairs_ind1s)
            f.create_dataset('pairs_ind2s', data=pairs_ind2s)
            f.create_dataset('chisqs', data=chisqs)


    print("loading up Gaia sources...")
    gaia_table_file = '../data/gaia-kepler-dustin.fits'
    hdul = fits.open(gaia_table_file)
    gaia_src_tbl = Table(hdul[1].data)
    gaia_src_tbl = gaia_src_tbl.to_pandas()
    
    chisq_nonzero_limit = 25.
    print("dropping objects with chisq_nonzero < {0}...".format(chisq_nonzero_limit))
    
    filename = 'matches_chisqlt5_nzlt{0}mask.hdf5'.format(int(chisq_nonzero_limit))
        
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    main(pool)        
