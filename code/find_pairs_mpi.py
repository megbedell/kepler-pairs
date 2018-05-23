import numpy as np
from astropy.table import Table, unique
from astropy import units as u
from astropy.io import fits
from tqdm import tqdm
import pandas as pd
from schwimmbad import SerialPool, MultiPool, MPIPool
import h5py
import time

min_columns = ['source_id', 'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'parallax_error',
                'pmra_error', 'pmdec_error', 'parallax_pmra_corr', 'parallax_pmdec_corr',
                'pmra_pmdec_corr']

def construct_table(gaia_table_file, kepler_table_file=None, minimal=True):
    hdul = fits.open(gaia_table_file)
    gaia_src_tbl = Table(hdul[1].data)
    if minimal:
        gaia_src_tbl = gaia_src_tbl[min_columns]
        return gaia_src_tbl.to_pandas()
    elif kepler_table_file is None:
        print('must supply kepler_table_file kwarg for non-minimal return')
    gaia_src_tbl = gaia_src_tbl.to_pandas()
    hdul = fits.open(kepler_table_file)
    kepler_tbl = Table(hdul[1].data)
    gaia_kepler_matches = kepler_tbl['kepid', 'kepler_gaia_ang_dist', 'source_id', 'nconfp', 'nkoi', 'planet?']
    gaia_kepler_matches = gaia_kepler_matches.to_pandas()
    gaia_kepler_matches.sort_values(['kepid', 'kepler_gaia_ang_dist'], inplace=True)
    gaia_kepler_matches.drop_duplicates('kepid', inplace=True)
    table = gaia_src_tbl.merge(gaia_kepler_matches, on='source_id', how='left')
    return table

def read_from_fits(filename):
    hdul = fits.open(filename)
    return hdul[0].data

def make_x(star):
    """
    returns a vector of x = [parallax, pmra, pmdec]
    """
    names = ['parallax', 'pmra', 'pmdec']
    return star.loc[names].values.astype('f')

def make_xerr(star):
    """
    returns a vector of xerr = [parallax_error, pmra_error, pmdec_error]
    """
    err_names = ['parallax_error', 'pmra_error', 'pmdec_error']
    return star.loc[err_names].values.astype('f')

def ppm_check(star1, star2, sigma=5.):
    """
    Returns True if the differences between parallax, pmra, and pmdec are all below
    the sigma threshold.
    """
    x1 = make_x(star1)
    x2 = make_x(star2)
    if np.any(np.isnan([x1,x2])):
        return False
    xerr1 = make_xerr(star1)
    xerr2 = make_xerr(star2)
    if np.any(np.isnan([xerr1, xerr2])):
        return False
    if np.any(np.abs(x1 - x2)/np.sqrt(xerr1**2 + xerr2**2) >= sigma):
        return False
    return True

def make_cov(star):
    """
    returns covariance matrix C corresponding to x
    """
    names = ['parallax', 'pmra', 'pmdec']
    C = np.diag(make_xerr(star)**2)
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if j >= i:
                continue
            corr = star.loc["{0}_{1}_corr".format(name2, name1)]
            C[i, j] = corr * np.sqrt(C[i, i] * C[j, j])
            C[j, i] = C[i, j]
    return C

def chisq(star1, star2):
    """
    calculates chisquared for two stars based on their parallax and 2D proper motions
    """
    deltax = make_x(star1) - make_x(star2)
    cplusc = make_cov(star1) + make_cov(star2)
    return np.dot(deltax, np.linalg.solve(cplusc, deltax))
    
def calc_chisq_nonzero(star):
    """
    Chisquared-like metric to diagnose how different from zero the proper motions
    Does NOT take parallax into account
    """
    x = make_x(star)[1:]
    cov = make_cov(star)[1:,1:]
    return np.dot(x, np.linalg.solve(cov, x))

def calc_chisq_for_pair(m, primary):
    if ppm_check(primary, m):
        return chisq(primary, m)
    else:
        return -1

def calc_chisqs_for_row(i,row):
    row_of_chisqs = np.zeros_like(row) - 1.
    primary = table.iloc[i]
    row_mask = (row > -1) & (row > i) # indices in row for matches to compute
    matches = table.iloc[row[row_mask]] # ignore non-matches and duplicates
    if np.sum(row_mask) > 0:
        row_of_chisqs[row_mask] = matches.apply(calc_chisq_for_pair, args=(primary,), axis=1)
    return row_of_chisqs

def worker(data):
    """
    Wrapper function for parallelization
    """
    i, row = data
    row_of_chisqs = calc_chisqs_for_row(i, row)
    return i, row_of_chisqs

def callback(results):
    """
    Save chisquared row
    """
    i, row_of_chisqs = results
    if (i % 1e6) == 0:
        print('{0}th row finished, {1:.2f} min elapsed'.format(i, 
              (time.time() - map_start)/60.))
    with h5py.File('chisqs.hdf5', 'r+') as f:
        f['chisqs'][i,:] = row_of_chisqs
        
def main(pool):
    """
    Main function for MPIPool
    see example: https://schwimmbad.readthedocs.io/en/latest/examples/index.html#using-mpipool
    """
    # make the output file
    with h5py.File('chisqs.hdf5', 'w') as f:
        chisqs = np.zeros_like(pairs) - 1.
        dset = f.create_dataset('chisqs', data=chisqs)
         
    # construct tasks list   
    tasks_start = time.time()
    tasks = list(enumerate(pairs))
    tasks_end = time.time()
    print("constructing tasks took {0} s".format(tasks_end - tasks_start))
    
    # run
    results = pool.map(worker, tasks, callback=callback)
    pool.close()

if __name__ == '__main__':
    print("loading data...")
    start = time.time()
    gaia_table_file = '../data/gaia-kepler-dustin.fits'
    table = construct_table(gaia_table_file, minimal=True) # table is a global variable
    print("loading data table took {0} s".format(time.time() - start))

    print("loading pair indices...")
    pairs_start = time.time()
    pairs_file = '../data/matched-pairs-dustin.fits'
    pairs = read_from_fits(pairs_file) # pairs is a global variable
    print("loading pairs array took {0} s".format(time.time() - pairs_start))

    print("calculating chisquared...")
    pool_start = time.time()
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    main(pool)
    pool_end = time.time()
    print("pool execution took {0} min".format((pool_end - pool_start)/60.))
    
    print("calculating individual non-zero PM chisquared...")
    nonzero_start = time.time()
    chisqs_nonzero = table.apply(calc_chisq_nonzero, axis=1)
    with h5py.File('chisqs_nonzero.hdf5', 'w') as f:
        dset = f.create_dataset('chisqs_nonzero', data=chisqs_nonzero)
    nonzero_end = time.time()
    print("calculation took {0} s".format(nonzero_end - nonzero_start))
    
    print("total execution took {0} s".format(time.time() - start))
