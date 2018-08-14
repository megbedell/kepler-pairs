"""
Trim the output for find_pairs.py down and generate a table of only the best matches.
"""
import numpy as np
from astropy.table import Table, unique
from astropy.io import fits
import astropy.coordinates as coord
from astropy import units as u
import pandas as pd
import h5py
from schwimmbad import MPIPool
import sys
import os.path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from find_pairs_mpi import make_x, make_cov

# common variables:
gaia_table_file = '../data/gaia-kepler-dustin.fits'
pairs_file = '../data/matched-pairs-dustin.fits'


def calc_chisq_nonzero(star):
    """
    Chisquared-like metric to diagnose how different from zero the proper motions
    Does NOT take parallax into account
    """
    x = make_x(star)[1:]
    cov = make_cov(star)[1:,1:]
    return np.dot(x, np.linalg.solve(cov, x))

def worker_chisqnonzero(data):
    """
    Require that both stars have non-zero proper motions
    """
    j,(i1,i2) = data
    chisq_nonzero1 = calc_chisq_nonzero(gaia_src_tbl.iloc[i1])
    chisq_nonzero2 = calc_chisq_nonzero(gaia_src_tbl.iloc[i2])
    if (chisq_nonzero1 > chisq_nonzero_limit) and (chisq_nonzero2 > chisq_nonzero_limit):
        return j, True
    else:
        return j, False
        
def photometry_check(star, primary=True):
    if primary:
        tol = 0.05 # % error allowed in Bp, Rp
    else:
        tol = 0.1
    if (star.loc['phot_g_mean_flux_over_error'] < 1./0.02) \
       or (star.loc['phot_bp_mean_flux_over_error'] < 1./tol) \
       or (star.loc['phot_rp_mean_flux_over_error'] < 1./tol):
        return False
    return True
    
def separation_check(star1, star2):
    coord1 = coord.SkyCoord(ra=star1.loc['ra'] * u.deg, dec=star1.loc['dec'] * u.deg)
    coord2 = coord.SkyCoord(ra=star2.loc['ra'] * u.deg, dec=star2.loc['dec'] * u.deg)
    if coord1.separation(coord2) > 50. * u.mas:
        return False
    return True
        
def worker_elbadry(data):
    """
    Apply the requirements listed in El Badry & Rix (2018)
    """
   j,(i1,i2) = data
   star1, star2 = gaia_src_tbl.iloc[i1], gaia_src_tbl.iloc[i2]
   # determine which is "primary" (brighter member)
   if star1.loc['phot_g_mean_mag'] < star2.loc['phot_g_mean_mag']:
       primary = star1
       secondary = star2
   else:
       primary = star2
       secondary = star1
   # primary nearby:
   if (primary.loc['parallax'] < 5.):
       return j, False
       # *** NOTE ***: code as currently written will falsely throw out the case where both stars have plx > 5
       # and the fainter star is better-measured (passes primary star requirements while brighter fails)       
   # high SNR photometry:
   if ~photometry_check(primary, primary=True) or ~photometry_check(secondary, primary=False):
       return j, False
   # high SNR parallaxes:
   if (primary.loc['parallax_over_error'] < 20.) or (secondary.loc['parallax_over_error'] < 5.):
       return j, False
   # separation < 5e4 AU:
   if ~separation_check(primary, secondary):
       return j, False
   return j, True 

def callback(results):
    j, b = results
    if (j%1e6) == 0:
        print("{0}th pair calculated".format(j))
    if b:
        with h5py.File(mask_file, 'r+') as f:
            f['mask'][j] = b
        
def main(pool, worker_func=None, matches_file='matches_chisqlt5.hdf5', 
            mask_file='mask.hdf5', good_sources_file='good_sources.pkl'):
    """
    Main function for MPIPool
    see example: https://schwimmbad.readthedocs.io/en/latest/examples/index.html#using-mpipool
    """
    print("starting the pool...")
    
    with h5py.File(matches_file, 'r') as f:
        pairs_ind1s = np.copy(f['pairs_ind1s'])
        pairs_ind2s = np.copy(f['pairs_ind2s'])
        chisqs = np.copy(f['chisqs'])
    
    # make the output file
    with h5py.File(mask_file, 'w') as f:
        f.create_dataset('mask', data=np.zeros(len(pairs_ind1s), dtype=bool))

    tasks = list(enumerate(zip(pairs_ind1s, pairs_ind2s)))
    print("tasks constructed")
    
    # run
    results = pool.map(worker_func, tasks, callback=callback)
    pool.close()
    
    # apply mask
    with h5py.File(mask_file, 'r+') as f:
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
    
    hdul = fits.open(gaia_table_file)  # idk why this needs to be reloaded??
    gaia_src_tbl = Table(hdul[1].data)
    gaia_src_tbl = gaia_src_tbl.to_pandas()
    
    gaia_to_save = gaia_src_tbl.iloc[inds]
    gaia_to_save.to_pickle(good_sources_file)



if __name__ == '__main__':
    
    chisq_limit = 5.
    matches_file = 'matches_chisqlt{0}.hdf5'.format(int(chisq_limit))
           
    if not os.path.isfile(matches_file):
        print("loading up the pairs...")
        hdul = fits.open(pairs_file)
        pairs = hdul[0].data
    
        with h5py.File('chisqs.hdf5') as f:
            chisqs = np.copy(f['chisqs'])
    
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
        with h5py.File(matches_file, 'w') as f:
            f.create_dataset('pairs_ind1s', data=pairs_ind1s)
            f.create_dataset('pairs_ind2s', data=pairs_ind2s)
            f.create_dataset('chisqs', data=chisqs)


    print("loading up Gaia sources...")
    hdul = fits.open(gaia_table_file)
    gaia_src_tbl = Table(hdul[1].data)
    gaia_src_tbl = gaia_src_tbl.to_pandas()
    
    if True: # mask out objects with PM consistent with zero
        chisq_nonzero_limit = 25.
        mask_file = 'matches_chisqlt{0}_nzlt{1}mask.hdf5'.format(int(chisq_limit), int(chisq_nonzero_limit))
        good_sources_file = 'sources_chisqlt{0}_nzlt{1}mask.pkl'.format(int(chisq_limit), int(chisq_nonzero_limit))
        
        print("dropping objects with chisq_nonzero < {0}...".format(chisq_nonzero_limit))       
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        main(pool, worker_func=worker_chisqnonzero, matches_file=matches_file, 
                    mask_file=mask_file, good_sources_file=good_sources_file)  
    
    if True: # apply El-Badry cuts
        mask_file = 'matches_elbadry.hdf5'
        good_sources_file = 'sources_elbadry.pkl'
        
        print("dropping objects using El-Badry cuts...")
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        main(pool, worker_func=worker_elbadry, mask_file=mask_file, matches_file=matches_file, 
                    mask_file=mask_file, good_sources_file=good_sources_file) 