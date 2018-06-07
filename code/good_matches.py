"""
Trim the output for find_pairs.py down and generate a table of only the best matches.
"""
import numpy as np
from astropy.table import Table, unique
from astropy.io import fits
#from tqdm import tqdm
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from find_pairs_mpi import make_x, make_cov, calc_chisq_nonzero

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
        with h5py.File('matches_chisqlt{0}.fits'.format(int(chisq_limit)), 'w') as f:
            f.create_dataset('pairs_ind1s', data=pairs_ind1s)
            f.create_dataset('pairs_ind2s', data=pairs_ind2s)
            f.create_dataset('chisqs', data=chisqs)
    else:
        with h5py.File('matches_chisqlt5.fits', 'r') as f:
            pairs_ind1s = np.copy(f['pairs_ind1s'])
            pairs_ind2s = np.copy(f['pairs_ind2s'])
            chisqs = np.copy(f['chisqs'])

    print("loading up Gaia sources...")
    gaia_table_file = '../data/gaia-kepler-dustin.fits'
    hdul = fits.open(gaia_table_file)
    gaia_src_tbl = Table(hdul[1].data)
    gaia_src_tbl = gaia_src_tbl.to_pandas()
    
    chisq_nonzero_limit = 25.
    print("dropping objects with chisq_nonzero < {0}...".format(chisq_nonzero_limit))

    keep_pairs = []
    for j,(i1,i2) in enumerate(zip(pairs_ind1s, pairs_ind2s)):
        if (j % 100000) == 0:
            print("{0}/{1} pairs checked; {2} good pairs found so far.".format(j,
                                        len(pairs_ind1s),len(keep_pairs)))
        chisq_nonzero1 = calc_chisq_nonzero(gaia_src_tbl.iloc[i1])
        chisq_nonzero2 = calc_chisq_nonzero(gaia_src_tbl.iloc[i2])
        if (chisq_nonzero1 > chisq_nonzero_limit) and (chisq_nonzero2 > chisq_nonzero_limit):
            keep_pairs.append(j)

    print("{0} pairs meet these criteria. saving outputs...".format(len(keep_pairs)))

    pairs_ind1s = pairs_ind1s[keep_pairs]
    pairs_ind2s = pairs_ind2s[keep_pairs]
    chisqs = chisqs[keep_pairs]
    with h5py.File('good_pairs.fits', 'w') as f:
        f.create_dataset('pairs_ind1s', data=pairs_ind1s)
        f.create_dataset('pairs_ind2s', data=pairs_ind2s)
        f.create_dataset('chisqs', data=chisqs)

    inds = np.unique(np.append(pairs_ind1s, pairs_ind2s))
    gaia_to_save = Table(gaia_src_table.iloc[inds])
    gaia_to_save.write('good_gaia_sources.fits', format='fits', overwrite=True)
        
    
        
    
