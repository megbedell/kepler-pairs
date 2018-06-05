"""
Trim the output for find_pairs.py down and generate a table of only the best matches.
"""
import numpy as np
from astropy.table import Table, unique
from astropy.io import fits
from tqdm import tqdm
import pandas as pd
import h5py

if __name__ == '__main__':
    print("loading up everything...")
    gaia_table_file = '../data/gaia-kepler-dustin.fits'
    hdul = fits.open(gaia_table_file)
    gaia_src_tbl = Table(hdul[1].data)
    gaia_src_tbl = gaia_src_tbl.to_pandas()
    
    pairs_file = '../data/matched-pairs-dustin.fits'
    hdul = fits.open(pairs_file)
    pairs = hdul[0].data
    
    with h5py.File('chisqs.hdf5') as f:
        chisqs = np.copy(f['chisqs'])
    
    chisq_limit = 5.    
    print("loaded. masking down to chisq < {0:.1f}...".format(chisq_limit))
    
    plt.hist(chisqs_keep[(chisqs > 0.) & (chisqs < 100.)], bins=100)
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
        
    
        
    