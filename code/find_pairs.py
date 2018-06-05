import numpy as np
from astropy.table import Table, unique
from astropy import units as u
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from schwimmbad import SerialPool, MultiPool, MPIPool
import h5py
#from plot_tools import error_ellipse
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

def save_as_fits(filename, data):
    print("saving as {0}...".format(filename))
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)
    hdul.close()

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

def calc_chisqs_for_table(table, pairs, save=False, save_every=1e6, save_name='chisqs.fits'):
    chisqs = np.zeros_like(pairs) - 1.
    for i,row in tqdm(enumerate(pairs)):
        chisqs[i] = calc_chisqs_for_row(i,row)
        if save and (i % save_every == 1):
            save_as_fits(save_name, chisqs)
    if save:
        save_as_fits(save_name, chisqs)
    return chisqs

def worker(data):
    """
    Wrapper function for parallelization of pairs chisquared
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

def read_match_attr(table, ind1, ind2, attr):
    return table.iloc[ind1][attr], table.iloc[ind2][attr]

def plot_xs(table, i, sigma=1):
    fs = 12
    star1 = table.iloc[pairs_ind1s[i]]
    star2 = table.iloc[pairs_ind2s[i]]
    x1 = make_x(star1)
    cov1 = make_cov(star1)
    x2 = make_x(star2)
    cov2 = make_cov(star2)
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(131)
    error_ellipse(ax1, x1[0], x1[1], cov1[:2,:2], ec='red', sigma=sigma)
    error_ellipse(ax1, x2[0], x2[1], cov2[:2,:2], ec='blue', sigma=sigma)
    ax1.set_xlim([min([x1[0], x2[0]]) - 5., max([x1[0], x2[0]]) + 5.])
    ax1.set_ylim([min([x1[1], x2[1]]) - 5., max([x1[1], x2[1]]) + 5.])
    ax1.set_xlabel('Parallax (mas)', fontsize=fs)
    ax1.set_ylabel('PM RA (mas yr$^{-1}$)', fontsize=fs)

    ax2 = fig.add_subplot(133)
    error_ellipse(ax2, x1[1], x1[2], cov1[1:,1:], ec='red', sigma=sigma)
    error_ellipse(ax2, x2[1], x2[2], cov2[1:,1:], ec='blue', sigma=sigma)
    ax2.set_xlim([min([x1[1], x2[1]]) - 5., max([x1[1], x2[1]]) + 5.])
    ax2.set_ylim([min([x1[2], x2[2]]) - 5., max([x1[2], x2[2]]) + 5.])
    ax2.set_xlabel('PM RA (mas yr$^{-1}$)', fontsize=fs)
    ax2.set_ylabel('PM Dec (mas yr$^{-1}$)', fontsize=fs)

    ax3 = fig.add_subplot(132)
    c1 = np.delete(np.delete(cov1, 1, axis=0), 1, axis=1)
    c2 = np.delete(np.delete(cov2, 1, axis=0), 1, axis=1)
    error_ellipse(ax3, x1[0], x1[2], c1, ec='red', sigma=sigma)
    error_ellipse(ax3, x2[0], x2[2], c2, ec='blue', sigma=sigma)
    ax3.set_xlim([min([x1[0], x2[0]]) - 5., max([x1[0], x2[0]]) + 5.])
    ax3.set_ylim([min([x1[2], x2[2]]) - 5., max([x1[2], x2[2]]) + 5.])
    ax3.set_xlabel('Parallax (mas)', fontsize=fs)
    ax3.set_ylabel('PM Dec (mas yr$^{-1}$)', fontsize=fs)

    fig.subplots_adjust(wspace = 0.5)
    fig.text(0.5, 0.95, 'match #{0}'.format(i), horizontalalignment='center',
             transform=ax3.transAxes, fontsize=fs+2)

if __name__ == '__main__':
    print("loading data...")
    start = time.time()
    gaia_table_file = '../data/gaia-kepler-dustin.fits'
    kepler_table_file = '../data/kepler_dr2_1arcsec.fits'
    table = construct_table(gaia_table_file, minimal=True) # table is a global variable
    print("loading data table took {0} s".format(time.time() - start))

    print("loading pair indices...")
    pairs_start = time.time()
    pairs_file = '../data/matched-pairs-dustin.fits'
    pairs = read_from_fits(pairs_file) # pairs is a global variable
    #pairs = pairs[:1000] # TEMPORARY FOR TESTING
    print("loading pairs array took {0} s".format(time.time() - pairs_start))

    print("calculating pairs chisquared...")
    with h5py.File('chisqs.hdf5', 'w') as f:
        chisqs = np.zeros_like(pairs) - 1.
        dset = f.create_dataset('chisqs', data=chisqs)

    tasks_start = time.time()
    tasks = list(enumerate(pairs))
    tasks_end = time.time()
    print("constructing tasks took {0} s".format(tasks_end - tasks_start))    

    pool = MultiPool(processes=16)
    map_start = time.time()
    results = pool.map(worker, tasks, callback=callback)
    map_end = time.time()
    print("mapping took {0} hr".format((map_end - map_start)/3600.))
    pool.close()
    
    print("calculating individual non-zero PM chisquared...")
    nonzero_start = time.time()
    chisqs_nonzero = table.apply(calc_chisq_nonzero, axis=1)
    with h5py.File('chisqs_nonzero.hdf5', 'w') as f:
        dset = f.create_dataset('chisqs_nonzero', data=chisqs_nonzero)
    nonzero_end = time.time()
    print("calculation took {0} s".format(nonzero_end - nonzero_start))  
    

    if False: # basic diagnostics
        with h5py.File('chisqs.hdf5', 'r+') as f:
            chisqs = np.copy(f['chisqs'])
        
        plt.hist(chisqs[(chisqs > 0.) & (chisqs < 50.)], bins=500)
        plt.xlabel('$\chi^2$', fontsize=16)
        plt.ylabel('# Pairs', fontsize=16)
        plt.yscale('log')
        plt.savefig('chisq_keplerpairs.png')

        matches_mask = (chisqs > 0) & (chisqs < 5)
        print("{0} matches found with chisq < 5".format(np.sum(matches_mask)))
        len_inds, len_matches = np.shape(pairs)
        pairs_inds = np.array([np.arange(len_inds),]*len_matches).transpose()
        pairs_ind1s = pairs_inds[matches_mask]
        pairs_ind2s = pairs[matches_mask]

        i = np.random.randint(0, len(pairs_ind1s))
        print("match {0}: source_ids {1}".format(i,
                    read_match_attr(table, pairs_ind1s[i], pairs_ind2s[i], 'source_id')))
        print("saved chisquared = {0:.5f}".format(chisqs[pairs_ind1s[i]][np.where(pairs[pairs_ind1s[i]]
                                                                            == pairs_ind2s[i])[0][0]]))
        plot_xs(i, sigma=3)

    print("total execution took {0} s".format(time.time() - start))
