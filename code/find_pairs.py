import numpy as np
from astropy.table import Table, unique
from astropy import units as u
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
#from schwimmbad import SerialPool, MultiPool
from plot_tools import error_ellipse

min_columns = ['source_id', 'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'parallax_error',
                'pmra_error', 'pmdec_error', 'parallax_pmra_corr', 'parallax_pmdec_corr',
                'pmra_pmdec_corr']

def construct_table(gaia_table_file, kepler_table_file, minimal=False):
    hdul = fits.open(gaia_table_file)
    gaia_src_tbl = Table(hdul[1].data)
    if minimal:
        gaia_src_tbl = gaia_src_tbl[min_columns]
        return gaia_src_tbl.to_pandas()
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
    
def calc_chisq(m, primary):
    if ppm_check(primary, m):
        return chisq(primary, m)
    else:
        return -1
        
def calc_chisqs_for_table(table, pairs, save=False, save_every=1e6, save_name='chisqs.fits'):
    chisqs = np.zeros_like(pairs) - 1.
    for i,row in tqdm(enumerate(pairs)):
        primary = table.iloc[i]
        row_mask = (row > -1) & (row > i) # indices in row for matches to compute
        matches = table.iloc[row[row_mask]] # ignore non-matches and duplicates
        if np.sum(row_mask) > 0:
            row_of_chisqs = matches.apply(calc_chisq, args=(primary,), axis=1)
            chisqs[i,row_mask] = row_of_chisqs.values
        if save and (i % save_every == 1):
            save_as_fits(save_name, chisqs)
    
    if save:
        save_as_fits(save_name, chisqs)        
    return chisqs
    
def worker(data):
    """
    Wrapper function for parallelization
    """
    table, pairs = data
    chisqs = calc_chisqs_for_table(table, pairs, save=False)
    return chisqs

    
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
    gaia_table_file = '../data/gaia-kepler-dustin.fits'
    kepler_table_file = '../data/kepler_dr2_1arcsec.fits'
    table = construct_table(gaia_table_file, kepler_table_file, minimal=True)
    
    print("loading pair indices...")
    pairs_file = '../data/matched-pairs-dustin.fits'
    pairs = read_from_fits(pairs_file)
    
    print("calculating chisqared...")    
    chisqs = calc_chisqs_for_table(table, pairs, save=True, save_name='../data/chisqs_matched-pairs.fits')
    
        
    '''
    pool = SerialPool()
    n_workers = 1e4
    n_rows = len(pairs)
    piece_size = n_rows % (n_workers-1)
    all_inds = np.arange(n_rows)
    piece_start_inds = [i*piece_size for i in range(n_workers)]
    piece_end_inds = [min((i+1)*piece_size,n_rows) for i in range(n_workers)]
    pieces = [[table[piece_start_inds[i]:piece_end_inds[i]],
                  pairs[piece_start_inds[i]:piece_end_inds[i]])] for i in range(n_workers)]
    '''
        
    
    print("chisqareds calculated, checking on matches...")    
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
    
    table = construct_table(gaia_table_file, kepler_table_file, minimal=False)
    
