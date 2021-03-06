'''
Code to process simulation results from "How diverse can spatial measures of cultural diversity be? Results from Monte Carlo simulations of an agent-based model", by Dani
Arribas-Bel, Peter Nijkamp and Jacques Poot
Author: Dani Arribas-Bel <daniel.arribas.bel@gmail.com>
...

Copyright (c) 2015, Daniel Arribas-Bel

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
  
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
  
* The name of Daniel Arribas-Bel may not be used to endorse or promote products
  derived from this software without specific prior written permission.
  
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

---

Process results from ABM simulations
'''

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import moment
from pysal.inequality import _indices as I

walk = {'0.5_0.5': 'Benchmark', \
        '0.7_0.3': 'Single minority', \
        '0.7_0.1_0.1_0.1': 'Multiple minority', \
        '0.4_0.4_0.1_0.1': 'Duopoly', \
        '0.4_0.3_0.2_0.1': 'Ladder', \
        '0.2_0.2_0.2_0.2_0.2': 'Diversity'}

def process_map(map_table_link):
    '''
    Map processor that calculates diversity indices. Processor happens
    sequentially as the file is read in chunks (this makes it substantially
    slower but scalable).
    ...

    Arguments
    ---------
    map_table_link  : str
                      Path to table from simulations indexed on tau, prop_mix,
                      and group (neighborhood) and with counts for each
                      population group. It also includes `ticks` but this is
                      dropped at this stage.

    Returns
    -------
    i_table         : DataFrame
                      Table mimicking old version of output from simulations.
                      This is indexed on tau, prop_mix, rep_id and group and
                      contains a column for each index calculated.
    '''
    t0 = time.time()
    reader = pd.read_csv(map_table_link, index_col=[0, 1, 3], chunksize=49)
    print map_table_link
    out = []
    for mapa in reader:
        rep_id = mapa['rep_id'].iloc[0]
        inds = spatial_diversity(mapa.drop(['rep_id', 'ticks'], axis=1)\
                .dropna(axis=1), rep_id)
        inds['ticks'] = mapa['ticks'].iloc[0]
        inds['tau'] = mapa.index.get_level_values('tau')[0]
        prop_mix = mapa.index.get_level_values('prop_mix')[0]
        inds['prop_mix'] = prop_mix
        group_map = {'g%i'%g: 'g%i-%s'%(g, str(p).ljust(4, '0')) for g, p in \
                enumerate(prop_mix.split('_'))}
        inds['group'] = inds.index.values
        inds['group'] = inds['group'].map(group_map)
        inds = inds.set_index(['tau', 'prop_mix', 'rep_id', 'group'])
        out.append(inds)
    out = pd.concat(out)
    t1 = time.time()
    print '\t %.2f seconds'%(t1 - t0)
    return out

def process_job(job):
    '''
    Take results from a single job (from multi-vacancy/urban setup
    simulations) and calculate diversity indices.

    Arguments
    ---------
    job             : DataFrame
                      Table from simulations indexed on tau, prop_mix,
                      rep_id and group (neighborhood) and with counts for each
                      population group. It also includes `ticks` but this is
                      dropped at this stage.

    Returns
    -------
    i_table         : DataFrame
                      Table mimicking old version of output from simulations.
                      This is indexed on tau, prop_mix, rep_id and group and
                      contains a column for each index calculated.
    '''
    t0 = time.time()
    out = []
    mapas = job.groupby(['job', 'tau', 'rep_id'])
    for ids, mapa in mapas:
        job, tau, rep_id = ids
        mapa = mapa.set_index(['tau', 'prop_mix', 'group'])
        todrop = ['rep_id', 'ticks', 'vacr', 'city', 'job']
        x = mapa.drop(todrop, axis=1).dropna(axis=1)
        inds = spatial_diversity(x, rep_id)
        inds['ticks'] = mapa['ticks'].iloc[0]
        inds['tau'] = tau
        prop_mix = mapa.index.get_level_values('prop_mix')[0]
        inds['prop_mix'] = prop_mix
        inds['vacr'] = mapa['vacr'].iloc[0]
        inds['city'] = mapa['city'].iloc[0]
        inds['job'] = mapa['job'].iloc[0]
        group_map = {'g%i'%g: 'g%i-%s'%(g, str(p).ljust(4, '0')) for g, p in \
                enumerate(prop_mix.split('_'))}
        inds['group'] = inds.index.values
        inds['group'] = inds['group'].map(group_map)
        inds = inds.set_index(['tau', 'prop_mix', 'rep_id', 'group'])
        out.append(inds)
    out = pd.concat(out)
    t1 = time.time()
    print '\t %.2f seconds'%(t1 - t0)
    return out

def spatial_diversity(df, id=None):
    '''
    Calculate all diversity spatial indices from the output of one replication and
    return output
    ...

    Arguments
    ---------
    df      : pd.DataFrame
              World output data from one replication
    id      : str
              ID to attach to output series as its name

    Returns
    -------
    out     : pd.Series
              Indices computed
    '''
    indices = [ \
            I.segregation_gsg, \
            #I.modified_segregation_msg, \
            I.theil_th, \
            I.isolation_ii, \
            #I.gini_gig, \
            I.ellison_glaeser_egg_pop, \
            #I.maurel_sedillot_msg_pop, \
            ]
    out = pd.DataFrame({ind.func_name: pd.Series(ind(df.values), index=df.columns) \
            for ind in indices})
    out.index.name = 'group'
    if id != None:
        out['rep_id'] = id
    return out

def sim_descriptives(res, folder=None):
    '''
    Obtain descriptives from simulations as per request by Jacques:

        * The number of observations used to calculated each mean index
          value
        * The standard deviation of the distribution
        * The skewness of the distribution
    '''
    stats = list(res.columns.values)
    funs = ['mean', 'std', 'skew']
    for stat in stats:
        print 'Drawing ', stat
        for fun in funs:
            print '\t', fun
            _ = build_tauplot((stat, res), fun=fun, \
                    title=fun + ' | ', \
                    saveto=folder+stat+'_'+fun+'.png')
    return None

def build_convfreqs(res, saveto=None, savefig=None):
    '''
    Build plot with simulation descriptives
    ...

    Arguments
    ---------
    res
    saveto
    savefig

    Returns
    -------
    out
    ts
    '''
    g = res.groupby(level=['prop_mix'])
    f, axes = plt.subplots(g.ngroups, 1, figsize=(6, 10))
    color = 'k'
    backcolor = '0.5'
    '''
    if type(axes) != list:
        axes = [axes]
    '''
    out = []
    ts = []
    for i, block in enumerate(zip(axes, g)):
        ax, pack = block
        id, s = pack
        g1 = s.index.get_level_values('group').unique()[0]
        plotblock = s.dropna().groupby(level=['tau', 'group'])\
                .size().unstack()[g1]
        plotblock = pd.DataFrame({id: plotblock})
        plotblock.plot(kind='line', ax=ax, color=color, grid=False, \
                alpha=1.0)
        ax.legend().set_visible(False)
        ax.set_ylim((0, 550))
        ax.set_xlabel('Tau', size=10, color=backcolor)
        ax.set_ylabel('Draws', size=10, color=backcolor)
        ax.tick_params(axis='both', which='major', labelsize=10,
                labelcolor=backcolor, color=backcolor)
        plt.setp(ax.spines.values(), color=backcolor)
        out.append(plotblock)
        ticks = s.dropna().groupby(level=['tau', 'group'])\
                .mean()['ticks'].unstack()[g1]
        ticks.name = id
        axT = ax.twinx()
        ticks.plot(kind='line', style='--', ax=axT, color=color, grid=False, \
                alpha=1.0)
        axT.set_xlim((0, 0.7))
        axT.set_ylabel('Iterations', size=10, color=backcolor)
        axT.tick_params(axis='both', which='major', labelsize=10,
                labelcolor=backcolor, color=backcolor)
        ts.append(ticks)
        title = id + ' | ' + walk[id]
        plt.title(title)
    d = plt.Line2D((0, 0), (1, 1), linestyle='-', c=color)
    t = plt.Line2D((0, 0), (1, 1), linestyle='--', c=color)
    plt.legend([d, t], ["Draws", "Iterations"], loc=0, frameon=False)
    f.tight_layout()
    if savefig:
        plt.savefig(savefig)
    else:
        plt.show()
    out = pd.concat(out, axis=1).fillna(0)
    ts = pd.concat(ts, axis=1)
    if saveto:
        fo = open(saveto, 'w')
        fo.write('\n\\vspace{2cm}\n')
        t0 = out.ix[:, :3].to_latex()
        fo.write(t0)
        fo.write('\n\\vspace{2cm}\n')
        t1 = out.ix[:, 3:].to_latex()
        fo.write(t1)
        fo.close()
    return out, ts

def build_tauplot_by_scenario(res, folder=None, scenarios='all', saveto=None, diffs=False, fun='mean', title=''):
    '''
    Plot of evolution of `stat` on tau for each scenario. Simply passes
    parameters to the plotting engine in _plot_scenario
    '''
    if scenarios == 'all':
        scenarios = list(set(res.index.get_level_values('prop_mix')))
    for sce in scenarios:
        print 'Plotting scenario ', sce
        sceblock = res.query('prop_mix == "%s"'%sce).drop('ticks', axis=1)
        if folder:
            outfile = folder + sce.replace('.', '') + '.png'
        else:
            outfile = None
        _ = _plot_scenario(sceblock, fun=fun, outfile=outfile)
    if not folder:
        plt.show()
    return sceblock

def _plot_scenario(sceblock, fun, outfile=None):
    symbols = ['--', '+', '-.', ':', '.', ',', '|', 'x', '2', '3', '^', '<',
            'v', 'o', '4', '>', '1', 's', 'p', '*', 'h', 'H', 'D', 'd', '_']
    pretty_nameof = {'segregation_gsg': 'Segregation index $SI_{gt}$', \
            'modified_segregation_msg': 'Modified segregation index $MS_{gt}$', \
            'isolation_isg': 'Isolation index $II_{gt}$', \
            'isolation_ii': 'Isolation index $II_{gt}$', \
            'theil_th': 'Theil index $TD_{t}$', \
            'gini_gig': 'Gini coefficient $GC_{gt}$', \
            'ellison_glaeser_egg_pop': \
            'Ellison & Glaeser Concentration Index $EG_{gt}$', \
            'maurel_sedillot_msg_pop': \
            'Maurel & Sedillot Concentration Index $MS_{gt}$'}
    backcolor = '0.5'
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #saveto = '../../output/2013_08_plots3_full_sims/figs/%s.png'%stat
    figsize = (6, 10)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    f, axes = plt.subplots(sceblock.shape[1], 1, figsize=figsize)
    cols_ord = ['segregation_gsg', 'modified_segregation_msg', 'isolation_isg', \
            'gini_gig', 'ellison_glaeser_egg_pop', 'maurel_sedillot_msg_pop', \
            'isolation_ii', 'theil_th']
    cols = [c for c in cols_ord if c in sceblock.columns]
    for ax, col in zip(axes, cols):
        if fun == 'mean':
            g = sceblock[col].groupby(level=['tau', 'group']).mean().unstack()
        elif fun == 'std':
            g = sceblock[col].groupby(level=['tau', 'group']).std().unstack()
        else:
            raise Exception, "`fun` needs to be 'mean' or 'std'"
        maxX = g
        g.plot(ax=ax, c='k', grid=False,
                style=symbols[:g.shape[1]])
        ax.set_xlabel('')
        ax.legend(fontsize=7, loc=0, frameon=False)
        ax.tick_params(axis='both', which='major', labelsize=7,
                labelcolor=backcolor, color=backcolor)
        plt.setp(ax.spines.values(), color=backcolor)
        ax.set_title(pretty_nameof[col])
    ax.set_xlabel('$\\tau$')
    f.tight_layout()
    if outfile:
        plt.savefig(outfile)
    return g

def build_tauplot(statres, saveto=None, diffs=False, fun='mean', title=''):
    '''
    Plot of evolution of `stat` on tau
    '''
    stat, res = statres
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #saveto = '../../output/2013_08_plots3_full_sims/figs/%s.png'%stat
    diffs = False
    figsize = (6, 10)
    #figsize = (3, 5)
    #figsize = (5, 2.5)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    symbols = ['--', '+', '-.', ':', '.', ',', '|', 'x', '2', '3', '^', '<',
            'v', 'o', '4', '>', '1', 's', 'p', '*', 'h', 'H', 'D', 'd', '_']
    backcolor = '0.5'
    #print "Statistic: ", stat
    g = res.groupby(level=['prop_mix'])[stat]

    minY = np.inf
    maxY = -np.inf
    maxX = res.index.get_level_values('tau').values.max()
    f, axes = plt.subplots(g.ngroups, 1, figsize=figsize)
    '''
    if type(axes) != list:
        axes = [axes]
    '''
    f.suptitle(title + g.name)
    ids = g.groups.keys()
    ids = pd.Series(map(lambda x: x.count('_'), ids), \
            index=ids).order().index.values.tolist()
    for i, ax in enumerate(axes):
        id = ids[i]
        s = g.get_group(id)
        if fun == 'mean':
            plotblock = s.groupby(level=['tau', 'group']).mean().unstack()
        if fun == 'std':
            plotblock = s.groupby(level=['tau', 'group']).std().unstack()
        if fun == 'skew':
            plotblock = s.groupby(level=['tau', 'group'])\
                    .apply(lambda x: moment(x, 3)).unstack()
        if plotblock.min().min() < minY:
            minY = plotblock.min().min()
        if plotblock.max().max() > maxY:
            maxY = plotblock.max().max()
        plotblock.plot(ax=ax, c='k', grid=False,
                style=symbols[:plotblock.shape[1]])
        ax.tick_params(axis='both', which='major', labelsize=7,
                labelcolor=backcolor, color=backcolor)
        ax.set_xlim((0, maxX))
        plt.setp(ax.spines.values(), color=backcolor)
        ax.legend(fontsize=5, loc=1, frameon=False)

        if diffs and (s.index.get_level_values('group').unique().shape[0] == 2):
            fd, dax = plt.subplots(1, figsize=figsize)
            ds = s.unstack()
            ds = (ds['g0-0.50'] - ds['g1-0.50'])**2 / \
                    ((ds['g0-0.50'] + ds['g0-0.50'])*0.5)
            ds.groupby(level=['tau']).mean()\
                    .plot(ax=dax, color='k', grid=False)
            dax.tick_params(axis='both', which='major', labelsize=7,
                    labelcolor=backcolor, color=backcolor)
            #dax.set_title("Sq. differences between each group's index")
    for ax in axes:
        ax.set_ylim((minY, maxY))
    if saveto:
        plt.savefig(saveto)
    else:
        plt.show()
    return s
    #return saveto

def build_denplot(statressaveto):
    '''
    Plot density distribution of `stat` for tau
    '''
    symbols = ['--', '+', '-.', ':', '.', ',', '|', 'x', '2', '3', '^', '<',
            'v', 'o', '4', '>', '1', 's', 'p', '*', 'h', 'H', 'D', 'd', '_']
    stat, res, saveto = statressaveto
    saveto = '../../output/2013_08_plots3_full_sims/figs/dens_%s.png'%stat
    print "Building plot for ", stat
    res = res.dropna()
    g = res.groupby(level=['prop_mix'])[stat]

    f, axes = plt.subplots(g.ngroups, \
            res.index.get_level_values('tau').unique().shape[0], \
            figsize=(60, 30))
    if type(axes) != list:
        axes = axes[:, None]
    mixes = g.groups.keys()
    mixes.sort()
    for mix in range(len(mixes)):
        s = g.get_group(mixes[mix])
        for pack, ax in zip(s.groupby(level='tau'), axes[mix]):
            try:
                id, case = pack
                case.unstack().dropna().plot(kind='kde', ax=ax, grid=False, \
                        style=symbols, color='k')
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_xlabel('$\\tau$ = %s'%str(round(id, ndigits=2)))
                ax.set_ylabel('')
                ax.legend(fontsize=9)
            except:
                print "\tFailed on %s | %s"%(stat, id)
    plt.suptitle(stat)
    if saveto:
        plt.savefig(saveto)
    else:
        plt.show()
    return saveto

def build_meanStd_plots(sces, stdX=False, saveto=None):
    '''
    Build full figure with Mean+Std plots for each scenario
    ...

    Arguments
    ---------
    sces    : DataFrame
              Table double indexed on "prop_mix" and "tau" with a "mean" and a
              "std" column.
    stdX    : Boolean/tuple
              [Optional. Default=False] Switch to standardize the length of X
              axis. If not False, it should contain the desired range.
    saveto  : None/str
              [Optional. Default=None] If passed, path to write plot to a
              file. Returns interactive plot otherwise.
    '''
    n_sces = len(set(sces.index.get_level_values('prop_mix')))
    f, axs = plt.subplots(n_sces, 1, figsize=(6, 10))
    if stdX:
        stdX = (0, sces['mean'].max())
    for i, (id, sce) in enumerate(sces.groupby(level='prop_mix')):
        sce = sce.xs(id, level='prop_mix')
        ax = meanStd_plot(sce, axs[i], stdX=stdX)
        ax.set_title(id)
    plt.tight_layout()
    if saveto:
        return plt.savefig(saveto)
    else:
        return plt.show()

def meanStd_plot(sce, ax, stdX=False):
    '''
    Build the Mean+Std plot of a single scenario
    ...

    Arguments
    ---------
    sce     : DataFrame
              Table single indexed on "tau" with a "mean" and a "std" column
    ax      : Axes
              Place holder axis
    stdX    : Boolean/tuple
              [Optional. Default=False] Switch to standardize the length of X
              axis. If not False, it should contain the desired range.

    Returns
    -------
    ax      : Axes
              Axis with plot
    '''
    _ = sce['mean'].plot(ax=ax, c='k')
    ax.fill_between(sce['mean'].index, sce['mean'] + 3. * sce['std'], \
                                       sce['mean'] - 3. * sce['std'], \
                    color='0.75')
    ax.set_ylabel('Theil index $TD_{t}$')
    ax.set_ylim((0, 1))
    if stdX:
        ax.set_xlim(stdX)
    return ax

def main_effect_plot(y, x, b, db, ax=None):
    '''
    Build Main Effect plots for the relationship between `x` and `y`, based on
    a model with data `db` and estimated parameters `b`

    **NOTE**: it assumes that powers of `x` are named in `db` as `xP`, where
    `P` is an int with the power (e.g. x2, x3, etc).
    ...

    Arguments
    ---------
    y       : str
              Name of dependent variable, contained in `db`
    x       : str
              Name of independent variable, contained in `db`.
    b       : Series
              Set of parameters from the fitted model
    db      : DataFrame
              Table with data
    '''
    rng = np.linspace(db[x].min(), db[x].max(), 100)
    allbutx = [i for i in b.index if x not in i]
    level = db[allbutx].mean().dot(b[allbutx])
    xs = [i for i in b.index if x in i]
    powers = [i.strip(x) for i in xs]
    powers.pop(powers.index(''))
    powers = map(int, powers)
    line = level + rng * b[x] + \
            np.sum([rng**i * b[x+str(i)] for i in powers], axis=0)
    ax.scatter(db[x], db[y], marker='.', c='k', s=0.5)
    ax.plot(rng, line, c='k', lw=1.5)
    ax.set_xlabel(x)
    #ax.set_ylabel(y)
    return ax

if __name__ == '__main__':

    res = pd.read_csv('../../data/rev1/sim_res.csv', index_col=[0, 1, 2, 3])

    stats = ['maurel_sedillot_msg_pop', 'ellison_glaeser_egg_pop']
    stats = list(res.columns.values)
    stats = ['gini_gig']

            # Tau plots
    #tauplots = map(build_tauplot, [(stat, res) for stat in stats])
    scep = build_tauplot_by_scenario(res, '/Users/arribasd/Desktop/figs/')
    '''

            # Densities
    #denplots = map(build_denplot, [(stat, res, 'filled_later') for stat in stats])

            # Convergence
    saveto = '../../output/revision_epa1/figs/table.tex'
    fig = '../../output/revision_epa1/figs/sim_descriptives.png'
    #fs = build_convfreqs(res, savefig=fig)

            # Descriptives for Jacques
    #des = sim_descriptives(res, '../../data/sim_descriptives/')
    '''

