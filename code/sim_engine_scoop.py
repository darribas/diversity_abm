'''
Code to ru simulations for "How diverse can spatial measures of cultural diversity be? Results from Monte Carlo simulations of an agent-based model", by Dani
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

Simulation engine for Schelling experiments
...

NOTE: multithreaded version run from the command line as

    > python -m scoop [-n 3] sim_engine_scoop.py

    -n for number of workers to deploy

'''

import os, time, copy, struct, sys
import numpy as np
import pandas as pd
import pysal as ps
import multiprocessing as mp
from scoop import futures
from pysal.inequality import _indices as I
from schelling import World, bounded_world

def god_multi_reps(taus, prop_groupsS, config, multi=True, max_iter=1000):
    '''
    Main controller for a grid simulation where multi-core processing is spanned at
    the different replications performed for every World
    ...

    Arguments
    ---------
    tau                 : float
                          Values of taus to be evaluated
    config              : dict
                          Set of static parameters that determine the world to be
                          created
    prop_groups         : list
                          Proportions of population for each n-1 groups
    multi               : Boolean
                          [Optional. Default=True] Switch to turn on the use
                          of scoop
    max_iter            : int
                          Maximum number of sequential steps to run before
                          giving up on a Schelling run

    Returns
    -------
    simout              : DataFrame
                          Output table
    '''
    out = []
    for prop_groups in prop_groupsS:
        props = prop_groups + [1.-sum(prop_groups)]
        prop_mix = '_'.join(map(str, props))
        any_good_before = True
        for tau in taus:
            if any_good_before:
                ti = time.time()
                if multi:
                    reps = futures.map(run_rep_multi, \
                            [(id, tau, prop_groups, config, max_iter) \
                            for id in np.arange(config['replications'])])
                else:
                    reps = map(run_rep_multi, \
                            [(id, tau, prop_groups, config, max_iter) \
                            for id in np.arange(config['replications'])])
                reps = pd.concat(reps)
                reps['tau'] = tau
                reps['prop_mix'] = prop_mix
                out.append(reps)
                tf = time.time()
                if reps.dropna().shape[0] == 0:
                    any_good_before = False
                print "Tau: %f | Proportions: "%tau, prop_groups, \
                        " finished in %.4f mins."%((tf-ti)/60.)
            else:
                pass
    ##
    out = pd.concat(out)
    out.index.name = 'group'
    out = out.set_index(['tau', 'prop_mix', 'rep_id'], append=True)\
            .swaplevel(0, 1).swaplevel(1, 2).swaplevel(2, 3)
    t1 = time.time()
    return out

def run_rep_multi(rep_id_tau_prop_groups_config_max_iter):
    '''
    Run replication for a combination of parameters-tau and return final
    pattern. Meant for multi-processing as it involves W creation within the
    replication to overpass ROD issue in ps.W
    ...

    Arguments
    ---------
    rep_id_tau_prop_groups_config_max_iter: tuple containing:

            rep_id          : int
                              Replication id to append to output series as name
            tau
            prop_groups
            config
            max_iter

    Returns
    -------
    tab                 : DataFrame
                          Frequency table with rows indexed on neighborhood and
                          columns on group
    '''
    seed = abs(struct.unpack('i',os.urandom(4))[0])
    np.random.seed(seed)
    # Setup the world
    t0 = time.time()
    rep_id, tau, prop_groups, config, max_iter = rep_id_tau_prop_groups_config_max_iter
    w, ns, xys = bounded_world(config['Yi'], config['Xi'], config['Yn'], config['Xn'])
    pop_size = int(round((1 - config['vacant']) * w.n))
    world = World(pop_size, tau, prop_groups, w, neighs=ns, max_iter=max_iter)
    t1 = time.time()
    # Model run
    world.setup()
    world.go()
    tab = world.export()
    # Plumbing out
    t2 = time.time()
    tab = tab.rename(index=lambda i: "n%i"%i, columns=lambda i: "g%i"%i)
    tab['rep_id'] = rep_id
    t3 = time.time()
    if not world.happy_ending:
        for col in tab.columns.drop('rep_id'):
            tab[col] = None
    tab['ticks'] = world.ticks
    return tab

def global_diversity(df, id=None):
    '''
    Calculate all diversity global indices from the output of one replication and
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
            I.abundance, \
            I.margalev_md, \
            I.menhinick_mi, \
            I.simpson_so, \
            I.simpson_sd, \
            I.herfindahl_hd, \
            I.fractionalization_gs, \
            I.shannon_se, \
            I.gini_gi, \
            I.gini_gi_m, \
            I.hoover_hi, \
            ]
    out = {ind.func_name: ind(df.values) for ind in indices}
    return pd.Series(out, name=id)

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
            I.modified_segregation_msg, \
            I.isolation_isg, \
            I.gini_gig, \
            I.ellison_glaeser_egg_pop, \
            I.maurel_sedillot_msg_pop, \
            ]
    out = pd.DataFrame({ind.func_name: pd.Series(ind(df.values), index=df.columns) \
            for ind in indices})
    out.index.name = 'group'
    if id != None:
        out['rep_id'] = id
    return out

if __name__=='__main__':

    #taus = list(np.linspace(0, 0.7, 30))
    taus = list(np.linspace(0, 0.2, 3)) #change2-above
    prop_groupsS = [ 
                # Homogenous
            [0.5],
            [0.2, 0.2, 0.2, 0.2],
               # Uneven-monopoly
            [0.7],
            [0.7, 0.1, 0.1],
               # Polarized
            [0.4, 0.4, 0.1],
               # Escalating
            [0.4, 0.3, 0.2],
               # Mimicking Amsterdam (approx.)
            #[0.2, 0.19, 0.18, 0.15, 0.14, 0.03, 0.03, 0.02]
            ]
    prop_groupsS = [[0.5], [0.5, 0.3]] #change2-above

    print '\n\n### Process start ###'

    config = {\
            # Geo
            ## Pixel rows
            'Yi': 100, \
            ## Pixel columns
            'Xi': 100, \
            ## Neigh. rows
            'Yn': 10, \
            ## Neigh. columns
            'Xn': 10, \
            'vacant': 0.25, \
            # Other
            'replications': 2 #change2-500
            }

    t0 = time.time()
    out = god_multi_reps(taus, prop_groupsS, config, multi=False, max_iter=2000)
    t1 = time.time()
    print 'Total time: %.2f seconds'%(t1-t0)
    out.to_csv('schelling_out_maps.csv')

