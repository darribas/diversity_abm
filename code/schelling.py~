'''
Code to run schelling models, from "How diverse can spatial measures of cultural diversity be? Results from Monte Carlo simulations of an agent-based model", by Dani
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

Schelling[1] segregation model

Inspired by Uri Wilensky's NetLogo code
http://ccl.northwestern.edu/netlogo/models/Segregation

[1] Schelling, T. (1978). Micromotives and Macrobehavior. New York: Norton

'''

import copy
import pandas as pd
import pysal as ps
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

class Agent():
    """
    Hold agent's attributes

    Arguments
    =========
    id      : int
              Agent ID
    group   : int
              ID of group to which agent belongs
    xyid    : int
              ID of the pixel in which the agent is located
    """
    def __init__(self, id, group, xyid):
        # Static
        self.id = id
        self.group = group
        # Dynamic
        self.xyid = xyid
        self.happy = False
        self.similar_nearby = None

class World():
    '''
    Controller of model

    NOTE: currently, happiness rule is as in NetLogo, so and agent is happy if
    total number of similar neighbors >= `pct_similar_wanted` * total
    neighbors
    ...
    
    Arguments
    =========
    pop_size            : int
                          Number of agents to generate
    pct_similar_wanted  : float
                          Proportion in [0, 1] of neighbors that need to be
                          the same for an agent to be happy
    pop_groups          : list
                          List with proportions of the population in each
                          group, except for the last one, which is calculated
                          as a residual
    w                   : pysal.W
                          Spatial weights object for the world (n is all the
                          possible locations in the world (pixels) to land on
    neighs              : ndarray
                          [Optional] List in the same order as w.id_order with
                          the neighborhood to which every observation belongs
                          to
    max_iter            : int
                          Maximum number of sequential steps to run before
                          giving up on a run

    Methods
    =======
    setup               : (run on init by default) Prepare the world to start
                          running by creating the agents and assigning them a
                          random location
    go                  : Run the model until convergence or 10,000
                          iterations, whatever comes first
    plot                : generate a figure with a depiction of the final
                          outcome of the world. Requires:

                            * xys           : ndarray
                                              Nx2 array with locations of all the pixels
                                              in w in the same ordered as passed to w
                            * neighborhoods : ndarray
                                              [Optional] Tuple with number of
                                              row (nr) and column (nc)
                                              neighborhoods
                            * shpfile       : str
                                              [Optional] Path to shapefile
                                              with geometries to be drawn
                            * outfile       : str
                                              [Optional] Path to output file
    '''
    def __init__(self, pop_size, pct_similar_wanted, prop_groups, w, neighs=None, max_iter=1000):
        # Static
        if neighs is not None:
            self.neighs = neighs
        else:
            self.neighs = 'Neighborhood cardinality of xys not passed'
        self.pop_size = pop_size
        self.pct_similar_wanted = pct_similar_wanted
        self.n_groups = len(prop_groups) + 1
        self.prop_groups = prop_groups
        self.w = w
        self.max_iter = max_iter

        self.setup()

    def setup(self):
        # Dynamic
        self.pct_happy = None
        # Geo
        self.free_xyids = self.w.id_order
        agent_xyids = self._random_xy_ids(self.pop_size)
        self.agent_xyids = agent_xyids
        # Agents
        agents = []
        group_map = [np.round(self.pop_size * prop) for prop in self.prop_groups]
        group_map = [[g] * group_map[g] for g in range(len(group_map))]
        group_map = [i for sublist in group_map for i in sublist]
        nlast = self.pop_size - len(group_map)
        ilast = len(self.prop_groups)
        group_map = group_map + [ilast] * nlast
        for id, i in enumerate(range(len(self.agent_xyids))):
            agents.append(Agent(id, group_map[i], self.agent_xyids[i]))
        self.agents = agents
        self.group_map = group_map
        # Setup agent topologies
        _ = self._update_topo()
        _ = map(self._update_agent_nl, self.agents)
        self.happy_ending = True
        self.pct_happy = sum([1 for a in self.agents if a.happy]) * 1. / self.pop_size
        self.ticks = 0
        
    def go(self):
        while not self._all_happy():
            if self.ticks > self.max_iter:
                #print "No happy ending :-("
                self.happy_ending = False
                break
            # Find unhappy
            happy_xyids, unhappy = [], []
            for a in self.agents:
                if not a.happy:
                    unhappy.append(a)
                else:
                    happy_xyids.append(a.xyid)
            self.happy_xyids = happy_xyids
            self.unhappy = unhappy
            self.free_xyids = self._which_free(self.happy_xyids)
            # Assign them different position
            _ = self._move_unhappy()
            self.agent_xyids = [a.xyid for a in self.agents]
            _ = self._update_topo()
            _ = map(self._update_agent_nl, self.agents)

            self.pct_happy = sum([1 for a in self.agents if a.happy]) * 1. / self.pop_size
            self.ticks += 1

    def plot(self, xys, neighborhoods=None, shpfile=None, outfile=None,
            title=None):
        if self.happy_ending:
            cm = get_cmap('RdBu')
            cm = get_cmap('Accent')
            f = plt.figure()
            ax = f.add_subplot(111)
            grid = ax.scatter(xys[:, 0], xys[:, 1], alpha=0.75, linewidths=0, \
                    s=80, marker=None, color='0.6', cmap=cm)
            axys = xys[self.agent_xyids, :]
            ax.scatter(axys[:, 0], axys[:, 1], alpha=0.8, linewidths=0, \
                    s=20, marker='o', c=self.group_map, cmap=cm)
            props = self.prop_groups + [1. - sum(self.prop_groups)]
            if title == None:
                title = "%i agents | %s share | %.2f similar wanted"\
                        %(self.pop_size, '_'.join(map(str, props)), \
                                self.pct_similar_wanted)
            plt.title(title, color='0.3')
            ax.set_frame_on(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            if neighborhoods:
                nr, nc = neighborhoods
                r = len(set(xys[:, 1]))
                ir = int(np.round((r*1.)/nr))
                nrb = [i+ir for i in range(-ir, r, ir)]
                nrb[-1] += 0.5
                nrb[0] -= 0.5
                c = len(set(xys[:, 0]))
                ic = int(np.round((c*1.)/nc))
                ncb = [i+ic for i in range(-ic, c, ic)]
                ncb[0] -= 0.5
                ncb[-1] += 0.5
                # Horizontal lines
                for i in nrb:
                    plt.hlines(i-0.5, xmin=-1, xmax=c, color='k')
                # Vertical lines
                for i in ncb:
                    plt.vlines(i-0.5, ymin=-1, ymax=r, color='k')
            elif shpfile:
                from pysal.contrib.viz import mapping as viz
                shp = ps.open(shpfile)
                patchco = viz.map_poly_shp(shp)
                patchco.set_facecolor('none')
                patchco.set_edgecolor('0.3')
                patchco.set_linewidth(0.3)
                ax = viz.setup_ax([patchco], ax)
            fc = '1'
            f.set_facecolor(fc)
            if not outfile:
                plt.show()
            else:
                plt.savefig(outfile, facecolor=fc)
        else:
            print 'No use in plotting a bad ending'

    def export(self):
        '''
        Encode the world into tabular form
        ...

        Returns
        -------
        tab     : DataFrame
                  Frequency table with rows indexed on neighborhood and
                  columns on group
        '''
        if type(self.neighs) is str:
            print ('Neighborhood cardinality of xys not passed. ' \
                    'Export not completed')
            return None
        agents = pd.DataFrame({\
                'neigh': self.neighs[self.agent_xyids], \
                'group': [a.group for a in self.agents] \
                })
        tab = agents.groupby(['neigh', 'group']).size().unstack().fillna(0)
        return tab

    def _random_xy_ids(self, r):
        np.random.shuffle(self.free_xyids)
        return self.free_xyids[: r]

    def _update_agent(self, agent):
        around = [self.agents[i].group for i in self.atopo[agent.id]]
        total = len(around)
        if total > 0:
            pct_similar_nearby = around.count(agent.group) * 1. / total
            if pct_similar_nearby > self.pct_similar_wanted:
                agent.happy = True
            else:
                agent.happy = False
        else:
            agent.happy = False
            pct_similar_nearby = 0
        agent.pct_similar_nearby = pct_similar_nearby

    def _update_agent_nl(self, agent):
        # Following NetLogo rule
        around = [self.agents[i].group for i in self.atopo[agent.id]]
        total = len(around)
        similar_nearby = around.count(agent.group)
        if similar_nearby >= (self.pct_similar_wanted * total * 1.):
            agent.happy = True
        else:
            agent.happy = False
        agent.similar_nearby = similar_nearby

    def _update_topo(self):
        atopo = ps.w_subset(self.w, self.agent_xyids, \
                silent_island_warning=True).neighbors
        xyid2agent_id = {xyid: agent_id for agent_id, xyid in \
                enumerate(self.agent_xyids)}
        self.atopo = {xyid2agent_id[i]: [xyid2agent_id[j] for j in atopo[i]] for i in atopo}# Values are agent order in self.agents!!!
    
    def _all_happy(self):
        for a in self.agents:
            if a.happy:
                pass
            else:
                return False
        return True

    def _which_free(self, taken_xyids):
        return list(set(self.w.id_order).difference(set(taken_xyids)))

    def _move_unhappy(self):
        new_xyids = self._random_xy_ids(len(self.unhappy))
        for new_xyid, unhappy in zip(new_xyids, self.unhappy):
            unhappy.xyid = new_xyid

def bounded_world(r, c, nr, nc):
    '''
    Create W object for a bounded neighborhood topology based on grids (pixels
    and neighborhoods)

    NOTE: if r/nr or c/nc are not natural, the number of neighborhoods can be
    slightly inacurate
    ...

    Arguments
    ---------
    r       : int
              Number of pixels on the Y axis (rows)
    c       : int
              Number of pixels on the X axis (columns)
    nr      : int
              Number of neighborhoods on the Y axis (rows)
    nc      : int
              Number of neighborhoods on the X axis (columns)

    Returns
    -------
    W       : pysal.W
              Weights object
    ns      : ndarray
              Cardinalities for every observation to a neighborhood
    xys     : ndarray
              Nx2 array with coordinates of pixels
    '''
    x, y = np.indices((r, c))
    ir = int(np.round((r*1.)/nr))
    ic = int(np.round((c*1.)/nc))
    nrb = [i+ir for i in range(-ir, r, ir)]
    ncb = [i+ic for i in range(-ic, c, ic)]
    world = np.zeros((r, c), dtype=int)
    n = 0
    for i in range(len(nrb)-1):
        for j in range(len(ncb)-1):
            world[nrb[i]: nrb[i+1], ncb[j]:ncb[j+1]] = n
            n += 1
    world = world.flatten()
    w = ps.block_weights(world)
    return w, world, np.hstack((x.flatten()[:, None], y.flatten()[:, None]))

def bounded_world_from_shapefile(path, n, n_as=None):
    '''
    Create W object for `n` agents with bounded locations assigned within
    polygons of a shapefile (neighbor if in the same polygon) in proportion to
    the polygon's area
    ...

    Arguments
    ---------
    path    : str
              Link to shapefile containing the geography
    n       : int
              Number of pixels to be created within the geography
    n_as    : array
              [Optional] Sequence with number of of agents to be assigned to
              every neighborhood, in the order of the dbf accompaigning the
              shapefile. If not provided, proportions are based on area.

    Returns
    -------
    W       : pysal.W
              Weights object
    ns      : ndarray
              Cardinalities for every observation to a neighborhood
    xys     : ndarray
              Nx2 array with coordinates of agents (randomly within polygons)
    '''
    if not n_as:
        shp = ps.open(path)
        n_shares = np.array([p.area for p in shp])
        shp.close()
        n_shares = n_shares / n_shares.sum()
        n_as = np.round(n * n_shares).astype(int)
        n_as[-1] = n - n_as[:-1].sum() # Hack to get proportions to sum to n
    xys = np.zeros((n, 2), dtype=int)
    shp = ps.open(path)
    polys = list(shp)
    shp.close()
    parss = [(n_a, poly) for n_a, poly in zip(n_as, polys)]
    pool = mp.Pool(mp.cpu_count())
    xys = pool.map(_random_pts_in_poly, parss)
    xys = np.concatenate(xys)
    ns = np.concatenate([np.array([neigh]*nn) for neigh, nn in enumerate(n_as)])
    w = ps.regime_weights(ns)
    return w, ns, xys

def _random_pts_in_poly(pars):
    '''
    Generate `n` random points inside a given `polygon`
    '''
    n_pts, poly = pars
    bbox = poly.bounding_box
    xys = np.zeros((n_pts, 2))
    for i in np.arange(xys.shape[0]):
        in_poly = 0
        while not in_poly:
            x = np.random.uniform(low=bbox.left, high=bbox.right, size=1)[0]
            y = np.random.uniform(low=bbox.lower, high=bbox.upper, size=1)[0]
            in_poly = poly.contains_point((x, y))
        xys[i] = (x, y)
    return xys

if __name__ == '__main__':
    import time
    import pandas as pd
    world_dims = 70, 70
    '''
    # Agent-centered neighborhood
    x, y = np.indices(world_dims)
    xys = np.hstack((x.flatten()[:, None], y.flatten()[:, None]))
    w = ps.lat2W(world_dims[0], world_dims[1], rook=False)
    '''
    # Bounded neighborhood
    nr, nc = neighs = (10, 10)
    w, ns, xys = bounded_world(world_dims[0], world_dims[1], nr, nc)
    '''
    # Adam shape
    shp = '../../data/adam/adam_projected.shp'
    n_pix = world_dims[0] * world_dims[1]
    ti = time.time()
    w, ns, xys = bounded_world_from_shapefile(link, n_pix)
    tf = time.time()
    fo = ps.open(link.replace('.shp', '_random_pts.gal'), 'w')
    fo.write(w)
    fo.close()
    import pandas as pd
    db = pd.DataFrame({'x': xys[:, 0], 'y': xys[:, 1], 'ns': ns})
    db.to_csv(link.replace('.shp', '_xys_ns.csv'), index=False)
    print "Time to create the geography: %.2f seconds"%(tf-ti)
    w = ps.open(shp.replace('.shp', '_random_pts.gal')).read()
    db = pd.read_csv(shp.replace('.shp', '_xys_ns.csv'))
    ns = db['ns'].values
    xys = db[['x', 'y']].values
    '''

    pop_size = int((world_dims[0] * world_dims[1]) * 0.75)
    pct_similar_wanted = 0.19
    prop_groups = [0.2]
    t0 = time.time()
    world = World(pop_size, pct_similar_wanted, prop_groups, w, ns)
    world.go()
    t1 = time.time()
    print "Total time: %.2f minutes"%((t1-t0) / 60.)
    world.plot(xys, neighborhoods=neighs)
    #world.plot(xys, shpfile='../../data/rev1/grid.shp')

