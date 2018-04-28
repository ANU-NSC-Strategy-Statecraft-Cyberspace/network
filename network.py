import networkx as nx
from random import random, choice
import numpy as np
from matplotlib.collections import LineCollection

lobby_color = np.array([0, 1.0, 0])
party_color = np.array([0, 1.0, 0])
default_color = np.array([0, 0, 0])
high_charisma_color = np.array([0, 0, 1.0])
high_firmness_color = np.array([1.0, 0, 0])
high_charisma_and_firmness_color = np.array([1.0, 0, 1.0])

within_region_edge_color = np.array([0.0, 0.0, 0.0, 1.0])
between_region_edge_color = np.array([1.0, 0.0, 0.0, 0.3])

def get_lobby_color(o):
    return lobby_color

def get_color(xs):
    """ Returns the colour for a list of Opinions xs occupying the same point """
    has_politician = False
    has_high_charisma = False
    has_high_firmness = False

    for x in xs:
        if x.is_politician:
            has_politician = True
        if x.has_high_charisma:
            has_high_charisma = True
        if x.has_high_firmness:
            has_high_firmness = True

    if has_politician:
        return party_color
    if has_high_charisma and has_high_firmness:
        return high_charisma_and_firmness_color
    if has_high_charisma:
        return high_charisma_color
    if has_high_firmness:
        return high_firmness_color
    return default_color

class Opinion:
    def __init__(self, dimensions, charisma, high_charisma, firmness, high_firmness, region, is_politician, is_lobby, lobby_dimension, lobby_charisma_hack):
        if is_lobby:
            if lobby_charisma_hack is not None:
                assert charisma == lobby_charisma_hack
                self.pos = np.array([0 for d in range(dimensions)])
            else:
                self.pos = np.array([(random()*2 - 1 if d == lobby_dimension else 0) for d in range(dimensions)])
        else:
            assert lobby_charisma_hack is None
            self.pos = np.array([random()*2 - 1 for _ in range(dimensions)])
        self.charisma = charisma
        self.high_charisma = high_charisma
        self.firmness = firmness
        self.high_firmness = high_firmness
        self.region = region
        assert sum([is_politician, is_lobby]) <= 1
        self.is_politician = is_politician
        self.is_lobby = is_lobby
        assert lobby_dimension < dimensions
        assert (lobby_dimension == -1 and not is_lobby) or (lobby_dimension >= 0 and is_lobby)
        self.lobby_dimension = lobby_dimension

    def add_noise(self, noise):
        result = self.pos + noise*np.array([random()*2 - 1 for _ in range(len(self.pos))])
        self.pos = np.clip(result, -1, 1)

    def get_lobby_pos(self, x):
        assert not x.is_lobby
        if self.is_lobby:
            result = np.array(x.pos)
            result[self.lobby_dimension] = self.pos[self.lobby_dimension]
            return result
        else:
            return self.pos

    def get_lobby_line(self):
        assert self.is_lobby
        if self.lobby_dimension == 0:
            return (self.pos[0], self.pos[0]), (-2, 2)
        elif self.lobby_dimension == 1:
            return (-2, 2), (self.pos[1], self.pos[1])
        else:
            assert False

# y_t = sqrt(y_0^2 - 2*k*t*m1*m2 + z*y_0^2)
def inverse_force(args):
    k=args.attract_strength
    z=args.repel_strength
    def update_func(x, ys, firmness, charismas):
        assert firmness >= 0 and firmness <= 1.0
        assert len(charismas) == len(ys)
        assert all(c >= 0 and c <= 1.0 for c in charismas)
        if len(ys) > 0:
            both = list(zip(ys, charismas))
            np.random.shuffle(both)
            for y, c in both:
                dist = np.linalg.norm(x-y)
                var = dist*dist*(1+z) - 2*k*c*firmness
                if var < 0:
                    x = y
                else:
                    x = y + (x - y) * np.sqrt(var) / dist
            return np.clip(x, -1, 1)
        else:
            return x
    return update_func
	
class Agent:
	def __init__(self, context, region):
		self.args = context.args
		self.region = region
		self.opinion = Opinion(self.args.dimensions, charisma, high_charisma, firmness, high_firmness, is_politician, is_lobby, lobby_dimension, lobby_charisma_hack)
		
		self.firmness = choice([self.args.low_firmness, self.args.high_firmness])
		self.has_high_firmness = self.firmness != self.args.low_firmness
		
		self.charisma = choice([self.args.low_charisma, self.args.high_charisma])
		self.has_high_charisma = self.charisma != self.args.low_charisma

class Party:
	def __init__(self, context, region):
		self.args = context.args
		self.region = region
		
		self.firmness = 1.0
		self.charisma = 1.0

class Lobby:
	def __init__(self, context, region):
		self.args = context.args
		self.region = region
		
		self.firmness = 1.0

		if self.args.lobby_charisma_hack is not None:
            self.lobby_dimension = 0
			self.charisma = self.args.lobby_charisma_hack
        else:
            self.lobby_dimension = choice(range(self.args.dimensions))
			self.charisma = 1.0

class Context:
    def __init__(self, args):
		self.args = args
        self.agents = {Agent(self, x % args.regions) for x in range(args.population)}
		self.parties = {Party(self, r) for r in range(args.regions) for _ in range(2)} if args.use_parties else {}
		self.lobbies = {Lobby(self, r) for r in range(args.regions) for _ in range(args.num_lobbies)}
        self.G = nx.Graph()
		self.G.add_nodes_from(self.agents)
		self.G.add_nodes_from(self.parties)
		self.G.add_nodes_from(self.lobbies)
        
        self.current_party_interval = 1
		
        self.update_func = inverse_force(args)
        self.region_distances = np.ones((args.regions,args.regions))-np.eye(args.regions)

    """ Under the dunbar number, agents make connections in strict order of close opinions
        Over the dunbar number, agents break connections with the least shared opinion
        At the dunbar number, agents randomly probe for other neighbours with closer opinions
    """

    def above_dunbar(self, x):
        self.G.remove_edge(x,max(self.G[x], key=lambda y: self.opinion_distance(x,y)))

    def below_dunbar(self, x):
        candidates = [y for y in self.G if y != x and y not in self.G[x]]
        if candidates:
            winner = self.G.add_edge(x,min(candidates, key=lambda y: self.opinion_distance(x,y)))

    def at_dunbar(self, x):
        if self.G[x]:
            y = choice(list(self.G.nodes))
            z = choice(list(self.G[x].keys()))
            if y != x and self.opinion_distance(x,y) < self.opinion_distance(x,z):
                self.G.add_edge(x,y)
                self.G.remove_edge(x,z)

    def update(self, axes, draw):
        self.update_parties()
        for x in self.agents:
			if len(self.G[x]) > args.dunbar:
				self.above_dunbar(x)
			elif len(self.G[x]) < args.dunbar:
				self.below_dunbar(x)
			else:
				self.at_dunbar(x)

                self.update_opinion(x,self.G[x])
        return self.draw_step(axes, draw)

    def draw_step(self, axes, draw):
        if draw:
            for i, ax in enumerate(axes):
                ax.clear()
                ax.set_xlim([-1.1,1.1])
                ax.set_ylim([-1.1,1.1])
                nodelist, edgelist, pos, colors, edge_colors, sizes = self.get_node_draw_data(self.G, i)
                self.draw_lobbies(self.G, i, ax)
                nx.draw_networkx(self.G, pos=pos, nodelist=nodelist, edgelist=edgelist, ax=ax, node_color=colors, edge_color=edge_colors, node_size=sizes, with_labels=False, alpha=None)
                self.draw_parties(i, ax)
                ax.set_xticks([])
                ax.set_yticks([])
        return axes
		
	def connected_components(self):
	    ccs = list(nx.connected_components(self.G))
        result = [sum(1 for cc in ccs if any(self.opinions[node].region == i for node in cc)) for i in range(self.args.regions)]
		return result
		
	def histogram_x(self):
		result = [0 for _ in range(100)]
		for o in self.opinions.values():
			if not o.is_lobby:
				fl = np.floor((o.pos[0]+1)*50.0)
				if fl >= 100:
					fl = 99
				result[int(fl)] += 1
		return result
		
	    def positions(self, G):
        return {x : o.pos for x,o in self.opinions.items()}

    def opinion_distance(self, x, y):
		x = self.opinions[x]
		y = self.opinions[y]
        if x.is_lobby and y.is_lobby:
            assert False
        elif x.is_lobby:
            return np.abs(x.pos[x.lobby_dimension] - y.pos[x.lobby_dimension]) / self.quality(x,y)
        elif y.is_lobby:
            return np.abs(x.pos[y.lobby_dimension] - y.pos[y.lobby_dimension]) / self.quality(x,y)
        else:
            return np.linalg.norm(x.pos - y.pos) / self.quality(x,y)

    def update_opinion(self, x, ys):
		x = self.opinions[x]
		ys = [self.opinions[y] for y in ys]
		x.pos = self.update_func(x.pos, [y.get_lobby_pos(x) for y in ys], self.firmness(x), [self.charisma(x,y) for y in ys])
        x.add_noise(self.args.noise)

    def get_politicians(self, axis):
        return [o for o in self.opinions.values() if o.is_politician and o.region == axis]

    def get_voters(self, axis):
        return [o for o in self.opinions.values() if o.region == axis and not o.is_politician]

    def draw_parties(self, axis, ax):
        if not self.args.use_parties:
            return
        a,b = self.get_politicians(axis)
        linex, liney = self.get_midline(a, b)
        voters = self.get_voters(axis)
        ax.plot(linex, liney, color='g')
        avotes = sum(1 for x in voters if np.linalg.norm(x.pos - a.pos) < np.linalg.norm(x.pos - b.pos))
        bvotes = sum(1 for x in voters if np.linalg.norm(x.pos - a.pos) > np.linalg.norm(x.pos - b.pos))
        ax.text(a.pos[0], a.pos[1], str(avotes), color='r', horizontalalignment='center', verticalalignment='center')
        ax.text(b.pos[0], b.pos[1], str(bvotes), color='b', horizontalalignment='center', verticalalignment='center')
        ax.text(-0.1, 1, str(avotes), color='r', horizontalalignment='right', verticalalignment='bottom')
        ax.text(0.1, 1, str(bvotes), color='b', horizontalalignment='left', verticalalignment='bottom')

    def draw_lobbies(self, G, axis, ax):
        for o in self.opinions.values():
            if o.is_lobby and o.region == axis:
                linex, liney = o.get_lobby_line()
                ax.plot(linex, liney, color=get_lobby_color(o), alpha=0.5, linewidth=1)

        edge_pos = []
        edge_colors = []
        for e in G.edges():
            o1 = self.opinions[e[0]]
            o2 = self.opinions[e[1]]
            if o1.region != axis and o2.region != axis:
                continue
            elif o1.is_lobby:
                assert not o2.is_lobby
            elif o2.is_lobby:
                assert not o1.is_lobby
                o1, o2 = o2, o1
            else:
                continue
            edge_pos.append((o1.get_lobby_pos(o2),o2.pos))
            edge_colors.append(within_region_edge_color if o1.region == o2.region else between_region_edge_color)

        edge_collection = LineCollection(edge_pos, colors=edge_colors, antialiaseds=(1,), transOffset=ax.transData)
        edge_collection.set_zorder(1)
        ax.add_collection(edge_collection)

    def update_parties(self):
        if not self.args.use_parties:
            return
        if self.current_party_interval < self.args.party_interval:
            self.current_party_interval += 1
            return
        else:
            self.current_party_interval = 1

        for axis in range(self.args.regions):
            a,b = self.get_politicians(axis)
            voters = self.get_voters(axis)
            a_allies = [x.pos for x in voters if np.linalg.norm(x.pos - a.pos) < np.linalg.norm(x.pos - b.pos)]
            b_allies = [x.pos for x in voters if np.linalg.norm(x.pos - a.pos) > np.linalg.norm(x.pos - b.pos)]

            new_a = sum(a_allies) / len(a_allies) if len(a_allies) > 0 else a.pos
            projections = [np.dot(p.pos - b.pos, new_a - b.pos) / np.linalg.norm(new_a - b.pos)**2 for p in voters]
            new_a = np.median(projections) * 2 * (new_a - b.pos) + b.pos if np.median(projections) <= 0.5 else new_a
            new_a = np.clip(a.pos + np.clip(new_a - a.pos, -self.args.party_inertia, self.args.party_inertia), -1, 1)

            new_b = sum(b_allies) / len(b_allies) if len(b_allies) > 0 else b.pos
            projections = [np.dot(p.pos - a.pos, new_b - a.pos) / np.linalg.norm(new_b - a.pos)**2 for p in voters]
            new_b = np.median(projections) * 2 * (new_b - a.pos) + a.pos if np.median(projections) <= 0.5 else new_b
            new_b = np.clip(b.pos + np.clip(new_b - b.pos, -self.args.party_inertia, self.args.party_inertia), -1, 1)

            a.pos = new_a
            b.pos = new_b


    def get_duplicate_lists(self, dd):
        rev = {}
        for k, v in dd.items():
            rev.setdefault(tuple(v), []).append(k)
        return list(rev.values())

    def get_size(self, xs):
        return 50 * (np.log(len(xs)) + 1)

    def get_node_draw_data(self, G, axis):
        positions = self.positions(G)
        positions_filtered = {x: p for x,p in positions.items() if self.opinions[x].region == axis and not self.opinions[x].is_lobby}
        unique_positions = self.get_duplicate_lists(positions_filtered)
        colors = {xs[0] : get_color([self.opinions[x] for x in xs]) for xs in unique_positions}
        sizes = {xs[0] : self.get_size(xs) for xs in unique_positions}
        nodes = [xs[0] for xs in unique_positions if self.opinions[xs[0]].region == axis]
        edges = [e for e in G.edges() if (self.opinions[e[0]].region == axis or self.opinions[e[1]].region == axis) and not (self.opinions[e[0]].is_lobby or self.opinions[e[1]].is_lobby)]
        edge_colors = [within_region if (self.opinions[e[0]].region == self.opinions[e[1]].region) else between_region for e in edges]
        # nodelist, edgelist, pos, colors, edge_colors, sizes
        return nodes, edges, positions, [colors[x] for x in nodes], edge_colors, [sizes[x] for x in nodes]

    def get_midline(self, x, y):
        a = x.pos
		b = y.pos
		assert any(a != b)
		midpoint = (a + b)/2
		vx, vy = b - midpoint
		vx, vy = vy, -vx
		if vx == 0:
			return (midpoint[0], midpoint[0]), (-2, 2)
		else:
			slope = vy / vx
			f = (lambda x: midpoint[1] + (x - midpoint[0]) * slope)
			return (-2, 2), (f(-2), f(2))

    def firmness(self, x):
        return x.firmness

    def charisma(self, x, y):
        return y.charisma * self.quality(x, y)

    def quality(self, x, y):
        return 1 - self.region_distances[x.region, y.region] * self.args.quality_loss
