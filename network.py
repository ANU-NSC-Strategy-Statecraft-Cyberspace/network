import networkx as nx
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from random import random, choice
import numpy as np
from matplotlib.collections import LineCollection
import csv
from enum import Enum

lobby_color = np.array([0, 1.0, 0])
party_color = np.array([0, 1.0, 0])
default_color = np.array([0, 0, 0])
high_charisma_color = np.array([0, 0, 1.0])
high_firmness_color = np.array([1.0, 0, 0])
high_charisma_and_firmness_color = np.array([1.0, 0, 1.0])

def get_lobby_color(o):
    return lobby_color

def get_color(xs):
    has_politician = False
    has_high_charisma = False
    has_high_firmness = False

    for x in xs:
        if x.is_politician:
            has_politician = True
        if x.high_charisma:
            has_high_charisma = True
        if x.high_firmness:
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

class Arguments:
    def __init__(self, **kwargs):
        self.dunbar = 5
        self.population=100
        self.snapshot=set()
        self.snapshot_name=''
        self.steps=500
        self.noise=0
        self.show=True
        self.save=False
        self.vid_name=""
        self.attract_strength = 0.001
        self.repel_strength = 0
        self.use_parties = False
        self.num_politicians = 1
        self.num_lobbies = 2
        self.party_interval = 1
        self.party_inertia = 0.01
        self.quality_loss = 0.5
        self.bubbles=4
        self.dimensions=2
        self.low_firmness = 0.1
        self.high_firmness = 0.9
        self.low_charisma = 1.0
        self.high_charisma = 1.0
        self.lobby_charisma_hack = None

        for key, value in kwargs.items():
            setattr(self, key, value)

def get_firmness(args, is_lobby, is_politician):
    if is_lobby or is_politician:
        return 1.0, False

    firmness = choice([args.low_firmness, args.high_firmness])
    return firmness, firmness != args.low_firmness

def get_charisma(args, is_lobby, is_politician, lobby_charisma_hack):
    if is_lobby or is_politician:
        if lobby_charisma_hack is not None:
            assert is_lobby
            return lobby_charisma_hack, False
        else:
            return 1.0, False
    assert lobby_charisma_hack is None
    charisma = choice([args.low_charisma, args.high_charisma])
    return charisma, charisma != args.low_charisma

class Opinion:
    def __init__(self, dimensions, charisma, high_charisma, firmness, high_firmness, bubble, is_politician, is_lobby, lobby_dimension, lobby_charisma_hack):
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
        self.bubble = bubble
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

class OpinionClass:
    def __init__(self, args):
        self.update_func = inverse_force(args)
        self.args = args
        self.bubble_distances = np.ones((args.bubbles,args.bubbles))-np.eye(args.bubbles)
        assert not args.use_parties or args.num_politicians == 2

    def opinion_distance(self,x,y):
        if x.is_lobby and y.is_lobby:
            assert False
        elif x.is_lobby:
            return np.abs(x.pos[x.lobby_dimension] - y.pos[x.lobby_dimension]) / self.quality(x,y)
        elif y.is_lobby:
            return np.abs(x.pos[y.lobby_dimension] - y.pos[y.lobby_dimension]) / self.quality(x,y)
        else:
            return np.linalg.norm(x.pos - y.pos) / self.quality(x,y)

    def new_opinion(self, index):
        bubble = index % self.args.bubbles
        assert self.args.num_politicians + self.args.num_lobbies <= self.args.population // self.args.bubbles
        is_politician = index < self.args.bubbles * self.args.num_politicians
        is_lobby = index < self.args.bubbles * (self.args.num_politicians + self.args.num_lobbies) and not is_politician

        lobby_charisma_hack = None
        if is_lobby:
            lobby_charisma_hack = self.args.lobby_charisma_hack

        firmness, high_firmness = get_firmness(self.args, is_lobby, is_politician)
        charisma, high_charisma = get_charisma(self.args, is_lobby, is_politician, lobby_charisma_hack)

        if is_lobby:
            if self.args.lobby_charisma_hack is not None:
                lobby_dimension = 0
            else:
                lobby_dimension = choice(range(self.args.dimensions))
        else:
            lobby_dimension = -1

        return Opinion(self.args.dimensions, charisma, high_charisma, firmness, high_firmness, bubble, is_politician, is_lobby, lobby_dimension, lobby_charisma_hack)

    def get_midline(self, x, y):
        return get_midline(x.pos, y.pos)

    def firmness(self, x):
        return x.firmness

    def charisma(self, x, y):
        return y.charisma * self.quality(x, y)

    def update_opinion(self,x,ys):
        x.pos = self.update_func(x.pos, [y.get_lobby_pos(x) for y in ys], self.firmness(x), [self.charisma(x,y) for y in ys])

    def quality(self, x, y):
        return 1 - self.bubble_distances[x.bubble, y.bubble] * self.args.quality_loss

def get_midline(a, b):
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

class AgentClass:
    def __init__(self, args):
        self.args = args
        self.OpinionClass = OpinionClass(args)
        self.opinions = {x : self.OpinionClass.new_opinion(x) for x in range(args.population)}
        assert args.noise >= 0 and args.noise <= 1
        self.current_party_interval = 1

    def positions(self, G):
        return {x : o.pos for x,o in self.opinions.items()}

    def get_xlim(self):
        return [-1.1,1.1]

    def get_ylim(self):
        return [-1.1,1.1]

    def opinion_distance(self, x, y):
        return self.OpinionClass.opinion_distance(self.opinions[x], self.opinions[y])

    def update_opinion(self, x, y):
        self.OpinionClass.update_opinion(self.opinions[x], [self.opinions[y] for y in ys])
        self.opinions[x].add_noise(self.args.noise)

    def get_politicians(self, axis):
        return [o for o in self.opinions.values() if o.is_politician and o.bubble == axis]

    def get_voters(self, axis):
        return [o for o in self.opinions.values() if o.bubble == axis and not o.is_politician]

    def draw_parties(self, axis, ax):
        if not self.args.use_parties:
            return
        assert self.args.dimensions == 2
        assert self.args.num_politicians == 2
        a,b = self.get_politicians(axis)
        linex, liney = self.OpinionClass.get_midline(a, b)
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
            if o.is_lobby and o.bubble == axis:
                linex, liney = o.get_lobby_line()
                ax.plot(linex, liney, color=get_lobby_color(o), alpha=0.5, linewidth=1)

        edge_pos = []
        edge_colors = []
        for e in G.edges():
            o1 = self.opinions[e[0]]
            o2 = self.opinions[e[1]]
            if o1.bubble != axis and o2.bubble != axis:
                continue
            elif o1.is_lobby:
                assert not o2.is_lobby
            elif o2.is_lobby:
                assert not o1.is_lobby
                o1, o2 = o2, o1
            else:
                continue
            edge_pos.append((o1.get_lobby_pos(o2),o2.pos))
            edge_colors.append(within_region if o1.bubble == o2.bubble else between_region)

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
        assert self.args.dimensions == 2
        assert self.args.num_politicians == 2

        for axis in range(self.args.bubbles):
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
        positions_filtered = {x: p for x,p in positions.items() if self.opinions[x].bubble == axis and not self.opinions[x].is_lobby}
        unique_positions = self.get_duplicate_lists(positions_filtered)
        colors = {xs[0] : get_color([self.opinions[x] for x in xs]) for xs in unique_positions}
        sizes = {xs[0] : self.get_size(xs) for xs in unique_positions}
        nodes = [xs[0] for xs in unique_positions if self.opinions[xs[0]].bubble == axis]
        edges = [e for e in G.edges() if (self.opinions[e[0]].bubble == axis or self.opinions[e[1]].bubble == axis) and not (self.opinions[e[0]].is_lobby or self.opinions[e[1]].is_lobby)]
        edge_colors = [within_region if (self.opinions[e[0]].bubble == self.opinions[e[1]].bubble) else between_region for e in edges]
        # nodelist, edgelist, pos, colors, edge_colors, sizes
        return nodes, edges, positions, [colors[x] for x in nodes], edge_colors, [sizes[x] for x in nodes]

within_region = (0.0, 0.0, 0.0, 1.0)
between_region = (1.0, 0.0, 0.0, 0.3)

class UpdateClass:
    def __init__(self, Agents, G, args):
        self.Agents = Agents
        self.G = G
        self.args = args

    """ Under the dunbar number, agents make connections in strict order of close opinions
        Over the dunbar number, agents break connections with the least shared opinion
        At the dunbar number, agents randomly probe for other neighbours with closer opinions
    """

    def above_dunbar(self, x):
        self.G.remove_edge(x,max(self.G[x], key=lambda y: self.Agents.opinion_distance(x,y)))

    def below_dunbar(self, x):
        candidates = [y for y in self.G if y != x and y not in self.G[x]]
        if candidates:
            winner = self.G.add_edge(x,min(candidates, key=lambda y: self.Agents.opinion_distance(x,y)))

    def at_dunbar(self, x):
        if self.G[x]:
            y = choice(list(self.G.nodes))
            z = choice(list(self.G[x].keys()))
            if y != x and self.Agents.opinion_distance(x,y) < self.Agents.opinion_distance(x,z):
                self.G.add_edge(x,y)
                self.G.remove_edge(x,z)

    def update(self, axes, draw):
        self.Agents.update_parties()
        for x in self.G:
            if self.Agents.opinions[x].is_lobby:
                pass
            elif self.Agents.opinions[x].is_politician and self.args.use_parties:
                pass
            else:
                if len(self.G[x]) > args.dunbar:
                    self.above_dunbar(x)
                elif len(self.G[x]) < args.dunbar:
                    self.below_dunbar(x)
                else:
                    self.at_dunbar(x)

                self.Agents.update_opinion(x,self.G[x])
        return self.draw_step(axes, draw)

    def draw_step(self, axes, draw):
        if draw:
            for i, ax in enumerate(axes):
                ax.clear()
                ax.set_xlim(self.Agents.get_xlim())
                ax.set_ylim(self.Agents.get_ylim())
                nodelist, edgelist, pos, colors, edge_colors, sizes = self.Agents.get_node_draw_data(self.G, i)
                self.Agents.draw_lobbies(self.G, i, ax)
                nx.draw_networkx(self.G, pos=pos, nodelist=nodelist, edgelist=edgelist, ax=ax, node_color=colors, edge_color=edge_colors, node_size=sizes, with_labels=False, alpha=None)
                self.Agents.draw_parties(i, ax)
                ax.set_xticks([])
                ax.set_yticks([])
        return axes

def update(num, Updater, axes, fig, args, draw):
    axes = Updater.update(axes, draw)
    if draw:
        fig.suptitle('Step: {}'.format(num))
    if num in args.snapshot:
        assert draw
        assert len(args.snapshot_name) > 0
        fig.savefig('{}-figure-{}.png'.format(args.snapshot_name, num))
    return axes

def animate(args = Arguments()):
    G = nx.empty_graph(args.population)
    Agents = AgentClass(args)
    Updater = UpdateClass(Agents, G, args)
    gridsize = int(np.ceil(np.sqrt(args.bubbles)))
    fig, axes_array = plt.subplots(gridsize, gridsize, squeeze=False)
    axes = axes_array.flatten()
    for ax in axes:
        ax.set_aspect('equal')
    graph_ani = animation.FuncAnimation(fig, update, args.steps, fargs=(Updater, axes, fig, args, True), interval=100, blit=False, repeat=False)
    if args.show:
        plt.show()
    if args.save:
        graph_ani.save('{}.mp4'.format(args.vid_name), fps=10)
    plt.close()


def run_simulation(args, components=None, histogram_x=None):
    G = nx.empty_graph(args.population)
    Agents = AgentClass(args)
    Updater = UpdateClass(Agents, G, args)
    for n in range(args.steps):
        update(n, Updater, [], None, args, False)
        if components is not None:
            ccs = list(nx.connected_components(G))
            result = [sum(1 for cc in ccs if any(Agents.opinions[node].bubble == i for node in cc)) for i in range(args.bubbles)]
            components.append(result)
        if histogram_x is not None:
            result = [0 for _ in range(100)]
            for o in Agents.opinions.values():
                if not o.is_lobby:
                    fl = np.floor((o.pos[0]+1)*50.0)
                    if fl >= 100:
                        fl = 99
                    result[int(fl)] += 1
            histogram_x.append(result)


figure_steps = 300
figure_repeats = 20
figure_inputs = 20

def figure_one():
    #graph of connected components across time (do people form filter bubbles?)
    runs = []
    for r in range(figure_repeats):
        components = []
        run_simulation(Arguments(steps=figure_steps, show=False, num_politicians=0, num_lobbies=0, quality_loss=0.5, bubbles=1), components=components)
        runs.append(components)
        print("{}/{}".format(1 + r, figure_repeats))
    runs = np.array(runs)
    runs = np.mean(runs, axis=(0,2))
    print(runs.shape)
    np.save('figure_1.npy', runs)

def figure_two():
    #graph of connected components per axis across communication penalty (use more regions)
    result = []
    inputs = np.linspace(0.0, 0.9999, num=figure_inputs)
    for i, quality_loss in enumerate(inputs):
        runs = []
        for r in range(figure_repeats):
            components = []
            run_simulation(Arguments(steps=figure_steps, show=False, num_politicians=0, num_lobbies=0, quality_loss=quality_loss, bubbles=10), components=components)
            runs.append(components[-1])
            print("{}/{}".format(1 + r + figure_repeats*i, figure_repeats*figure_inputs))
        runs = np.array(runs)
        runs = np.mean(runs, axis=(0,1))
        result.append(runs)
    result = np.array(result)
    print(result.shape)
    np.save('figure_2.npy', result)
    np.save('figure_2_inputs.npy', inputs)


def figure_three():
    #heat map of opinions on one axis against lobby charisma
    result = []
    inputs = np.linspace(0.0, 1.0, num=figure_inputs)
    for i, lobby_charisma in enumerate(inputs):
        runs = []
        for r in range(figure_repeats):
            # # TODO TODO fix lobby position
            histogram = []
            run_simulation(Arguments(steps=figure_steps, show=False, num_politicians=0, num_lobbies=1, quality_loss=0.5, bubbles=1, lobby_charisma_hack=lobby_charisma), histogram_x=histogram)
            runs.append(histogram[-1])
            print("{}/{}".format(1 + r + figure_repeats*i, figure_repeats*figure_inputs))
        runs = np.array(runs)
        runs = np.mean(runs, axis=0)
        result.append(runs)
    result = np.array(result)
    print(result.shape)
    np.save('figure_3.npy', result)
    np.save('figure_3_inputs.npy', inputs)

def make_figures():
    figure_1 = np.load('figure_1.npy')
    figure_2 = np.load('figure_2.npy')
    figure_2_inputs = np.load('figure_2_inputs.npy')
    figure_3 = np.load('figure_3.npy')
    figure_3_inputs = np.load('figure_3_inputs.npy')

    plt.close()
    plt.plot(figure_1)
    plt.xlabel("Time")
    plt.ylabel("Connected components")
    plt.savefig('A-figure_1.png')

    plt.close()
    plt.plot(figure_2_inputs, figure_2)
    plt.xlabel("Cost of cross-region communication")
    plt.ylabel("Connected components per region at t=300")
    plt.gca().invert_xaxis()
    plt.savefig('A-figure_2.png')

    plt.close()
    plt.imshow(np.sqrt(figure_3))
    plt.xlabel("X-axis opinion positions of agents")
    plt.xticks([])
    plt.yticks([0, len(figure_3_inputs) - 1], [figure_3_inputs[0], figure_3_inputs[-1]])
    plt.ylabel("Charisma of lobby")
    #plt.colorbar()
    plt.savefig('A-figure_3.png')


## new diagrams:
def run_diagrams():
    animate(Arguments(firmness_type=Firmness.Inverse, num_politicians=0, use_parties=False, num_lobbies=2, bubbles=1, quality_loss=0.5,  snapshot={10,200}, snapshot_name='A-lobbies', steps=201))
    animate(Arguments(firmness_type=Firmness.Inverse, num_politicians=0, use_parties=False, num_lobbies=0, bubbles=4, quality_loss=0.5,  snapshot={10,200}, snapshot_name='A-connect', steps=201))
    animate(Arguments(firmness_type=Firmness.Inverse, num_politicians=0, use_parties=False, num_lobbies=0, bubbles=4, quality_loss=0.99, snapshot={10,200}, snapshot_name='A-apart',   steps=201))
    animate(Arguments(firmness_type=Firmness.Inverse, num_politicians=2, use_parties=True,  num_lobbies=0, bubbles=1, quality_loss=0.5,  snapshot={10,200}, snapshot_name='A-parties', steps=201))

def video_diagrams():
    animate(Arguments(firmness_type=Firmness.Inverse, num_politicians=0, use_parties=False, num_lobbies=2, bubbles=1, quality_loss=0.5,  steps=201, show=False, save=True, vid_name="A-lobbies-vid"))
    #animate(Arguments(firmness_type=Firmness.Inverse, num_politicians=0, use_parties=False, num_lobbies=0, bubbles=4, quality_loss=0.5,  steps=201, show=False, save=True, vid_name="A-connect-vid"))
    #animate(Arguments(firmness_type=Firmness.Inverse, num_politicians=0, use_parties=False, num_lobbies=0, bubbles=4, quality_loss=0.99, steps=201, show=False, save=True, vid_name="A-apart-vid"))
    #animate(Arguments(firmness_type=Firmness.Inverse, num_politicians=2, use_parties=True,  num_lobbies=0, bubbles=1, quality_loss=0.5,  steps=201, show=False, save=True, vid_name="A-parties-vid"))