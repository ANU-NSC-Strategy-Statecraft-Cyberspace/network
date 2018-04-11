import networkx as nx
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from random import random, choice
import numpy as np
import colorsys
from copy import copy, deepcopy
from functools import partial
from matplotlib.collections import LineCollection
import csv
import typing
from typing import Dict, List, Set, Tuple, Callable, Any
from enum import Enum



# Ideas from discussion:
#
#  - A repulsive force that is *equal* in x and y directions even when the separation differs
#  - randomise order of moves, make the opinion boundary circular, make noise circular
#  - Discrete opinions with properties, Meme objects; Other idea-attractiveness functions e.g. priviledged ideas
#  - reasons to vote or disengage, willingness to make compromises
#  - reject Y because associated with X
#  - look at dunbar's parameter - dynamics at village scale (population = dunbar), city scale (population > dunbar), cyber scale (population >> dunbar) and think about interactions with spatial limits
#  - Remove dunbar limit for sources
#  - ageing process to decrease convinceability and/or conservativeness
#  - mechanic of capturing connections by being nearby on one dimensions, then pulling them on another one
#  - sources with looser views on one dimensions that allows them to pull people in easier
#  - ratio of strength between desire to connect and desire to confirm opinions


# Accomplishments:
# 1. Inertia for parties
# 2. Poisson distribution for Dunbar numbers
# 3. Graphs of connected components

Position = Any
Color = Any
Update_Func = Callable[[Position, List[Position], float, List[float]], Position]
Layout_Map = Dict[int, Position]
Network = Any
Axis = Any
Edge = Any
Figure = Any

class DunbarType(Enum):
    Fixed = 'fixed'
    Poisson = 'poisson'

class Selectivity(Enum):
    Nothing = 'none'
    Recommend = 'recommend'
    Selective = 'selective'

class Layout(Enum):
    Spring = 'spring'
    Opinion = 'opinion'

class Update(Enum):
    Average = 'average'
    Inverse = 'inverse'

class Firmness(Enum):
    Fixed = 'fixed'
    Random = 'random'
    Inverse = 'inverse'

class Charisma(Enum):
    Fixed = 'fixed'
    Random = 'random'

class Color_Func(Enum):
    Opinion = 'opinion'
    Firmness = 'firmness'
    Charisma = 'charisma'

class Arguments(typing.NamedTuple):
    dunbar:int = 5
    dunbar_type:DunbarType = DunbarType.Fixed
    population:int=100
    snapshot:Set[int]=set()
    snapshot_name:str=''
    steps:int=500
    selectivity:Selectivity = Selectivity.Selective
    rec_factor:float=0
    noise:float=0
    show:bool=True
    save:bool=False
    vid_name:str=""
    layout:Layout = Layout.Opinion
    num_sources:int = 0
    update_func:Update = Update.Inverse
    attract_strength:float = 0.001
    repel_strength:float = 0
    repel_extreme_factor:float = 0
    use_parties:bool = False
    num_politicians:int = 1
    num_lobbies:int = 2
    party_interval:int = 1
    party_inertia:float = 0.01
    quality_loss:float = 0.5
    bubbles:int=4
    dimensions:int=2
    firmness_type:Firmness = Firmness.Fixed
    firmness:float = 0.1
    charisma_type:Charisma = Charisma.Fixed
    charisma:float = 1.0
    color_func:Color_Func = Color_Func.Firmness
    lobby_charisma_hack:float = None

def get_firmness(args:Arguments, is_source:bool, is_lobby:bool, is_politician:bool) -> float:
    if is_source or is_lobby or is_politician:
        return 1.0

    if args.firmness_type == Firmness.Fixed:
        return args.firmness
    elif args.firmness_type == Firmness.Random:
        return random()
    elif args.firmness_type == Firmness.Inverse:
        return choice([args.firmness, 1.0 - args.firmness])
    else:
        assert False

def get_charisma(args:Arguments, is_source:bool, is_lobby:bool, is_politician:bool, lobby_charisma_hack:float) -> float:
    if is_source or is_lobby or is_politician:
        if lobby_charisma_hack is not None:
            assert is_lobby
            return lobby_charisma_hack
        else:
            return 1.0
    assert lobby_charisma_hack is None
    if args.charisma_type == Charisma.Fixed:
        return args.charisma
    elif args.charisma_type == Charisma.Random:
        return random()
    else:
        assert False

class Opinion:
    def __init__(self, dimensions:int, charisma:float, firmness:float, bubble:int, is_source:bool, is_politician:bool, is_lobby:bool, lobby_dimension:int, lobby_charisma_hack:float) -> None:
        if is_lobby:
            if lobby_charisma_hack is not None:
                assert charisma == lobby_charisma_hack
                self.pos:Position = np.array([0 for d in range(dimensions)])
            else:
                self.pos:Position = np.array([(random()*2 - 1 if d == lobby_dimension else 0) for d in range(dimensions)])
        else:
            assert lobby_charisma_hack is None
            self.pos:Position = np.array([random()*2 - 1 for _ in range(dimensions)])
        self.charisma:float = charisma
        self.firmness:float = firmness
        self.bubble:int = bubble
        assert sum([is_source, is_politician, is_lobby]) <= 1
        self.is_source:bool = is_source
        self.is_politician:bool = is_politician
        self.is_lobby:bool = is_lobby
        assert lobby_dimension < dimensions
        assert (lobby_dimension == -1 and not is_lobby) or (lobby_dimension >= 0 and is_lobby)
        self.lobby_dimension:int = lobby_dimension

    def add_noise(self, noise:float) -> None:
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

    def get_lobby_line(self) -> Tuple[Position, Position]:
        assert self.is_lobby
        if self.lobby_dimension == 0:
            return (self.pos[0], self.pos[0]), (-2, 2)
        elif self.lobby_dimension == 1:
            return (-2, 2), (self.pos[1], self.pos[1])
        else:
            assert False

Opinion_Map = Dict[int, Opinion]

def move_towards_average(args:Arguments) -> Update_Func:
    def update_func(x:Position, ys:List[Position], firmness:float, charismas:List[float]) -> Position:
        assert firmness >= 0 and firmness <= 1.0
        assert len(charismas) == len(ys)
        assert all(c >= 0 and c <= 1.0 for c in charismas)
        if len(ys) > 0:
            return sum(y * c * firmness + x*(1 - firmness) for y, c in zip(ys, charismas)) / sum(c * firmness + 1 - firmness for c in charismas)
        else:
            return np.clip(x, -1, 1)
    return update_func

# y_t = sqrt(y_0^2 - 2*k*t*m1*m2 + z*y_0^2)
def inverse_force(args:Arguments) -> Update_Func:
    k=args.attract_strength
    z=args.repel_strength
    z_factor=args.repel_extreme_factor
    def update_func(x:Position, ys:List[Position], firmness:float, charismas:List[float]) -> Position:
        assert firmness >= 0 and firmness <= 1.0
        assert len(charismas) == len(ys)
        assert all(c >= 0 and c <= 1.0 for c in charismas)
        extreme = np.linalg.norm(x)
        zz = z + extreme * z_factor
        if len(ys) > 0:
            both = list(zip(ys, charismas))
            np.random.shuffle(both)
            for y, c in both:
                dist = np.linalg.norm(x-y)
                var = dist*dist*(1+zz) - 2*k*c*firmness
                if var < 0:
                    x = y
                else:
                    x = y + (x - y) * np.sqrt(var) / dist
            return np.clip(x, -1, 1)
        else:
            return x
    return update_func

class OpinionClass:
    def __init__(self, args:Arguments) -> None:
        self.update_func:Update_Func = update_funcs[args.update_func](args)
        self.args:Arguments = args
        self.get_color:Callable[[List[Opinion]], Color] = color_funcs[args.color_func]
        self.nodesorter:Callable[[List[int], Dict[int, Opinion]], float] = sorter_funcs[args.color_func]
        self.bubble_distances = np.ones((args.bubbles,args.bubbles))-np.eye(args.bubbles)
        assert not args.use_parties or args.num_politicians == 2

    def opinion_distance(self,x:Opinion,y:Opinion) -> float:
        if x.is_lobby and y.is_lobby:
            assert False
        elif x.is_lobby:
            return np.abs(x.pos[x.lobby_dimension] - y.pos[x.lobby_dimension]) / self.quality(x,y)
        elif y.is_lobby:
            return np.abs(x.pos[y.lobby_dimension] - y.pos[y.lobby_dimension]) / self.quality(x,y)
        else:
            return np.linalg.norm(x.pos - y.pos) / self.quality(x,y)

    def new_opinion(self, index:int) -> Opinion:
        bubble = index % self.args.bubbles
        assert self.args.num_politicians + self.args.num_sources + self.args.num_lobbies <= self.args.population // self.args.bubbles
        is_politician = index < self.args.bubbles * self.args.num_politicians
        is_source = index < self.args.bubbles * (self.args.num_sources + self.args.num_politicians) and not is_politician
        is_lobby = index < self.args.bubbles * (self.args.num_sources + self.args.num_politicians + self.args.num_lobbies) and not is_politician and not is_source

        lobby_charisma_hack = None
        if is_lobby:
            lobby_charisma_hack = self.args.lobby_charisma_hack

        firmness = get_firmness(self.args, is_source, is_lobby, is_politician)
        charisma = get_charisma(self.args, is_source, is_lobby, is_politician, lobby_charisma_hack)

        if is_lobby:
            if self.args.lobby_charisma_hack is not None:
                lobby_dimension = 0
            else:
                lobby_dimension = choice(range(self.args.dimensions))
        else:
            lobby_dimension = -1

        return Opinion(self.args.dimensions, charisma, firmness, bubble, is_source, is_politician, is_lobby, lobby_dimension, lobby_charisma_hack)

    def get_midline(self, x:Opinion, y:Opinion) -> Tuple[Position, Position]:
        return get_midline(x.pos, y.pos)

    def firmness(self, x:Opinion) -> float:
        return x.firmness

    def charisma(self, x:Opinion, y:Opinion) -> float:
        return y.charisma * self.quality(x, y)

    def update_opinion(self,x:Opinion,ys:List[Opinion]) -> None:
        x.pos = self.update_func(x.pos, [y.get_lobby_pos(x) for y in ys], self.firmness(x), [self.charisma(x,y) for y in ys])

    def quality(self, x:Opinion, y:Opinion) -> float:
        return 1 - self.bubble_distances[x.bubble, y.bubble] * self.args.quality_loss

def get_single_color(x:Position) -> Color:
    return np.array([(x[0] + 1) / 2, 1.0 - 0.25*(x[0]+1 + x[1]+1), (x[1]+1)/2])

def opinion_color(xs:List[Opinion]) -> Color:
    for x in xs:
        if x.is_source or x.is_lobby:
            return np.array([0, 1.0, 0])
        if x.is_politician:
            return np.array([0, 0, 1.0])

    color = get_single_color(xs[0].pos)
    assert all(all(get_single_color(x.pos) == color) for x in xs)
    return color

def opinion_sorter(xs:List[int], os:Dict[int, Opinion]) -> float:
    for x in xs:
        if os[x].is_politician or os[x].is_source:
            return 2.0
    return 1.0

def firmness_color(xs:List[Opinion]) -> Color:
    for x in xs:
        if x.is_source or x.is_lobby:
            return np.array([0, 1.0, 0])
        if x.is_politician:
            return np.array([0, 0, 1.0])

    return np.array([max(x.firmness for x in xs), 0, 0])

def firmness_sorter(xs:List[int], os:Dict[int, Opinion]) -> float:
    for x in xs:
        if os[x].is_politician or os[x].is_source:
            return 2.0
    return max(os[x].firmness for x in xs)

def charisma_color(xs:List[Opinion]) -> Color:
    for x in xs:
        if x.is_source or x.is_lobby:
            return np.array([0, 1.0, 0])
        if x.is_politician:
            return np.array([0, 0, 1.0])

    return np.array([max(x.charisma for x in xs), 0, 0])

def charisma_sorter(xs:List[int], os:Dict[int, Opinion]) -> float:
    for x in xs:
        if os[x].is_politician or os[x].is_source:
            return 2.0
    return max(os[x].charisma for x in xs)

def get_midline(a:Position, b:Position) -> Tuple[Position, Position]:
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

"""
class ProportionalOpinion(OpinionClass):
    def strength(self, x):
        strength = np.linalg.norm(x - np.array([0,0])) / np.linalg.norm(np.array([1,1]))
        assert strength >= 0.0
        assert strength <= 1.0
        return strength

    def update_opinion(self,x,ys):
        return (x[0], self.update_func(x[1], [y[1] for y in ys], 1-self.strength(x[1]), [1.0 for _ in ys]))

    def get_single_color(self, x):
        return np.array([self.strength(x), 0, 0])
"""

class LayoutClass:
    def __init__(self) -> None:
        assert False
        self.xlim:List[float] = []
        self.ylim:List[float] = []

    def positions(self, layoutpos:Layout_Map, G:Network, opinions:Opinion_Map, opinion_class:OpinionClass) -> Layout_Map:
        assert False
        return {}

class SpringLayout(LayoutClass):
    def __init__(self) -> None:
        self.xlim:List[float] = [-0.25,1.25]
        self.ylim:List[float] = [-0.25,1.25]

    def positions(self, layoutpos:Layout_Map, G:Network, opinions:Opinion_Map, opinion_class:OpinionClass) -> Layout_Map:
        return nx.spring_layout(G, pos=layoutpos, iterations=10, k=0.5/np.sqrt(len(layoutpos)))

class OpinionLayout(LayoutClass):
    def __init__(self) -> None:
        self.xlim:List[float] = [-1.1,1.1]
        self.ylim:List[float] = [-1.1,1.1]

    def positions(self, layoutpos:Layout_Map, G:Network, opinions:Opinion_Map, opinion_class:OpinionClass) -> Layout_Map:
        return {x : o.pos for x,o in opinions.items()}

class AgentClass:
    def __init__(self, args:Arguments) -> None:
        assert args.num_sources <= args.population
        self.args:Arguments = args
        self.OpinionClass:OpinionClass = OpinionClass(args)
        self.opinions:Opinion_Map = {x : self.OpinionClass.new_opinion(x) for x in range(args.population)}
        self.layoutpos:Layout_Map = None
        self.LayoutClass:LayoutClass = graph_layout[args.layout]
        assert args.noise >= 0 and args.noise <= 1
        assert args.num_sources == 0 or not args.use_parties
        self.current_party_interval:int = 1

    def positions(self, G:Network) -> Layout_Map:
        self.layoutpos = self.LayoutClass.positions(self.layoutpos, G, self.opinions, self.OpinionClass)
        return self.layoutpos

    def get_xlim(self) -> List[float]:
        return self.LayoutClass.xlim

    def get_ylim(self) -> List[float]:
        return self.LayoutClass.ylim

    def opinion_distance(self, x:int, y:int) -> float:
        return self.OpinionClass.opinion_distance(self.opinions[x], self.opinions[y])

    def update_opinion(self, x:int, ys:List[int]) -> None:
        self.OpinionClass.update_opinion(self.opinions[x], [self.opinions[y] for y in ys])
        self.opinions[x].add_noise(self.args.noise)

    def get_politicians(self, axis:int) -> List[Opinion]:
        return [o for o in self.opinions.values() if o.is_politician and o.bubble == axis]

    def get_voters(self, axis:int) -> List[Opinion]:
        return [o for o in self.opinions.values() if o.bubble == axis and not o.is_politician and not o.is_source]

    def draw_parties(self, axis:int, ax:Axis) -> None:
        if not self.args.use_parties:
            return
        assert self.args.dimensions == 2
        assert self.args.num_politicians == 2
        assert isinstance(self.LayoutClass, OpinionLayout)
        a,b = self.get_politicians(axis)
        linex, liney = self.OpinionClass.get_midline(a, b)
        voters:List[Opinion] = self.get_voters(axis)
        ax.plot(linex, liney, color='g')
        avotes:int = sum(1 for x in voters if np.linalg.norm(x.pos - a.pos) < np.linalg.norm(x.pos - b.pos))
        bvotes:int = sum(1 for x in voters if np.linalg.norm(x.pos - a.pos) > np.linalg.norm(x.pos - b.pos))
        ax.text(a.pos[0], a.pos[1], str(avotes), color='r', horizontalalignment='center', verticalalignment='center')
        ax.text(b.pos[0], b.pos[1], str(bvotes), color='b', horizontalalignment='center', verticalalignment='center')
        ax.text(-0.1, 1, str(avotes), color='r', horizontalalignment='right', verticalalignment='bottom')
        ax.text(0.1, 1, str(bvotes), color='b', horizontalalignment='left', verticalalignment='bottom')

    def draw_lobbies(self, G:Network, axis:int, ax:Axis) -> None:
        for o in self.opinions.values():
            if o.is_lobby and o.bubble == axis:
                linex, liney = o.get_lobby_line()
                ax.plot(linex, liney, color=self.OpinionClass.get_color([o]), alpha=0.5, linewidth=1)

        edge_pos:List[Position] = []
        edge_colors:List[Color] = []
        for e in G.edges():
            o1:Opinion = self.opinions[e[0]]
            o2:Opinion = self.opinions[e[1]]
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

    def update_parties(self) -> None:
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
            voters:List[Opinion] = self.get_voters(axis)
            a_allies:List[Position] = [x.pos for x in voters if np.linalg.norm(x.pos - a.pos) < np.linalg.norm(x.pos - b.pos)]
            b_allies:List[Position] = [x.pos for x in voters if np.linalg.norm(x.pos - a.pos) > np.linalg.norm(x.pos - b.pos)]

            new_a:Position = sum(a_allies) / len(a_allies) if len(a_allies) > 0 else a.pos
            projections:List[float] = [np.dot(p.pos - b.pos, new_a - b.pos) / np.linalg.norm(new_a - b.pos)**2 for p in voters]
            new_a = np.median(projections) * 2 * (new_a - b.pos) + b.pos if np.median(projections) <= 0.5 else new_a
            new_a = np.clip(a.pos + np.clip(new_a - a.pos, -self.args.party_inertia, self.args.party_inertia), -1, 1)

            new_b:Position = sum(b_allies) / len(b_allies) if len(b_allies) > 0 else b.pos
            projections = [np.dot(p.pos - a.pos, new_b - a.pos) / np.linalg.norm(new_b - a.pos)**2 for p in voters]
            new_b = np.median(projections) * 2 * (new_b - a.pos) + a.pos if np.median(projections) <= 0.5 else new_b
            new_b = np.clip(b.pos + np.clip(new_b - b.pos, -self.args.party_inertia, self.args.party_inertia), -1, 1)

            a.pos = new_a
            b.pos = new_b


    def get_duplicate_lists(self, dd:Layout_Map) -> List[List[int]]:
        rev:Dict[Position, List[int]] = {}
        for k, v in dd.items():
            rev.setdefault(tuple(v), []).append(k)
        return list(rev.values())

    def get_size(self, xs:List[int]) -> float:
        return 50 * (np.log(len(xs)) + 1)

    def get_node_draw_data(self, G:Network, axis:int) -> Tuple[List[int], List[Edge], Layout_Map, List[Color], List[Color], List[float]]:
        positions:Layout_Map = self.positions(G)
        positions_filtered:Layout_Map = {x: p for x,p in positions.items() if self.opinions[x].bubble == axis and not self.opinions[x].is_lobby}
        unique_positions:List[List[int]] = self.get_duplicate_lists(positions_filtered)
        colors:Dict[int, Color] = {xs[0] : self.OpinionClass.get_color([self.opinions[x] for x in xs]) for xs in unique_positions}
        sizes:Dict[int, float] = {xs[0] : self.get_size(xs) for xs in unique_positions}
        unique_positions.sort(key=lambda xs: self.OpinionClass.nodesorter(xs, {x : self.opinions[x] for x in xs}))
        nodes:List[int] = [xs[0] for xs in unique_positions if self.opinions[xs[0]].bubble == axis]
        edges:List[Edge] = [e for e in G.edges() if (self.opinions[e[0]].bubble == axis or self.opinions[e[1]].bubble == axis) and not (self.opinions[e[0]].is_lobby or self.opinions[e[1]].is_lobby)]
        edge_colors:List[Color] = [within_region if (self.opinions[e[0]].bubble == self.opinions[e[1]].bubble) else between_region for e in edges]
        # nodelist, edgelist, pos, colors, edge_colors, sizes
        return nodes, edges, positions, [colors[x] for x in nodes], edge_colors, [sizes[x] for x in nodes]

within_region:Color = (0.0, 0.0, 0.0, 1.0)
between_region:Color = (1.0, 0.0, 0.0, 0.3)

class DunbarClass:
    def __init__(self, args:Arguments) -> None:
        sample:Callable[[int], int] = dunbar_funcs[args.dunbar_type]
        self.dunbar:Dict[int, int] = {x: sample(args.dunbar) for x in range(args.population)}

    def get(self, x:int) -> int:
        return self.dunbar[x]

class UpdateClass:
    def __init__(self, Dunbar:DunbarClass, Agents:AgentClass, G:Network, args:Arguments) -> None:
        self.Dunbar:DunbarClass = Dunbar
        self.Agents:AgentClass = Agents
        self.G:Network = G
        self.args:Arguments = args
        self.assert_all()

    def assert_all(self) -> None:
        pass

    def above_dunbar(self, x:int) -> None:
        pass

    def below_dunbar(self, x:int) -> None:
        pass

    def at_dunbar(self, x:int) -> None:
        pass

    def update(self, axes:List[Axis], draw:bool) -> List[Axis]:
        self.Agents.update_parties()
        for x in self.G:
            if self.Agents.opinions[x].is_source or self.Agents.opinions[x].is_lobby:
                pass
            elif self.Agents.opinions[x].is_politician and self.args.use_parties:
                pass
            else:
                if len(self.G[x]) > self.Dunbar.get(x):
                    self.above_dunbar(x)
                elif len(self.G[x]) < self.Dunbar.get(x):
                    self.below_dunbar(x)
                else:
                    self.at_dunbar(x)

                self.Agents.update_opinion(x,self.G[x])
        return self.draw_step(axes, draw)

    def draw_step(self, axes:List[Axis], draw:bool) -> List[Axis]:
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

def update(num:int, Updater:UpdateClass, axes:List[Axis], fig:Figure, args:Arguments, draw:bool) -> List[Axis]:
    axes = Updater.update(axes, draw)
    if draw:
        fig.suptitle('Step: {}'.format(num))
    if num in args.snapshot:
        assert draw
        assert len(args.snapshot_name) > 0
        fig.savefig('{}-figure-{}.png'.format(args.snapshot_name, num))
    return axes

class UpdateNone(UpdateClass):
    """ Under the dunbar number, agents make connections at random with anyone within range
        Over the dunbar number, agents break connections at random
        At the dunbar number, agents do nothing
    """
    def assert_all(self) -> None:
        assert self.args.rec_factor == 0

    def above_dunbar(self, x:int) -> None:
        self.G.remove_edge(x,choice(self.G.neighbors(x)))

    def below_dunbar(self:UpdateClass, x:int) -> None:
        y = choice(self.G.nodes())
        if y != x:
            self.G.add_edge(x,y)

class UpdateHigh(UpdateClass):
    """ Under the dunbar number, agents make connections in strict order of close opinions
        Over the dunbar number, agents break connections with the least shared opinion
        At the dunbar number, agents randomly probe for other neighbours with closer opinions
    """
    def assert_all(self) -> None:
        assert self.args.rec_factor == 0

    def above_dunbar(self, x:int) -> None:
        self.G.remove_edge(x,max(self.G[x], key=lambda y: self.Agents.opinion_distance(x,y)))

    def below_dunbar(self:UpdateClass, x:int) -> None:
        candidates = [y for y in self.G if y != x and y not in self.G[x]]
        if candidates:
            winner = self.G.add_edge(x,min(candidates, key=lambda y: self.Agents.opinion_distance(x,y)))

    def at_dunbar(self, x:int) -> None:
        if self.G[x]:
            y = choice(list(self.G.nodes))
            z = choice(list(self.G[x].keys()))
            if y != x and self.Agents.opinion_distance(x,y) < self.Agents.opinion_distance(x,z):
                self.G.add_edge(x,y)
                self.G.remove_edge(x,z)

class UpdateRec(UpdateClass):
    """ Under the dunbar number, agents make with the closest opinion or at random with probability rec_factor
        Over the dunbar number, agents break connections with the least shared opinion
        At the dunbar number, agents randomly probe for other neighbours with closer opinions
    """
    def assert_all(self) -> None:
        assert self.args.rec_factor >= 0 and self.args.rec_factor <= 1

    above_dunbar = UpdateHigh.above_dunbar

    def below_dunbar(self, x:int) -> None:
        if random() < self.args.rec_factor:
            UpdateNone.below_dunbar(self,x)
        else:
            UpdateHigh.below_dunbar(self,x)

    at_dunbar = UpdateHigh.at_dunbar


selectivity_funcs:Dict[Selectivity, typing.Type[UpdateClass]] = {Selectivity.Nothing: UpdateNone, Selectivity.Recommend: UpdateRec, Selectivity.Selective: UpdateHigh}
graph_layout:Dict[Layout, LayoutClass] = {Layout.Spring: SpringLayout(), Layout.Opinion: OpinionLayout()}
update_funcs:Dict[Update, Callable[[Arguments], Update_Func]] = {Update.Average: move_towards_average, Update.Inverse: inverse_force}
dunbar_funcs:Dict[DunbarType, Callable[[int], int]] = {DunbarType.Fixed: (lambda d: d), DunbarType.Poisson: (lambda d: np.random.poisson(d))}
color_funcs:Dict[Color_Func, Callable[[List[Opinion]],Color]] = {Color_Func.Opinion: opinion_color, Color_Func.Firmness: firmness_color, Color_Func.Charisma: charisma_color}
sorter_funcs:Dict[Color_Func, Callable[[List[int], Dict[int, Opinion]], float]] = {Color_Func.Opinion: opinion_sorter, Color_Func.Firmness: firmness_sorter, Color_Func.Charisma: charisma_sorter}


def animate(args:Arguments = Arguments()) -> None:
    G:Network = nx.empty_graph(args.population)
    Agents:AgentClass = AgentClass(args)
    Dunbar:DunbarClass = DunbarClass(args)
    Updater:UpdateClass = selectivity_funcs[args.selectivity](Dunbar, Agents, G, args)
    gridsize:int = int(np.ceil(np.sqrt(args.bubbles)))
    fig, axes_array = plt.subplots(gridsize, gridsize, squeeze=False)
    axes:List[Axis] = axes_array.flatten()
    for ax in axes:
        ax.set_aspect('equal')
    graph_ani = animation.FuncAnimation(fig, update, args.steps, fargs=(Updater, axes, fig, args, True), interval=100, blit=False, repeat=False)
    if args.show:
        plt.show()
    if args.save:
        graph_ani.save('{}.mp4'.format(args.vid_name), fps=10)
    plt.close()


def run_simulation(args:Arguments, components:List[List[int]]=None, histogram_x:List[List[int]]=None) -> None:
    G:Network = nx.empty_graph(args.population)
    Agents:AgentClass = AgentClass(args)
    Dunbar:DunbarClass = DunbarClass(args)
    Updater:UpdateClass = selectivity_funcs[args.selectivity](Dunbar, Agents, G, args)
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