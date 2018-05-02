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

class Agent:
    """ Represents an ordinary agent that forms connections and changes its opinion """
    def __init__(self, context, region):
        self.args = context.args
        self.region = region

        # Opinions are n-tuples of real numbers in [-1,1]
        self.opinion = np.array([random()*2 - 1 for _ in range(self.args.dimensions)])

        # How strongly the agent holds on to its opinion (vs adopting that of its communication partners)
        self.firmness = choice([self.args.low_firmness, self.args.high_firmness])
        self.has_high_firmness = self.firmness != self.args.low_firmness

        # How strongly the agent influences its communication partners
        self.charisma = choice([self.args.low_charisma, self.args.high_charisma])
        self.has_high_charisma = self.charisma != self.args.low_charisma


    def choose_connection_to_break(self, neighbors):
        """ When we have too many connections: break the one whose opinion is least similar """
        return max(neighbors, key=self.opinion_distance)

    def choose_connection_to_make(self, non_neighbors):
        """ When we have too few connections: pick the one whose opinion is most similar """
        return min(non_neighbors, key=self.opinion_distance, default=None)

    def choose_connection_to_change(self, neighbors, non_neighbors):
        """ When we have the right number of connections: pick a random non-neighbour and switch to them if they are closer """
        neighbors = list(neighbors)
        non_neighbors = list(non_neighbors)
        if len(neighbors) > 0 and len(non_neighbors) > 0:
            y = choice(non_neighbors)
            z = choice(neighbors)
            if self.opinion_distance(y) < self.opinion_distance(z):
                # Change z to y
                return z, y
        # Return None, None means no change
        return None, None

    def opinion_distance(self, x):
        """ Returns distance in opinion space between self and the Agent/Party/Lobby x
            Lower connection quality is implemented as increasing the distance
        """
        if isinstance(x, Lobby):
            return np.abs(self.opinion[x.lobby_dimension] - x.opinion) / self.connection_quality(x)
        else:
            return np.linalg.norm(self.opinion - x.opinion) / self.connection_quality(x)

    def connection_quality(self, x):
        return 1 if self.region == x.region else 1 - self.args.quality_loss

    def update_opinion(self, neighbors):
        """ Update opinion based on neighbors: the core logic is in inverse_force() but this function adds noise """
        self.opinion = self.inverse_force(neighbors)
        self.opinion += self.args.noise * np.array([random()*2 - 1 for _ in range(len(self.opinion))])
        self.opinion = np.clip(self.opinion, -1, 1)

    def inverse_force(self, neighbors):
        """ Update opinion based on neighbors
            Basically we want agents to move closer to neighbors with similar opinions
            And maybe also to be repelled by those with very different ones
            We want the attraction to be stronger the closer they are
            We also want the computation to be affected by the agent's firmness, the neighbors' charismas, and the connection quality

            We tried a few different functions and this one works nicely
            Consider the case of just one neighbor: suppose the distance between the agent and the neighbor is "dist"
            Then we move the agent closer by updating dist to:
            dist -> sqrt(dist^2(1 + repel_strength) - 2 * attract_strength * charisma * (1 - firmness) * connection_quality)
            This can go negative, in which case we just set dist to zero because unlike a Newtonian system we don't want agents to overshoot
            Also, rather than deal with the case of multiple neighbors, we just process the neighbors one at a time in a random order
        """
        k = self.args.attract_strength
        z = self.args.repel_strength
        x = self.opinion
        f = self.firmness
        attractors = [(y.get_nearest_lobby_point_to(self) if isinstance(y, Lobby) else y.opinion, y.charisma, self.connection_quality(y)) for y in neighbors]
        if len(attractors) > 0:
            np.random.shuffle(attractors)
            for y, c, q in attractors:
                dist = np.linalg.norm(x-y)
                var = dist*dist*(1+z) - 2*k*c*(1-f)*q
                if var < 0:
                    x = y
                else:
                    x = y + (x - y) * np.sqrt(var) / dist
            return np.clip(x, -1, 1)
        return self.opinion

class Party:
    """ Represents a political party that changes its opinion to win votes.
        The dynamics for parties are defined in the Context class
    """
    def __init__(self, context, region):
        self.args = context.args
        self.region = region

        self.firmness = 1.0
        self.charisma = 1.0

        self.opinion = np.array([random()*2 - 1 for _ in range(self.args.dimensions)])

class Lobby:
    """ Represents a lobby group with a fixed agenda. A lobby group only has an opinion in one dimension.
        On other dimensions, it is ambivalent. Thus, a lobby is represented by an n-1 dimensional hyperplane,
        and agents can connect to whichever point on that plane is closest to them.
    """
    def __init__(self, context, region):
        self.args = context.args
        self.region = region

        self.firmness = 1.0
        self.charisma = 1.0 if self.args.fixed_lobby_charisma is None else self.args.fixed_lobby_charisma
        # A lobby's opinion is a single number in [-1,1] representing an opinion in one dimension
        self.opinion = random()*2 - 1 if self.args.fixed_lobby_opinion is None else self.args.fixed_lobby_opinion
        self.lobby_dimension = choice(range(self.args.dimensions)) if self.args.fixed_lobby_dimension is None else self.args.fixed_lobby_dimension

    def get_nearest_lobby_point_to(self, x):
        """ Projects an agent x's opinion onto the lobby. This is easy because lobbies are parallel to the axes. """
        result = np.array(x.opinion)
        result[self.lobby_dimension] = self.opinion
        return result

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

    def update(self, axes, should_draw):
        """ Run one frame of the simulation, optionally updating a set of matplotlib axes (one axis for each region) """
        if self.args.use_parties:
            if self.current_party_interval < self.args.party_interval:
                self.current_party_interval += 1
            else:
                self.current_party_interval = 1
                self.update_parties()

        for x in self.agents:
            """ Under the dunbar number, agents make connections in strict order of close opinions
                Over the dunbar number, agents break connections with the least shared opinion
                At the dunbar number, agents randomly probe for other neighbours with closer opinions
            """
            if len(self.G[x]) > self.args.dunbar:
                self.G.remove_edge(x, x.choose_connection_to_break(self.G[x]))
            elif len(self.G[x]) < self.args.dunbar:
                new_neighbor = x.choose_connection_to_make(nx.non_neighbors(self.G, x))
                if new_neighbor:
                    self.G.add_edge(x, new_neighbor)
            else:
                old_neighbor, new_neighbor = x.choose_connection_to_change(self.G[x], nx.non_neighbors(self.G, x))
                if new_neighbor is not None:
                    self.G.add_edge(x,new_neighbor)
                    self.G.remove_edge(x,old_neighbor)

            x.update_opinion(self.G[x])

        if should_draw:
            for region, ax in enumerate(axes):
                ax.clear()
                ax.set_xlim([-1.1,1.1])
                ax.set_ylim([-1.1,1.1])
                self.draw_agents(region, ax)
                if self.args.num_lobbies > 0:
                    self.draw_lobbies(region, ax)
                if self.args.use_parties:
                    self.draw_parties(region, ax)
                ax.set_xticks([])
                ax.set_yticks([])

        return axes

    def update_parties(self):
        """ A party's voters are agents closer to that party than the other one
            Parties have two desires: they want to satisfy their base and to win votes
            We thus implement their behaviour in two phases
            First, the party moves to the average of its current voters' opinion
            Then, it moves toward the other party until it has 50% of the vote
            Of course, the other party does the same computation simultaneously, so the result is unpredictable
            Also, we limit the maximum speed at which parties can move, so after performing the computation above,
            the party may only move part of the way towards the computed position.
        """
        for region in range(self.args.regions):
            a,b = [p for p in self.parties if p.region == region]
            voters = [a for a in self.agents if a.region == region]
            a_allies = [x.opinion for x in voters if np.linalg.norm(x.opinion - a.opinion) < np.linalg.norm(x.opinion - b.opinion)]
            b_allies = [x.opinion for x in voters if np.linalg.norm(x.opinion - a.opinion) > np.linalg.norm(x.opinion - b.opinion)]

            new_a = sum(a_allies) / len(a_allies) if len(a_allies) > 0 else a.opinion
            projections = [np.dot(p.opinion - b.opinion, new_a - b.opinion) / np.linalg.norm(new_a - b.opinion)**2 for p in voters]
            new_a = np.median(projections) * 2 * (new_a - b.opinion) + b.opinion if np.median(projections) <= 0.5 else new_a
            new_a = np.clip(a.opinion + np.clip(new_a - a.opinion, -self.args.party_inertia, self.args.party_inertia), -1, 1)

            new_b = sum(b_allies) / len(b_allies) if len(b_allies) > 0 else b.opinion
            projections = [np.dot(p.opinion - a.opinion, new_b - a.opinion) / np.linalg.norm(new_b - a.opinion)**2 for p in voters]
            new_b = np.median(projections) * 2 * (new_b - a.opinion) + a.opinion if np.median(projections) <= 0.5 else new_b
            new_b = np.clip(b.opinion + np.clip(new_b - b.opinion, -self.args.party_inertia, self.args.party_inertia), -1, 1)

            a.opinion = new_a
            b.opinion = new_b

    def draw_parties(self, region, ax):
        """ Draws everything *except* the blue circle, which is drawn like an Agent in draw_agents() """
        a,b = [p for p in self.parties if p.region == region]

        linex, liney = self.get_midline(a, b)
        ax.plot(linex, liney, color='g')

        voters = [a for a in self.agents if a.region == region]
        avotes = sum(1 for x in voters if np.linalg.norm(x.opinion - a.opinion) < np.linalg.norm(x.opinion - b.opinion))
        bvotes = sum(1 for x in voters if np.linalg.norm(x.opinion - a.opinion) > np.linalg.norm(x.opinion - b.opinion))
        ax.text(a.opinion[0], a.opinion[1], str(avotes), color='r', horizontalalignment='center', verticalalignment='center')
        ax.text(b.opinion[0], b.opinion[1], str(bvotes), color='b', horizontalalignment='center', verticalalignment='center')
        ax.text(-0.1, 1, str(avotes), color='r', horizontalalignment='right', verticalalignment='bottom')
        ax.text(0.1, 1, str(bvotes), color='b', horizontalalignment='left', verticalalignment='bottom')

    def get_midline(self, x, y):
        """ Finds the line dividing the voters for x from the voters for y (i.e. the perpendicular bisector of the interval x-y)"""
        a = x.opinion
        b = y.opinion
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

    def draw_lobbies(self, region, ax):
        """ The trickiest part here is that we have to manually draw the edges connecting agents to lobbies,
            because networkx doesn't let us connect an edge to somewhere other than a node's fixed location,
            but our edges can connect to any part of a lobby's hyperplane.
            The code is based on the networkx.draw methods.
        """
        edge_pos = []
        edge_colors = []

        for x in self.lobbies:
            if x.region == region:
                if x.lobby_dimension == 0:
                    linex, liney = (x.opinion, x.opinion), (-2, 2)
                elif x.lobby_dimension == 1:
                    linex, liney = (-2, 2), (x.opinion, x.opinion)
                else:
                    assert False
                ax.plot(linex, liney, color=self.get_lobby_color(x), alpha=0.5, linewidth=1)

                neighbors = self.G[x]
                for y in neighbors:
                    edge_pos.append((x.get_nearest_lobby_point_to(y),y.opinion))
                    edge_colors.append(within_region_edge_color if x.region == y.region else between_region_edge_color)

        edge_collection = LineCollection(edge_pos, colors=edge_colors, antialiaseds=(1,), transOffset=ax.transData)
        edge_collection.set_zorder(1)
        ax.add_collection(edge_collection)

    def get_lobby_color(self, x):
        return lobby_color

    def draw_agents(self, region, ax):
        """ The tricky part here is that multiple agents can occupy the same position. We pick one agent to represent them all,
            and define get_color() and get_size() methods to compute what the colour and size of the aggregation should be
        """
        pos_to_nodelists = self.group_by_position([x for x in self.agents if x.region == region] + [x for x in self.parties if x.region == region])
        ordered_positions = list(pos_to_nodelists.keys())

        node_color = [self.get_color(pos_to_nodelists[p]) for p in ordered_positions]
        node_size = [self.get_size(pos_to_nodelists[p]) for p in ordered_positions]
        nodelist = [pos_to_nodelists[p][0] for p in ordered_positions]
        node_to_pos = {pos_to_nodelists[p][0] : p for p in ordered_positions}

        # Don't draw lobby edges because we do them manually in draw_lobbies()
        edgelist = [e for e in self.G.edges() if (e[0].region == region or e[1].region == region) and not (isinstance(e[0], Lobby) or isinstance(e[1], Lobby))]
        edge_color = [within_region_edge_color if (e[0].region == e[1].region) else between_region_edge_color for e in edgelist]

        for e in edgelist:
            if e[0] not in node_to_pos:
                node_to_pos[e[0]] = e[0].opinion
            if e[1] not in node_to_pos:
                node_to_pos[e[1]] = e[1].opinion

        nx.draw_networkx(self.G, pos=node_to_pos, nodelist=nodelist, edgelist=edgelist, ax=ax, node_color=node_color, edge_color=edge_color, node_size=node_size, with_labels=False, alpha=None)

    def group_by_position(self, agents):
        """ Returns {pos: [agent]} for the unique positions and lists of agents with those positions """
        rev = {}
        for x in agents:
            rev.setdefault(tuple(x.opinion), []).append(x)
        return rev

    def get_size(self, xs):
        """ Returns the size for a list of Agents xs occupying the same point
            This formula is arbitrary, I thought it looked good
        """
        return 50 * (np.log(len(xs)) + 1)

    def get_color(self, xs):
        """ Returns the colour for a list of Agents xs occupying the same point """
        has_party = False
        has_high_charisma = False
        has_high_firmness = False

        for x in xs:
            if isinstance(x, Party):
                has_party = True
                break
            if x.has_high_charisma:
                has_high_charisma = True
            if x.has_high_firmness:
                has_high_firmness = True

        if has_party:
            return party_color
        if has_high_charisma and has_high_firmness:
            return high_charisma_and_firmness_color
        if has_high_charisma:
            return high_charisma_color
        if has_high_firmness:
            return high_firmness_color
        return default_color

    def connected_components(self):
        """ Returns a list with the number of connected components in the connection graph for each region
            Note that a single component can span multiple regions, in which case it gets counted each time
        """
        ccs = list(nx.connected_components(self.G))
        return [sum(1 for cc in ccs if any(x.region == i for x in cc)) for i in range(self.args.regions)]

    def histogram_x(self):
        """ Returns a histogram of the x-values of agents' opinions (i.e. the number of agents with an x-value in each of 100 buckets) """
        result = [0 for _ in range(100)]
        for x in self.agents:
            fl = np.floor((x.opinion[0]+1)*50.0)
            if fl >= 100:
                fl = 99
            result[int(fl)] += 1
        return result