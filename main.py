import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from network import Context

class Arguments:
    def __init__(self, **kwargs):
        ##### Population properties
        self.dunbar = 5     # Dunbar number: how many connections normal agents want
        self.population=100 # Number of ordinary agents in the game
        self.regions=4      # Number of geographical regions agents are separated into
        self.dimensions=2   # Number of opinion dimensions (only 2 can be displayed, but more or less can be simulated if show=False and save=False)

        ##### Simulation properties
        self.snapshot=set()    # Frames of the simulation to save as snapshots
        self.snapshot_name=''  # Base file name for the snapshots
        self.steps=500         # Frames to run the simulation for
        self.show=True         # Whether to show the matplotlib animation as its running
        self.save=False        # Whether to save a video of the matplotlib animation
        self.vid_name=""       # File name for the saved video

        ##### General dynamics properties
        self.noise=0                   # Magnitude of noise to perturb agents every frame
        self.quality_loss = 0.5        # Cost of communicating between regions
        self.attract_strength = 0.001  # Factor affecting how much agents are attracted to others with similar opinions
        self.repel_strength = 0        # Factor affecting how much agents are repelled from others with different opinions

        ##### Opinion properties
        self.low_firmness = 0.1        # Firmness (resistance to change) for low firmness agents
        self.high_firmness = 0.9       # Firmness for high firmness agents
        self.low_charisma = 1.0        # Charisma (power to change others) for low charisma agents
        self.high_charisma = 1.0       # Charisma for high charisma agents

        ##### Party properties
        # Parties are special agents that compete for votes. They adjust their opinion to try to capture 50% of the vote
        self.use_parties = False
        self.party_interval = 1     # Let the parties move once per this many frames
        self.party_inertia = 0.01   # Limit on the distance parties can move per interval

        ##### Lobby properties
        # Lobbies are agents with only one opinion dimension that they care about, who are smeared across the other dimensions
        self.num_lobbies = 2
        self.fixed_lobby_dimension = None # If 'None' each lobby picks a random dimension, otherwise they all pick this one (e.g. 0, 1)
        self.fixed_lobby_charisma = None  # If 'None' each lobby has charisma 1.0, otherwise this value
        self.fixed_lobby_opinion = None   # If 'None' each lobby picks a random opinion value (in [-1,1]), otherwise this value

        for key, value in kwargs.items():
            setattr(self, key, value)

        assert self.noise >= 0 and self.noise <= 1
        assert self.dimensions == 2 or (not self.show and not self.save)


def update(num, context, axes, fig, args, should_draw):
    """ Run one frame of the simulation in context, writing changes to axes and fig """
    axes = context.update(axes, should_draw)
    if should_draw:
        fig.suptitle('Step: {}'.format(num))
    if num in args.snapshot:
        assert should_draw
        assert len(args.snapshot_name) > 0
        fig.savefig('{}-figure-{}.png'.format(args.snapshot_name, num))
    return axes

def animate(args = Arguments()):
    """ Initialise and run a single run of the simulation """
    context = Context(args)
    gridsize = int(np.ceil(np.sqrt(args.regions)))
    fig, axes_array = plt.subplots(gridsize, gridsize, squeeze=False)
    axes = axes_array.flatten()
    for ax in axes:
        ax.set_aspect('equal')
    graph_ani = animation.FuncAnimation(fig, update, args.steps, fargs=(context, axes, fig, args, True), interval=100, blit=False, repeat=False)
    if args.show:
        plt.show()
    if args.save:
        graph_ani.save('{}.mp4'.format(args.vid_name), fps=10)
    plt.close()

def run_diagrams():
    """ Some interesting runs to show off the model
        Run 1: Simple model with two random lobbies
        Run 2: Model with 4 regions and low cost of communication
        Run 3: Model with 4 regions and high cost of communication
        Run 4: Demonstration of parties
    """
    animate(Arguments(use_parties=False, num_lobbies=2, regions=1, quality_loss=0.5,  steps=201))
    animate(Arguments(use_parties=False, num_lobbies=0, regions=4, quality_loss=0.5,  steps=201))
    animate(Arguments(use_parties=False, num_lobbies=0, regions=4, quality_loss=0.99, steps=201))
    animate(Arguments(use_parties=True,  num_lobbies=0, regions=1, quality_loss=0.5,  steps=201))

def video_diagrams():
    """ Save the animations as videos """
    animate(Arguments(use_parties=False, num_lobbies=2, regions=1, quality_loss=0.5,  steps=201, show=False, save=True, vid_name="A-lobbies-vid", snapshot={10,200}, snapshot_name='A-lobbies'))
    animate(Arguments(use_parties=False, num_lobbies=0, regions=4, quality_loss=0.5,  steps=201, show=False, save=True, vid_name="A-connect-vid", snapshot={10,200}, snapshot_name='A-connect'))
    animate(Arguments(use_parties=False, num_lobbies=0, regions=4, quality_loss=0.99, steps=201, show=False, save=True, vid_name="A-apart-vid",   snapshot={10,200}, snapshot_name='A-apart'))
    animate(Arguments(use_parties=True,  num_lobbies=0, regions=1, quality_loss=0.5,  steps=201, show=False, save=True, vid_name="A-parties-vid", snapshot={10,200}, snapshot_name='A-parties'))


def run_simulation(args, components=None, histogram_x=None):
    """ Initialise and run a single run of the simulation without any animation
        Optionally, record some data in "components" and "histogram_x" arrays
    """
    context = Context(args)
    for n in range(args.steps):
        update(n, context, [], None, args, False)
        if components is not None:
            components.append(context.connected_components())
        if histogram_x is not None:
            histogram_x.append(context.histogram_x())


# Settings for the multi-simulation figures
figure_steps = 300  # How long to make each run
figure_repeats = 20 # How many runs to average over
figure_inputs = 20  # How many distinct parameter values to use on the X-axis for figures that vary some parameter

# These figures aggregate data from multiple runs. As such, they take a long time to run, so the process of making them is split in two:
# The first step is to generate the data and save it as a numpy *.npy file
# The second step is to generate the figure from the data (so you can experiment with different formatting without rerunning the simulation)

def figure_one():
    """ Graph of connected components across time (do people form filter bubbles?)
        No filter bubble = single connected component
        More bubbles = more components
    """
    runs = []
    for r in range(figure_repeats):
        components = []
        run_simulation(Arguments(steps=figure_steps, show=False, use_parties=False, num_lobbies=0, quality_loss=0.5, regions=1), components=components)
        runs.append(components)
        print("{}/{}".format(1 + r, figure_repeats))
    runs = np.array(runs)
    runs = np.mean(runs, axis=(0,2))
    print(runs.shape)
    np.save('figure_1.npy', runs)

def figure_two():
    """ Graph of "connected components per region" across communication penalty
        One component per region = everybody is geographically segregated but connected within their region
        Many components per region = even within a region people have filter bubbles (but maybe connected across regions)
    """
    result = []
    inputs = np.linspace(0.0, 0.9999, num=figure_inputs)
    for i, quality_loss in enumerate(inputs):
        runs = []
        for r in range(figure_repeats):
            components = []
            run_simulation(Arguments(steps=figure_steps, show=False, use_parties=False, num_lobbies=0, quality_loss=quality_loss, regions=10), components=components)
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
    """ Heatmap of the distribution of X-axis opinions against the charisma of a lobby on the X-axis
        As the lobby gets more charismatic, the opinions become more concentrated
    """
    result = []
    inputs = np.linspace(0.0, 1.0, num=figure_inputs)
    for i, lobby_charisma in enumerate(inputs):
        runs = []
        for r in range(figure_repeats):
            histogram = []
            run_simulation(Arguments(steps=figure_steps, show=False, use_parties=False, num_lobbies=1, quality_loss=0.5, regions=1, fixed_lobby_dimension=0, fixed_lobby_opinion=0, fixed_lobby_charisma=lobby_charisma), histogram_x=histogram)
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

if __name__ == "__main__":
    run_diagrams()
