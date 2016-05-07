# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Wright-Fisher model of mutation and random genetic drift

# <markdowncell>

# A Wright-Fisher model has a fixed population size *N* and discrete non-overlapping generations. Each generation, each individual has a random number of offspring whose mean is proportional to the individual's fitness. Each generation, mutation may occur.

# <headingcell level=2>

# Setup

# <codecell>

import numpy as np
import itertools

# <headingcell level=2>

# Make population dynamic model

# <headingcell level=3>

# Basic parameters

# <codecell>

pop_size = 60

# <codecell>

seq_length = 100

# <codecell>

alphabet = ['A', 'T', 'G', 'C']

# <codecell>

base_haplotype = "AAAAAAAAAA"

# <headingcell level=3>

# Setup a population of sequences

# <markdowncell>

# Store this as a lightweight Dictionary that maps a string to a count. All the sequences together will have count *N*.

# <codecell>

pop = {}

# <codecell>

pop["AAAAAAAAAA"] = 40

# <codecell>

pop["AAATAAAAAA"] = 30

# <codecell>

pop["AATTTAAAAA"] = 30

# <codecell>

pop["AAATAAAAAA"]

# <headingcell level=3>

# Add mutation

# <markdowncell>

# Mutations occur each generation in each individual in every basepair.

# <codecell>

mutation_rate = 0.0001 # per gen per individual per site

# <markdowncell>

# Walk through population and mutate basepairs. Use Poisson splitting to speed this up (you may be familiar with Poisson splitting from its use in the [Gillespie algorithm](https://en.wikipedia.org/wiki/Gillespie_algorithm)). 
# 
#  * In naive scenario A: take each element and check for each if event occurs. For example, 100 elements, each with 1% chance. This requires 100 random numbers.
#  * In Poisson splitting scenario B: Draw a Poisson random number for the number of events that occur and distribute them randomly. In the above example, this will most likely involve 1 random number draw to see how many events and then a few more draws to see which elements are hit.

# <markdowncell>

# First off, we need to get random number of total mutations

# <codecell>

def get_mutation_count():
    mean = mutation_rate * pop_size * seq_length
    return np.random.poisson(mean)

# <markdowncell>

# Here we use Numpy's [Poisson random number](http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.poisson.html).

# <codecell>

get_mutation_count()

# <markdowncell>

# We need to get random haplotype from the population.

# <codecell>

pop.keys()

# <codecell>

[x/float(pop_size) for x in pop.values()]

# <codecell>

def get_random_haplotype():
    haplotypes = pop.keys() 
    frequencies = [x/float(pop_size) for x in pop.values()]
    total = sum(frequencies)
    frequencies = [x / total for x in frequencies]
    return np.random.choice(haplotypes, p=frequencies)

# <markdowncell>

# Here we use Numpy's [weighted random choice](http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html).

# <codecell>

get_random_haplotype()

# <markdowncell>

# Here, we take a supplied haplotype and mutate a site at random.

# <codecell>

def get_mutant(haplotype):
    site = np.random.randint(seq_length)
    possible_mutations = list(alphabet)
    possible_mutations.remove(haplotype[site])
    mutation = np.random.choice(possible_mutations)
    new_haplotype = haplotype[:site] + mutation + haplotype[site+1:]
    return new_haplotype

# <codecell>

get_mutant("AAAAAAAAAA")

# <markdowncell>

# Putting things together, in a single mutation event, we grab a random haplotype from the population, mutate it, decrement its count, and then check if the mutant already exists in the population. If it does, increment this mutant haplotype; if it doesn't create a new haplotype of count 1. 

# <codecell>

def mutation_event():
    haplotype = get_random_haplotype()
    if pop[haplotype] > 1:
        pop[haplotype] -= 1
        new_haplotype = get_mutant(haplotype)
        if new_haplotype in pop:
            pop[new_haplotype] += 1
        else:
            pop[new_haplotype] = 1

# <codecell>

mutation_event()

# <codecell>

pop

# <markdowncell>

# To create all the mutations that occur in a single generation, we draw the total count of mutations and then iteratively add mutation events.

# <codecell>

def mutation_step():
    mutation_count = get_mutation_count()
    for i in range(mutation_count):
        mutation_event()

# <codecell>

mutation_step()

# <codecell>

pop

# <headingcell level=3>

# Add genetic drift

# <markdowncell>

# Given a list of haplotype frequencies currently in the population, we can take a [multinomial draw](https://en.wikipedia.org/wiki/Multinomial_distribution) to get haplotype counts in the following generation.

# <codecell>

def get_offspring_counts():
    haplotypes = pop.keys() 
    frequencies = [x/float(pop_size) for x in pop.values()]
    return list(np.random.multinomial(pop_size, frequencies))

# <markdowncell>

# Here we use Numpy's [multinomial random sample](http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html).

# <codecell>

get_offspring_counts()

# <markdowncell>

# We then need to assign this new list of haplotype counts to the `pop` dictionary. To save memory and computation, if a haplotype goes to 0, we remove it entirely from the `pop` dictionary.

# <codecell>

def offspring_step():
    counts = get_offspring_counts()
    for (haplotype, count) in zip(pop.keys(), counts):
        if (count > 0):
            pop[haplotype] = count
        else:
            del pop[haplotype]

# <codecell>

offspring_step()

# <codecell>

pop

# <headingcell level=3>

# Combine and iterate

# <markdowncell>

# Each generation is simply a mutation step where a random number of mutations are thrown down, and an offspring step where haplotype counts are updated.

# <codecell>

def time_step():
    mutation_step()
    offspring_step()

# <markdowncell>

# Can iterate this over a number of generations.

# <codecell>

generations = 500

# <codecell>

def simulate():
    for i in range(generations):
        time_step()

# <codecell>

simulate()

# <codecell>

pop

# <headingcell level=3>

# Record

# <markdowncell>

# We want to keep a record of past population frequencies to understand dynamics through time. At each step in the simulation, we append to a history object.

# <codecell>

pop = {"AAAAAAAAAA": pop_size}

# <codecell>

history = []

# <codecell>

def simulate():
    clone_pop = dict(pop)
    history.append(clone_pop)
    for i in range(generations):
        time_step()
        clone_pop = dict(pop)
        history.append(clone_pop)

# <codecell>

simulate()

# <codecell>

pop

# <codecell>

history[0]

# <codecell>

history[1]

# <codecell>

history[2]

# <headingcell level=2>

# Analyze trajectories

# <headingcell level=3>

# Calculate diversity

# <markdowncell>

# Here, diversity in population genetics is usually shorthand for the statistic *&pi;*, which measures pairwise differences between random individuals in the population. *&pi;* is usually measured as substitutions per site.

# <codecell>

pop

# <markdowncell>

# First, we need to calculate the number of differences per site between two arbitrary sequences.

# <codecell>

def get_distance(seq_a, seq_b):
    diffs = 0
    length = len(seq_a)
    assert len(seq_a) == len(seq_b)
    for chr_a, chr_b in zip(seq_a, seq_b):
        if chr_a != chr_b:
            diffs += 1
    return diffs / float(length)

# <codecell>

get_distance("AAAAAAAAAA", "AAAAAAAAAB")

# <markdowncell>

# We calculate diversity as a weighted average between all pairs of haplotypes, weighted by pairwise haplotype frequency.

# <codecell>

def get_diversity(population):
    haplotypes = population.keys()
    haplotype_count = len(haplotypes)
    diversity = 0
    for i in range(haplotype_count):
        for j in range(haplotype_count):
            haplotype_a = haplotypes[i]
            haplotype_b = haplotypes[j]
            frequency_a = population[haplotype_a] / float(pop_size)
            frequency_b = population[haplotype_b] / float(pop_size)
            frequency_pair = frequency_a * frequency_b
            diversity += frequency_pair * get_distance(haplotype_a, haplotype_b)
    return diversity

# <codecell>

get_diversity(pop)

# <codecell>

def get_diversity_trajectory():
    trajectory = [get_diversity(generation) for generation in history]
    return trajectory

# <codecell>

get_diversity_trajectory()

# <headingcell level=3>

# Plot diversity

# <markdowncell>

# Here, we use [matplotlib](http://matplotlib.org/) for all Python plotting.

# <codecell>

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl

# <markdowncell>

# Here, we make a simple line plot using matplotlib's `plot` function.

# <codecell>

plt.plot(get_diversity_trajectory())

# <markdowncell>

# Here, we style the plot a bit with x and y axes labels.

# <codecell>

def diversity_plot():
    mpl.rcParams['font.size']=14
    trajectory = get_diversity_trajectory()
    plt.plot(trajectory, "#447CCD")    
    plt.ylabel("diversity")
    plt.xlabel("generation")

# <codecell>

diversity_plot()

# <headingcell level=3>

# Analyze and plot divergence

# <markdowncell>

# In population genetics, divergence is generally the number of substitutions away from a reference sequence. In this case, we can measure the average distance of the population to the starting haplotype. Again, this will be measured in terms of substitutions per site.

# <codecell>

def get_divergence(population):
    haplotypes = population.keys()
    divergence = 0
    for haplotype in haplotypes:
        frequency = population[haplotype] / float(pop_size)
        divergence += frequency * get_distance(base_haplotype, haplotype)
    return divergence

# <codecell>

def get_divergence_trajectory():
    trajectory = [get_divergence(generation) for generation in history]
    return trajectory

# <codecell>

get_divergence_trajectory()

# <codecell>

def divergence_plot():
    mpl.rcParams['font.size']=14
    trajectory = get_divergence_trajectory()
    plt.plot(trajectory, "#447CCD")
    plt.ylabel("divergence")
    plt.xlabel("generation") 

# <codecell>

divergence_plot()

# <headingcell level=3>

# Plot haplotype trajectories

# <markdowncell>

# We also want to directly look at haplotype frequencies through time.

# <codecell>

def get_frequency(haplotype, generation):
    pop_at_generation = history[generation]
    if haplotype in pop_at_generation:
        return pop_at_generation[haplotype]/float(pop_size)
    else:
        return 0

# <codecell>

get_frequency("AAAAAAAAAA", 4)

# <codecell>

def get_trajectory(haplotype):
    trajectory = [get_frequency(haplotype, gen) for gen in range(generations)]
    return trajectory

# <codecell>

get_trajectory("AAAAAAAAAA")

# <markdowncell>

# We want to plot all haplotypes seen during the simulation.

# <codecell>

def get_all_haplotypes():
    haplotypes = set()   
    for generation in history:
        for haplotype in generation:
            haplotypes.add(haplotype)
    return haplotypes

# <codecell>

get_all_haplotypes()

# <markdowncell>

# Here is a simple plot of their overall frequencies.

# <codecell>

haplotypes = get_all_haplotypes()
for haplotype in haplotypes:
    plt.plot(get_trajectory(haplotype))
plt.show()

# <codecell>

colors = ["#781C86", "#571EA2", "#462EB9", "#3F47C9", "#3F63CF", "#447CCD", "#4C90C0", "#56A0AE", "#63AC9A", "#72B485", "#83BA70", "#96BD60", "#AABD52", "#BDBB48", "#CEB541", "#DCAB3C", "#E49938", "#E68133", "#E4632E", "#DF4327", "#DB2122"]

# <codecell>

colors_lighter = ["#A567AF", "#8F69C1", "#8474D1", "#7F85DB", "#7F97DF", "#82A8DD", "#88B5D5", "#8FC0C9", "#97C8BC", "#A1CDAD", "#ACD1A0", "#B9D395", "#C6D38C", "#D3D285", "#DECE81", "#E8C77D", "#EDBB7A", "#EEAB77", "#ED9773", "#EA816F", "#E76B6B"]

# <markdowncell>

# We can use `stackplot` to stack these trajectoies on top of each other to get a better picture of what's going on.

# <codecell>

def stacked_trajectory_plot(xlabel="generation"):
    mpl.rcParams['font.size']=18
    haplotypes = get_all_haplotypes()
    trajectories = [get_trajectory(haplotype) for haplotype in haplotypes]
    plt.stackplot(range(generations), trajectories, colors=colors_lighter)
    plt.ylim(0, 1)
    plt.ylabel("frequency")
    plt.xlabel(xlabel)

# <codecell>

stacked_trajectory_plot()

# <headingcell level=3>

# Plot SNP trajectories

# <codecell>

def get_snp_frequency(site, generation):
    minor_allele_frequency = 0.0
    pop_at_generation = history[generation]
    for haplotype in pop_at_generation.keys():
        allele = haplotype[site]
        frequency = pop_at_generation[haplotype] / float(pop_size)
        if allele != "A":
            minor_allele_frequency += frequency
    return minor_allele_frequency

# <codecell>

get_snp_frequency(3, 5)

# <codecell>

def get_snp_trajectory(site):
    trajectory = [get_snp_frequency(site, gen) for gen in range(generations)]
    return trajectory

# <codecell>

get_snp_trajectory(3)

# <markdowncell>

# Find all variable sites.

# <codecell>

def get_all_snps():
    snps = set()   
    for generation in history:
        for haplotype in generation:
            for site in range(seq_length):
                if haplotype[site] != "A":
                    snps.add(site)
    return snps

# <codecell>

def snp_trajectory_plot(xlabel="generation"):
    mpl.rcParams['font.size']=18
    snps = get_all_snps()
    trajectories = [get_snp_trajectory(snp) for snp in snps]
    data = []
    for trajectory, color in itertools.izip(trajectories, itertools.cycle(colors)):
        data.append(range(generations))
        data.append(trajectory)    
        data.append(color)
    plt.plot(*data)   
    plt.ylim(0, 1)
    plt.ylabel("frequency")
    plt.xlabel(xlabel)

# <codecell>

snp_trajectory_plot()

# <headingcell level=2>

# Scale up

# <markdowncell>

# Here, we scale up to more interesting parameter values.

# <codecell>

pop_size = 50
seq_length = 100
generations = 500
mutation_rate = 0.0001 # per gen per individual per site

# <markdowncell>

# In this case there are $\mu$ = 0.01 mutations entering the population every generation.

# <codecell>

seq_length * mutation_rate

# <markdowncell>

# And the population genetic parameter $\theta$, which equals $2N\mu$, is 1.

# <codecell>

2 * pop_size * seq_length * mutation_rate

# <codecell>

base_haplotype = ''.join(["A" for i in range(seq_length)])
pop.clear()
del history[:]
pop[base_haplotype] = pop_size

# <codecell>

simulate()

# <codecell>

plt.figure(num=None, figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')
plt.subplot2grid((3,2), (0,0), colspan=2)
stacked_trajectory_plot(xlabel="")
plt.subplot2grid((3,2), (1,0), colspan=2)
snp_trajectory_plot(xlabel="")
plt.subplot2grid((3,2), (2,0))
diversity_plot()
plt.subplot2grid((3,2), (2,1))
divergence_plot()

# <codecell>


