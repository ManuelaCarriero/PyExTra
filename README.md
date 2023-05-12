# Introduction
## Aim of this repository and target users
<p align="center">
  <img 
    src="https://github.com/ManuelaCarriero/protein-synthesis-modeling/blob/main/Images/2023-01-23-Genelogo.png"
  >
</p>

This repository is made in order to provide python programs that make simulations of the number of molecules produced by a gene. 
Of course, it is a very wide world because we can have different reactions and different models that describe such process in biology.
In this repository we explore three main genetic circuits described in section **Models** and provide also programs to analyse simulation data (that can not just be generated and left there!).
We have wrapped this repository into a unique name called *PyExTra* because of the acronym with gene *ex*pression and *tra*nslation processes that are simulated with his python programs.
Users which can be interested in *PyExTra* are beginners with gene expression simulations and analysis. You can just use it or explore its code to have some help for your research work.
*PyExTra*, indeed, was born with the aim to investigate about NF-kB model in colon cancer cells for my master degree thesis in Physics (it is [Thesis.tex](https://github.com/ManuelaCarriero/PyExTra/blob/main/Thesis/Thesis.tex) available in [Thesis](https://github.com/ManuelaCarriero/PyExTra/tree/main/Thesis) folder and the thesis presentation [MDThesisPresentation_ManuelaCarriero.pptx](https://github.com/ManuelaCarriero/PyExTra/blob/main/MDThesisPresentation_ManuelaCarriero.pptx)).
   
## Dependencies
**Python verion**: 3.9.x <br>
**Python modules**: 
* numpy, scipy, pandas, stats.models, matplotlib, seaborn;<br> 
* scikit-learn, keras, tensorflow;<br>
* ssqueezepy, pywt, scaleogram.

**OS**: Windows10, Linux (Ubuntu).

## Structure of the project
1. First of all, make the simulation using the [Stochastic Simulation Algorithm (SSA)](https://github.com/ManuelaCarriero/PyExTra#stochastic-simulation-algorithm-ssa): <br>
[ssa_simulation.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/ssa_simulation.py) SSA simulation of the [**first protein synthesis model**](https://github.com/ManuelaCarriero/PyExTra#first-protein-synthesis-model);<br>
[ssa_simulation_autorepressor.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/ssa_simulation_autorepressor.py) SSA simulation of the [**Autorepressor model**](https://github.com/ManuelaCarriero/PyExTra#autorepressor-model);<br>
[ssa_simulation_2_ind_genes.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/ssa_simulation_2_ind_genes.py) SSA simulation of two independent genes;<br>
[ssa_simulation_toggleswitch.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/ssa_simulation_toggleswitch.py) SSA simulation of the [**Toggle-Switch model**](https://github.com/ManuelaCarriero/PyExTra#toggle-switch-model).<br>
[ssa_simulation_nfkb.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/ssa_simulation_nfkb.py) SSA simulation of the first NF-kB model.<br>
2. Then using:<br>
[plots.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/plots.py) you can plot the number of molecules as function of time and the distribution of states;<br>
[acf.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/acf.py) you can plot the autocorrelation functions refered to the temporal series. [statistics.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/statistics.py) is a python *script* to calculate and plot the quantiles autocorrelation.<br>
You can also use [fft.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/fft.py) and [wavelet.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/wavelet.py): they are python *scripts* for the Fast Fourier Transform and the Continuous Wavelet Transform analysis.<br>
3. Furthermore, [acfk_DNNregressor.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/acfk_DNNregressor.py) is a *script* that given the autocorrelation values, it estimates the model parameters using a deep neural network built with keras. The autocorrelation values are generated by either [firstmodel_ssa_acfs_ks.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/firstmodel_ssa_acfs_ks.py) or [autorepressor_ssa_acfs_ks.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/autorepressor_ssa_acfs_ks.py) or [toggleswitch_ssa_acfs_ks.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/toggleswitch_ssa_acfs_ks.py).<br>
4. Other types of algorithms to simulate biological stochastic processes have been implemented:<br>
[tauleap_simulation.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/tauleap_simulation.py): it makes simulations using the [Tau-leap algorithm](https://github.com/ManuelaCarriero/PyExTra#tau-leap-algorithm); <br>
[hybrid_simulation.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/hybrid_simulation.py): it makes the simulation using an algorithm that combines [SSA](https://github.com/ManuelaCarriero/PyExTra#stochastic-simulation-algorithm-ssa) algorithm with the [Tau-leap](https://github.com/ManuelaCarriero/PyExTra/blob/main/hybrid_simulation.py) (**NOTE: it is new !** The algorithm has been developed by me and my Supervisor during the thesis work).
5. [ODEvsStoch.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/ODEvsStoch.py) is a *script* to compare deterministic approach (Ordinary Differential Equation) vs stochastic simulation results.

## Testing
As you can notice from the Dependencies section, we have made use of many built-in functions. However, this work is mainly based on the simulations 
and the autocorrelation function, hence tests are performed on these two main topics, on which this research is based, in order to ensure that
the basics work properly.

# Models

## First protein synthesis model 
A simple model for central dogma of biology is represented by the following biological circuit:
<p align="center">
  <img 
    width="400"
    src="https://github.com/ManuelaCarriero/protein-synthesis-modeling/blob/main/Images/simplest_protein_synthesis_model.jpg"
  >
</p>

The number of messenger RNAs **m** and proteins **n** produced is a stochastic process and the *Chemical Master Equation* describes the time evolution of the probability of a stochastic system to be in a particular configuration.

## Autorepressor model
In this model there is a gene and the gene product **p** represses its own transcription.
<p align="center">
  <img
    width="500"	 
    src="https://github.com/ManuelaCarriero/protein-synthesis-modeling/blob/main/Images/autorepressor.jpg"
  >
</p>

## Toggle-switch model
The genetic toggle-switch is a system of two mutually repressing genes.
<p align="center">
  <img
    width="590" 
    src="https://github.com/ManuelaCarriero/PyExTra/blob/main/Thesis/toggleswitchrightmodel.png"
  >
</p>

## First NF-kB model
<p align="center">
  <img
    width="590" 
    src="https://github.com/ManuelaCarriero/PyExTra/blob/main/Images/nfkbschemaconparametriNUOVO.png"
  >
</p>

# Algorithms
In this documentation, the stochastic simulation algorithms are explained by considering the [first protein synthesis model](https://github.com/ManuelaCarriero/PyExTra#first-protein-synthesis-model). If you want to have a look at all the transitions and the reaction probability rates considered for the other models, see the presentation [MDThesisPresentation_ManuelaCarriero.pptx](https://github.com/ManuelaCarriero/PyExTra/blob/main/MDThesisPresentation_ManuelaCarriero.pptx).

## Stochastic Simulation Algorithm (SSA)

**Gillespie algorithm** (i.e. **stochastic simulation algorithm**) samples the probability distribution described by the master equation. The basic idea is that events are rare, discrete, *separate* events, i.e., each event is an arrivial of a Markov process and the algorithm is based on the following steps:
1. Choose some **initial states** (in the [first protein synthesis model](https://github.com/ManuelaCarriero/PyExTra#first-protein-synthesis-model) case, initial state of gene that is active or inactive, initial number of mRNA molecules **m<sub>0</sub>** and initial number of proteins **p<sub>0</sub>**); 
2. A state change will happen. In this model, the event that can happen is one of the following: 
    * *gene activation*: the state of the gene switches from inactive to active and, in such case, there is no increase or decrease in the number of RNA or protein molecules;   
    * *gene inactivation*: the state of the gene switches from active to inactive and, in such case, there is no increase or decrease in the number of RNA or protein molecules;     
    * *RNA synthesis*: the number of RNA molecules increases by 1;  
    * *RNA degradation*: the number of RNA molecules decreases by 1;
    * *protein synthesis*: the number of protein molecules increases by 1; 
    * *protein degradation*: the number of protein molecules decreases by 1.</ul>
We can also call these events as *transitions* and each one happens with a certain **transition rate**. Higher rate of transition means that the event happens faster and so this event is more likely to occur. In this modeling, when the gene is activated, RNA molecules are produced with rate k<sub>1</sub>, and degraded with rate k<sub>2</sub>. When the gene is inactive, RNA molecules can only be degraded (**Φ** is the degraded molecule), but not produced. Proteins are produced with rate k<sub>3</sub> and degraded with rate k<sub>4</sub> and this happens only if the number of RNA molecules is greater than 0. Hence, the rate of protein production depends on the number of mRNAs but not vice-versa. Finally, activation and deactivation happen respectively with rate k<sub>a</sub> and k<sub>i</sub>. Thus, the second step of the algorithm is to **define all the possible transitions** that can happen and to **calculate the transition probabilities**. All the reactions are summarized in the table below:

      | transition  | transition rate | m | p  |
      | ------------- | ------------- | :---: | :---: |
      | Gene(I) &rarr; Gene(A)  | k<sub>a</sub>  | 0 | 0 |
      | Gene(A) &rarr; Gene(I)  | k<sub>i</sub>  | 0  | 0 |
      | Gene(A) &rarr; Gene(A) + RNA  | k<sub>1</sub>  | +1  | 0 |
      | RNA &rarr; Φ  | k<sub>2</sub>  | -1  | 0 |
      | RNA &rarr; RNA + Protein  | k<sub>3</sub>  | 0  | +1 |
      | Protein &rarr; Φ  | k<sub>4</sub>  | 0 | -1 |
3. Calculate the **time of residency**, that is how much time the system is in that specific state. Since the distance between consecutive events in Markov processes is Exponentially distributed, the time of residency has an *exponential* distribution with a characteristic time equal to the inverse of the sum of the total rates. 
4. Choose what state change, i.e. transition, will happen. For this purpose we use `random.choices` method that returns a list with the randomly selected element (i.e. a randomly selected transition) from the specified list of transitions with the possibility to choose the probability of each element selection.
5. Increment time by time step you calculate in step 3.
6. Update the state according to the state change chosen in step 4.
7. If the total time spent is less than a pre-determined stopping time, go to step 2. Else stop.

You can look at the flowchart that explains in a more "intuitive" way the SSA algorithm:<br>

<p align="center">
  <img 
    src="https://github.com/ManuelaCarriero/PyExTra/blob/main/Images/SSAPresentation.png"
  >
</p>
 
## Tau-leap algorithm
1. Choose some **initial states** and **reaction time step** $\tau$ (tau);
2. Identification of all possible reaction events;
3. Calculation of reaction probabilities;
4. The number of reactions in the reaction time step is given by a Poisson distribution with mean equal to reactions probabilities times $\tau$;
5. Increment time by $\tau$;
6. Update the state according to the state change decided in step 4;
7. If the total time spent is less than a pre-determined stopping time, go to step 2. Else stop.   

You can look at the flowchart that explains in a more "intuitive" way the Tauleap algorithm:<br>

<p align="center">
  <img 
    src="https://github.com/ManuelaCarriero/PyExTra/blob/main/Images/TauleapPresentation.png"
  >
</p>

## SSA/Tau-leap algorithm

SSA/Tau-leap algorithm, as such called "hybrid", is presented directly through the flowchart:<br>

<p align="center">
  <img 
    src="https://github.com/ManuelaCarriero/PyExTra/blob/main/Images/HybridPresentation.png"
  >
</p>

1. The SSA updates more numerous molecular species such as genes, the tau-leap less numerous molecular species such as RNAs and proteins;
2. The tau-leap algorithm uses the gene state residency time as time step in the algorithm;
3. There are two important checks: the number of updated molecules by the tauleap algorithm has to be greater or equal than 0 and the difference between reaction probabilities has to lower or equal to a threashold (to monitor accuracy of results);
4. If step 3 is not satisfied, it means that we need to make smaller steps, to we divide it by one half; 
we check if the new smaller tau is greater of equal than 3 times the characteristic time of the system state given by the SSA considering all the reactions
because, if it is smaller, it is worth to make the SSA algorithm;
If yes, we go on applying the tau-leap algorithm to the new residency time untill we reach a time that is close to the initial gene state residency time value (the code is `math.isclose(time_count, gil_time, rel_tol=0.5)`).


# Running PyExTra

## Installation
In order to use PyExTra you can just clone this repository in the folder that you desire.
So first open your terminal, move in the directory you want to put PyExTra and then type:
<br>
`git clone <git_repo_url> <directory_name_where_you_clone_PyExTra>`
<br> 
For example, if you want to clone this repository in a folder named "PyExTra", you
can type this command:
<br>
`git clone https://github.com/ManuelaCarriero/PyExTra PyExTra`
<br> 

## Usage
PyExTra is mainly based on command lines to let you use the available Python programs in a easier and friendly way. <br>

## Command line syntax
PyExTra command lines are described following the [Structure of the project](https://github.com/ManuelaCarriero/PyExTra#structure-of-the-project) order. <br>

1. Starting from the basics, if you want to run the SSA simulation of one of the studied models, the command line synthax is the following: <br> 
`python <simulation_model.py> <configuration.txt> -run` <br>
If you want to run more than one simulation, beware of specifying the number of simulations that you desire in the `configuration.txt` file and change the command line simply in this way: <br> 
`python <simulation_model.py> <configuration.txt> -run_multiplesimulations` <br>
After running one of these, you will have a file called `gillespiesimulation_results.csv` in your working directory or, if you choose to run more than one simulation, the program gives a list of files named `gillespieresults_seed<number_of_the_random_seed>` (i.e. run 4 simulations, you will obtain 4 files named `gillespieresults_seed1`, `gillespieresults_seed2`, `gillespieresults_seed3` and `gillespieresults_seed4`). <br>
You can ask for the help `python <simulation_model.py> -h` in order to know all the possible commands you can ask to the program with the description of what they do.

2. Stay in the same working directory where the simulation results are. You can run this command line in order to plot the number of molecules over time of observation: <br>
`python plots.py <configuration.txt> -time_plot` <br>
This is in case of first model simulation results. For the other models the syntax is similar: <br>
`python plots.py <configuration.txt> -<model>_time_plot` <br>
You need to specify the type of model you are considering. <br>
For the precise list of commands you can ask for the help even in this case: `python plots.py -h` <br>
Importantly, you can plot the distribution of states (i.e. the stationary distribution) using: <br>
`python plots.py <configuration.txt> -distribution`

3. In order to plot the autocorrelation values as function of sampling time: <br>
`python acf.py <configuration.txt> -plot_acf_RNA` <br>
This is the case you want to plot the autocorrelation of only the RNA number of molecules in case of SSA simulation results. <br>
Also in this case, you can ask for the help: `python acf.py -h`<br>

4. Follow the command line syntax already described in 1. and 2. for [hybrid_simulation.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/hybrid_simulation.py) and 
[tauleap_simulation.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/tauleap_simulation.py). 


The others are scripts that you can use for the analysis of results.






## Example of usage and results
Let us consider the [first protein synthesis model](https://github.com/ManuelaCarriero/PyExTra#first-protein-synthesis-model) and play with its parameters configuration. <br>
Open file `configuration.txt`. You can start from a basic configuration where rate constants: ka = 1, ki = 0.5, k1 = 1, k2 = 0.1, k3 = 1, k4 = 0.1, k5 = 0, that is a gene that tends to be more active than inactive.<br>
Run the SSA simulation `python ssa_simulation.py configuration.txt -run` <br>
Then you would like to observe the time course behavior: `python plots.py configuration.txt -time_plot` <br>

<p align="center">
  <img 
    src="https://github.com/ManuelaCarriero/PyExTra/blob/main/Images/ka1ki0.5timeplot.png"
  >
</p>

And mostly the stationary distribution: `python plots.py configuration.txt -distribution` after modifying the time limit value from 1000 to 10000 in `configuration.txt` in order to obtain the Poisson distribution:<br> 

<p align="center">
  <img 
    src="https://github.com/ManuelaCarriero/PyExTra/blob/main/Images/ka1ki0.5distributionplot.png"
  >
</p>

If you change the type of regulation by making the gene more inactive than active (for instance, ka = 0.1 and ki = 1), you should see a distribution of states whose states with higher residency time are those with lower number of molecules (in particular, a peak at zero molecules). 
Thereby, the Poisson distribution will change its shape assuming a longer tail on the right. <br>
<br>
Try yourself and, if you want, let me know ! PyExTra let you simulate gene expression in a way that you can manipulate your biological system.

# References

https://github.com/UniboDIFABiophysics/programmingCourseDIFA <br>

In particular: <br>

https://unibodifabiophysics.github.io/programmingCourseDIFA/Lesson_AF_03_continuous_time_random_walks.html <br>

<p align="center">
  <img 
    src="https://github.com/ManuelaCarriero/PyExTra/blob/main/Images/Genethatisreallygreeting.gif"
  >
</p>