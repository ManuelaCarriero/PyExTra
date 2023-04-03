# Introduction
## Aim of this repository and target users
<p align="center">
  <img 
    src="https://github.com/ManuelaCarriero/protein-synthesis-modeling/blob/main/Images/2023-01-23-Genelogo.png"
  >
</p>

This repository has been built in order to provide python programs that make simulations of the number of molecules produced by a gene. 
Of course, it is a very wide world because we can have different reactions and different models that describe such process in biology.
In this repository we explore three main genetic circuits described in section **Models** and provide also programs to analyse simulation data (that can not just be generated and left there!).
We have wrapped this repository into a unique name called *PyExTra* because of the acronym with gene *ex*pression and *tra*nslation processes that are simulated with his python programs.
Users which can be interested in *PyExTra* are beginners with gene expression simulations and analysis. You can just use it or explore its code to have some help for your research work.
*PyExTra*, indeed, was born with the aim to investigate about NF-kB model in colon cancer cells for my master degree thesis in Physics.
   

## Dependencies
**Python verion**: 3.9.x <br>
**Python modules**: numpy, scipy, pandas, stats.models,<br> 
scikit-learn, keras, tensorflow.<br>
**OS**: Windows10, Linux (Ubuntu).

## Structure of the project
First of all, make the simulation using the [Stochastic Simulation Algorithm (SSA)](https://github.com/ManuelaCarriero/PyExTra#stochastic-simulation-algorithm-ssa): <br>
[ssa_simulation.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/ssa_simulation.py) SSA simulation of the [**first protein synthesis model**](https://github.com/ManuelaCarriero/PyExTra#first-protein-synthesis-model);<br>
[ssa_simulation_autorepressor.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/ssa_simulation_autorepressor.py) SSA simulation of the [**Autorepressor model**](https://github.com/ManuelaCarriero/PyExTra#autorepressor-model);<br>
[ssa_simulation_2_ind_genes.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/ssa_simulation_2_ind_genes.py) SSA simulation of two independent genes;<br>
[ssa_simulation_toggleswitch.py](https://github.com/ManuelaCarriero/PyExTra/blob/main/ssa_simulation_toggleswitch.py) SSA simulation of the [**Toggle-Switch model**](https://github.com/ManuelaCarriero/PyExTra#toggle-switch-model);


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

The number of messenger RNAs **m** and proteins **n** produced can be described by a *chemical master equation*. 

## Autorepressor model

<p align="center">
  <img
    width="500"	 
    src="https://github.com/ManuelaCarriero/protein-synthesis-modeling/blob/main/Images/autorepressor.jpg"
  >
</p>

## Toggle-switch model

<p align="center">
  <img
    width="590" 
    src="https://github.com/ManuelaCarriero/protein-synthesis-modeling/blob/main/Images/toggleswitch.jpg"
  >
</p>

# Algorithms

## Stochastic Simulation Algorithm (SSA)

**Gillespie algorithm** (i.e. **stochastic simulation algorithm**) samples the probability distribution described by the master equation. The basic idea is that events are rare, discrete, *separate* events, i.e., each event is an arrivial of a Markov process and the algorithm is based on the following steps:
1. Choose some **initial states** (in this case, initial state of gene that is active or inactive, initial number of mRNA molecules **m<sub>0</sub>** and initial number of proteins **p<sub>0</sub>**); 
2. A state change will happen. In this approach, the event that can happen is one of the following: 
    * *gene activation*: the state of the gene switches from inactive to active and, in such case, there is no increase or decrease in the number of RNA or protein molecules;   
    * *gene inactivation*: the state of the gene switches from active to inactive and, in such case, there is no increase or decrease in the number of RNA or protein molecules;     
    * *RNA synthesis*: the number of RNA molecules increases by 1;  
    * *RNA degradation*: the number of RNA molecules decreases by 1;
    * *protein synthesis*: the number of protein molecules increases by 1; 
    * *protein degradation*: the number of protein molecules decreases by 1.</ul>
We can also call these events as *transitions* and each one happens with a certain **transition rate**. Higher rate of transition means that the event happens faster and so this event is more likely to occur. In this modeling, when the gene is activated, RNA molecules are produced with rate k<sub>1</sub>, and degraded with rate k<sub>2</sub>. When the gene is inactive, RNA molecules can only be degraded (**Φ** is the degraded molecule), but not produced. Proteins are produced with rate k<sub>3</sub> and degraded with rate k<sub>4</sub> and this happens only if the number of RNA molecules is greater than 0. Hence, the rate of protein production depends on the number of mRNAs but not vice-versa. Finally, activation and deactivation happen respectively with rate k<sub>a</sub> and k<sub>i</sub>. Thus, the second step of the algorithm is to define all the possible transitions that can happen and to decide the transition rates. All the reactions are summarized in the table below:

      | transition  | transition rate | m | p  |
      | ------------- | ------------- | :---: | :---: |
      | Gene(I) &rarr; Gene(A)  | k<sub>a</sub>  | 0 | 0 |
      | Gene(A) &rarr; Gene(I)  | k<sub>i</sub>  | 0  | 0 |
      | Gene(A) &rarr; Gene(A) + RNA  | k<sub>1</sub>  | +1  | 0 |
      | RNA &rarr; Φ  | k<sub>2</sub>  | -1  | 0 |
      | RNA &rarr; RNA + Protein  | k<sub>3</sub>  | 0  | +1 |
      | Protein &rarr; Φ  | k<sub>4</sub>  | 0 | -1 |
3. Calculate the **time of residency**, that is how much time the system is in that specific state. Since the distance between consecutive events in a Poisson process is Exponentially distributed, the time of residency has an *exponential* distribution with a characteristic time equal to the inverse of the sum of the total rates. 
4. Choose what state change, i.e. transition, will happen. For this purpose we use `random.choices` method that returns a list with the randomly selected element (i.e. a randomly selected transition) from the specified list of transitions with the possibility to choose the probability of each element selection.
5. Increment time by time step you calculate in step 3.
6. Update the state according to the state change chosen in step 4.
7. If the total time spent is less than a pre-determined stopping time, go to step 2. Else stop.

## Tau-leap algorithm

## SSA/Tau-leap algorithm

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

## Command line syntax
`python <configuration file> <simulation.py> -run` <br>
`python <configuration file> <simulation.py> -run_multiplesimulations`

## Example of results

## References