# Structure of the project

# First protein synthesis model 
A simple model for central dogma of biology is represented by the following biological circuit:
<p align="center">
  <img 
    width="400"
    src="https://github.com/ManuelaCarriero/protein-synthesis-modeling/blob/main/Images/simplest_protein_synthesis_model.jpg"
  >
</p>

The number of messenger RNAs **m** and proteins **n** produced can be described by a *chemical master equation*. 

#Stochastic Simulation Algorthm (SSA)

**Gillespie algorithm** (i.e. **stochastic simulation algorithm**) samples the probability distribution described by the master equation. The basic idea is that events are rare, discrete, *separate* events, i.e., each event is an arrivial of a Poisson process and the algorithm is based on the following steps:
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

# Tau-leap

# Example of results