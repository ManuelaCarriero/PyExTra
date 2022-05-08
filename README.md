# Stochastic simulation of biological circuits
A simple model for central dogma of biology is represented by the following sketch:
<p align="center">
  <img 
    width="450"
    src="https://github.com/ManuelaCarriero/protein-synthesis-modeling/blob/main/Images/simplest_protein_synthesis_model_.jpg"
  >
</p>

The number of messenger RNAs (mRNAs) and proteins can be described by a *chemical master equation*. The rate of protein production depends on the number of mRNAs but not vice-versa (**m** and **n** are the numbers of mRNA and proteins respectively; **Φ** is the degraded molecule).

**Gillespie algorithm** (i.e. **stochastic simulation algorithm**) samples the probability distribution described by the master equation. The basic idea is that events are rare, discrete, *separate* events, i.e., each event is an arrivial of a Poisson process and the algorithm is based on the following steps:
1. It starts with some **initial states** (in this case, initiale state of gene that is active or inactive, initial number of molecules of mRNA **m<sub>0</sub>** and initial number of proteins **p<sub>0</sub>**); 
2. Then a state change will happen. In this approach, the event that can happen is one of the following: 
    * gene activation: the state of the gene switches from inactive to active and, in such case, there is no increase or decrease in the number of RNA or protein molecules.   
    * gene inactivation: the state of the gene switches from active to inactive and, in such case, there is no increase or decrease in the number of RNA or protein molecules;     
    * RNA synthesis: the number of RNA molecules increases by 1;  
    * RNA degradation: the number of RNA molecules decreases by 1;
    * protein synthesis: the number of protein molecules increases by 1; 
    * protein degradation: the number of protein molecules decreases by 1.</ul>
We can also call these events as *transitions* and each one happens with a certain **transition rate**. Higher rate of transition represents an event that happens faster and so an event that is more likely to occur. In this modeling, when the gene is activated, RNA molecules are produced with rate k<sub>1</sub>, and degraded with rate k<sub>2</sub>. When the gene is inactive, RNA molecules can only be degraded, but not produced. Proteins are produced with rate k<sub>3</sub> and degraded with rate k<sub>4</sub> and only if the number of RNA molecules is greater than 0. 
Hence, the second step of the algorithm is to enumerate all the possible transitions that can happen and to decide the transition rates.
4. Calculate the **time of residency**, i.e. how much time the system is in that specific state. Since the time it takes for arrival of a Poisson process is Exponentially distributed, the time of residency has an *exponential* distribution with a characteristic time equal to the inverse of the sum of the total rates. 
5.
