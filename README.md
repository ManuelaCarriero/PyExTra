# Stochastic simulation of biological circuits
The simples model for central dogma of biology is the following:
<br />

<br />
The number of messenger RNAs (mRNAs) and proteins can be described by a chemical master equation. The rate of protein production depends on the number of mRNAs but not vice-versa (m and n are the numbers of mRNA and proteins respectively; Φ is the degraded molecule).

Gillespie algorithm samples the probability distribution described by the master equation. The basic idea is that events are rare, discrete, separate events, i.e., each event is an arrivial of a Poisson process. In this case, each event is a change of state that move either the copy number of mRNA or protein up or down by 1 in this case. Gillepsie algorithm starts with some initial states (m<sub>0</sub> and p<sub>0</sub>).
