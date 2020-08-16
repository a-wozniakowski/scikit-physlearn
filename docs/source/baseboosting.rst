=============
Base Boosting
=============

Introduction
============

The challenge of learning from a dearth of data has reinvigorated interest
in building machine learning systems, which emulate human learning and
thinking. As human researchers in mathematics and science regularly leverage
the understanding gained by their predecessors in order to make intellectual
progress, base boosting is a modification of the algorithmic paradigm of
boosting for improving the performance of an explicit model of the domain. 
Namely, base boosting supplants a weak learning algorithm that performs
slightly better than random guessing with an explicit model of the domain.
Hence, base boosting is a way of fitting an additive expansion in a set of
elementary basis functions, wherein the intercept term is generated by the
predecessors' explicit model of the domain.

Example
-------

To study the efficacy of this approach, we will study a proxy of expert
human-level performance on the benchmark task of simultaneously predicting
the entire energy spectrum of a Hamiltonian in a superconducting quantum
device calibration application. In this scenario, there is a shortage of
data due to operational cost of the experiment and the explicit model of
the device’s quantum behavior is state-of-the-art. 

References
----------
- Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
  "Boosting on the shoulders of giants in quantum device calibration",
  arXiv preprint arXiv:2005.06194 (2020).

- John Tukey. "Exploratory Data Analysis", Addison-Wesley (1977).

- Jerome Friedman. "Greedy function approximation: A gradient boosting machine,"
  Annals of Statistics, 29(5):1189–1232 (2001).

- Trevor Hastie, Robert Tibshirani, and Jerome Friedman.
  "The Elements of Statistical Learning", Springer (2009).