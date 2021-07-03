# Are we just in the perturbative regime?

In the 2009 paper [Pairwise Maximum Entropy Models for Studying Large Biological Systems: When They Can Work and When They Can't](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000380), Roudi et al. suggest that when we are in the perturbative regime, characterised by a small mean probability of observing neurons spike and small number of neurons `N`, the pairwise maxent model can appear to be a good model for a distribution. However, we cannot extrapolate the behaviour of the pairwise model to larger `N`, and predict that it will remain a good fit outside of the perturbative regime. We try and investigate these claims computationally.

In the `PertReigme.ipynb` notebook, we fit an Ising model to different distributions and see how good of a fit it is. 
