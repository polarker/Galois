#Neural Network Library#

This library is a research of implementing ultimately generic neural network without performance overhead.

##Interesting Features:
  * Network inside network: every network is considered as a filter and could be used to contruct more complicated networks;
  * Network sharing and cloning: sub-networks could share paramenters and are clonable;
  * In place memory optimization by default: one neuron could accept signals from several other neurons with just one copy of n-dim array memory;
  * Dynamic traning: it's able to train only part of the whole network (e.g. RNN with varied input lenght); it's able to fix part of the whole network;
  * Dynamic network [WIP]: fast dynamic network construction, optimization with cache.

The guiding principles of design includes both efficiency and convenience. For user guide, please have a look at the example fold. The library uses extensively new features of C++11 to make the code simple and clear. Using Galois is just as simple as drawing dataflow graphs. Galois is also efficient. For the same mnist_mlp model (from torch demos) on Mac Pro 2013, the consumed time of each epoch is: Torch ~ 40s; Keras ~ 60s; Galois ~ 30s. Only implemented for CPU for the moment.
