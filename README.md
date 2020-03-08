# keras-a3c
Asynchronous Advantage Actor-Critic in Tensorflow 2.0 w/ Keras

## Details

A clean implementation of A3C focused on readability. The goal was to make everything as close as possible to pseudocode while also illustrating important aspects of _implementation_ that are glossed over in the theory. 
Some of these aspects include: 
- How to share updates across asynchronous agents
- How to elegantly add entropy term to loss in DL frameworks (methology carries over to PyTorch)
