# flowpm
Particle Mesh simulation in TensorFlow

Minimal working example is in flowpm.py. The steps are as follows - 
- create a config object which has details of simulation, such as box size, number of mesh points, number of steps
- define a graph to do PM. This function takes in the config object and is hence can be reused i.e. it can be used inside, as a part of a bigger graph
- do sess.run() using the graph to evaluate the variables one is interested in.

example_graphs.py has some more graphs showing how to define a graph that does a PM simulation from an initial field, how to combine the pm graph with other modules etc.
