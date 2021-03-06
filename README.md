## Synopsis

EENlib (Easy Neural Network library) is a small and simple cross platform C++ library to create neural networks.

## Motivations

Deep learning frameworks are usually huge and difficult to handle if you're new to machine learning.
This library, specified on neural networks, offers a very easy way to understand neural networks.
The best thing to do is to read the pdf (coming soon) and follow the examples. Once you're done, you can try to make some neural networks yourself!

Since the library is meant to be easy to handle, it certainly doesn't reach the best performance it could have. 
However, it was programmed with optimisation in mind and was structured to be easily parallelized on CPU and GPU.

## Features

* Easy to use
* Handles feed forward networks
* Handles stochastic gradient descent method
* Built with genetic algorithm training in mind and parallelization for maximum performance

## Todo

* Add reccurent networks support
* Add the possibility to train networks with a genetic algorithm instead of the gradient descent
* Parallelize with openMP and (later) CUDA

## Compile

Simply open your terminal at the root of the projet and type `make`. This will build the library.  
To compile the examples, type `make examples`.  
To create the documentation, type `make doc` (should already be done in the repository).

## Documentation

The library has:
* A full documentation generated by Doxygen
* A large set of progressive examples
* A pdf (coming soon) explaining the math of neural networks

## License

The library is licensed under the Do What The Fuck You Want To Public License (WTFPL).
http://www.wtfpl.net/
