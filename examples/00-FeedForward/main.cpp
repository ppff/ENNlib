#include <iostream>
#include <vector>

#include "enn.hpp"

int main()
{
	std::cout << "ENNlib example n0 : initiation to neural networks." << std::endl;
	std::cout << "We will create a very simple neural network (2 inputs connected to 1 output) and try computing some values." << std::endl;
	std::cout << "It is called a feed forward neural network because there are no cycles in the network." << std::endl;
			
	//Create the network	  
	std::cout << std::endl;
	std::cout << "Let's create a network with 2 input neurons connected to 1 output neurons with the following weights: 0.4 and 0.6." << std::endl;
	
	ENN::NeuralNetwork nn;
	
	nn.addLayer(2); //2 input neurons.
	nn.addLayer(1); //1 output neuron.
	
	nn.connect(0, 0, 1, 0); //Connect input0 to output
	nn.connect(0, 1, 1, 0); //Connect input1 to output
	
	//Set weights
	nn.setConnectionWeight(0, 0, 1, 0, 0.4);
	nn.setConnectionWeight(0, 1, 1, 0, 0.6);
	
	//Print the network
	std::cout << "The neural network's formula (as returned by the network) is: " << nn.toString() << " (should be tanh(0.4*x0 + 0.6*x1))." << std::endl;
	
	//Process some values
	std::cout << std::endl;
	std::cout << "Let's now make the NN compute some values and compare them to what we should get with our function." << std::endl;
	std::vector<ENN::LearningVector> values { {0.1, 0.5}, {0.2, 0.9}, {1.5, -0.5} };
	
	for (ENN::LearningVector & pairOfInputs : values)
	{
		ENN::LearningVector results = nn.process(pairOfInputs);
		float expectedResult = tanh(0.4*pairOfInputs[0] + 0.6*pairOfInputs[1]);
		std::cout << "Input (" << pairOfInputs[0] << ", " << pairOfInputs[1] << ") => output " << results[0] << " (should be " << expectedResult << " -- difference: " << expectedResult - results[0] << ")." << std::endl;
	}
	
	std::cout << "If the differences aren't null (or very close to 0), you might want to burn your PC right now." << std::endl;
	
	return EXIT_SUCCESS;
}
