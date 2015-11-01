#include <iostream>
#include <vector>
#include <algorithm>

#include "enn.hpp"

#define LEARNING_POINTS_STEPS 0.01

int main()
{
	std::cout << "ENNlib example n2 : backpropagation test." << std::endl;
	std::cout << "In this example, we create a set of points with an arbitrary function, then make a 4-neuron neural network try to find it." << std::endl;
	std::cout << "We still use the gradient descent method but add a hidden neuron, hence using the backpropagation algorithm." << std::endl;
	std::cout << "Note that the other inner neuron is a bias neuron. We then have the 4 different types of neurons: input, output, hidden, bias." << std::endl;

	//Create the network	  
	ENN::NeuralNetwork nn;
	
	nn.addLayer(1); //1 input neuron.
	nn.addLayer(2); //1 hidden neuron and 1 bias neuron.
	nn.addLayer(1); //1 output neuron
	
	nn.connect(0, 0, 1, 0); //Connect input to hidden
	nn.connect(1, 0, 2, 0); //Connect hidden to output
	nn.connect(1, 1, 2, 0); //Connect bias to output
	
	//Create the training set
	std::vector<float> desiredValues;
	for (float x=-0.5f ; x<=0.5f ; x+=LEARNING_POINTS_STEPS)
	{
		desiredValues.push_back( tanh(0.9*tanh(0.2*x) + 0.2) ); //tanh(0.9*tanh(0.2*x) + 0.2) is the function we want the NN to find
		nn.addLearningPoint( ENN::LearningVector({x}), ENN::LearningVector({desiredValues.back()}) );
	}
	
	//Let's train the neuron network and see what we get ; the parameters of the gradient descent will be given by the user
	std::cout << std::endl;
	float learningRate;
	std::cout << "Please type your learning rate here : " << std::flush;
	std::cin >> learningRate;
	
	std::cout << std::endl;
	std::cout << "Starting algorithm..." << std::endl;
	
	nn.setLearningRate(learningRate);
	unsigned cycles = nn.train(ENN::Verbose::Medium);
	std::cout << "After " << cycles << " training cycles, the neural network stabilized on this formula: " << nn.toString() << " (we want tanh(0.9*tanh(0.2*x0) + 0.2))." << std::endl;
	std::cout << "Note that it is possible that the function we end with doesn't really look like the original one, but that's normal considering the fact that at this point, already multiple weight sets might give the same results." << std::endl;
	
	//Now let's compute some values with this NN and compare them to the original ones
	std::cout << std::endl;
	std::cout << "Let's compare the values of our network with the ones we should have..." << std::endl;
	
	std::vector<float> results;
	for (float x=-0.5f ; x<=0.5f ; x+=LEARNING_POINTS_STEPS)
	{
		results.push_back( nn.process(ENN::LearningVector({x}))[0] );
	}
	
	float error = 0.f;
	for (unsigned i=0 ; i<desiredValues.size() ; i++)
	{
		error += pow(desiredValues[i] - results[i], 2) / 2.f;
	}
	
	if (error < 0.0001)
		std::cout << "The result is really good! (error: " << error << ")." << std::endl;
	else
		std::cout << "The result is not satisfying (error: " << error << "). The algorithm might have stabilized on a non-global minimum, try entering a smaller learning rate." << std::endl;
	
	std::cout << "If you run this example again, you will probably not find the same solution. When the connections initalize their weights, they take a random value, hence, they place the neural network on another point of the error function, changing the direction the gradient will take and the minimum it will reach." << std::endl;
	
	return EXIT_SUCCESS;
}
