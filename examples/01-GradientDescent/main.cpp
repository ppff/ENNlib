#include <iostream>
#include <vector>
#include <algorithm>

#include "enn.hpp"

#define LEARNING_POINTS_STEPS 0.1

int main()
{
	std::cout << "ENNlib example n1 : (stochastic) gradient descent test." << std::endl;
	std::cout << "In this example, we create the same neuron network as in example 0 and train it in order to find the function we want." << std::endl;
	std::cout << "Note that since we don't have hidden neurons here, there's no need for the backpropagation algorithm." << std::endl;
			
	//Create the network	  
	ENN::NeuralNetwork nn;
	
	nn.addLayer(2); //2 input neurons.
	nn.addLayer(1); //1 output neuron.
	
	nn.connect(0, 0, 1, 0); //Connect input0 to output
	nn.connect(0, 1, 1, 0); //Connect input1 to output
	
	//Create the training set
	std::vector<float> desiredValues;
	for (float x=-0.5f ; x<=0.5f ; x+=LEARNING_POINTS_STEPS)
	{
		for (float y=-0.5f ; y<=0.5f ; y+=LEARNING_POINTS_STEPS)
		{
			desiredValues.push_back(tanh(0.4*x + 0.6*y)); //tanh(0.4*x + 0.6*y) is the function we want the NN to find
			nn.addLearningPoint( ENN::LearningVector({x, y}), ENN::LearningVector({desiredValues.back()}) ); 
		}
	}
	
	//Let's train the neuron network and see what we get ; the parameters of the gradient descent will be given by the user
	std::cout << std::endl;
	float learningRate;
	std::cout << "In order to update their weights, the connections need a value called a learning rate. The higher it is, the more their values will change.\
				  \nIf it is too great, the algorithm might diverge. If it is too small, the convergence might be very slow. A good value is 0.01.\
				  \nPlease type your value here : " << std::flush;
	std::cin >> learningRate;
	
	std::cout << std::endl;
	std::cout << "Starting algorithm..." << std::endl;
	
	nn.setLearningRate(learningRate);
	unsigned cycles = nn.train(ENN::Verbose::Medium);
	std::cout << "After " << cycles << " training cycles, the neural network stabilized on this formula: " << nn.toString() << " (we want tanh(0.4*x0 + 0.6*x1))" << std::endl;
	
	//Now let's compute some values with this NN and compare them to the original ones
	std::cout << std::endl;
	std::cout << "We are now going to compare the values calculated with our arbitrary function to the values calculated with the neural network." << std::endl;
	std::cout << "The error is computed as the sum of the error on each point, this individual error being 1/2 * (desired value - NN output)^2." << std::endl;
	
	std::vector<float> results;
	for (float x=-0.5f ; x<=0.5f ; x+=LEARNING_POINTS_STEPS)
	{
		for (float y=-0.5f ; y<=0.5f ; y+=LEARNING_POINTS_STEPS)
		{
			results.push_back( nn.process(ENN::LearningVector({x, y}))[0] );
		}
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
	
	return EXIT_SUCCESS;
}
