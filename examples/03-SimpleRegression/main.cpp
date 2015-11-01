#include <iostream>
#include <vector>
#include <algorithm>

#include "enn.hpp"

#define LEARNING_POINTS_STEPS 0.01

int main()
{
	srand(static_cast<unsigned>(time(0)));
	std::cout << "ENNlib example n3 : simple regression." << std::endl;
	std::cout << "Let's create our first *really* useful neural network." << std::endl;
	std::cout << "We'll compute some values with the function of example 2 and add up some noise. Then, we'll train the network and see if it can bypass the noise and get back to the original function (yeah, that's a regression)." << std::endl;

	//Create the network	  
	ENN::NeuralNetwork nn;
	
	nn.addLayer(1); //1 input neuron.
	nn.addLayer(2); //1 hidden neuron and 1 bias neuron.
	nn.addLayer(1); //1 output neuron
	
	nn.connect(0, 0, 1, 0); //Connect input to hidden
	nn.connect(1, 0, 2, 0); //Connect hidden to output
	nn.connect(1, 1, 2, 0); //Connect bias to output
	
	//Create the training set
	std::cout << std::endl;
	float maxNoise;
	std::cout << "Please enter the maximum noise value which can be added to the training values (-maxNoise <= noise <= +maxNoise) : " << std::flush;
	std::cin >> maxNoise;
	
	std::vector<float> valuesWithoutNoise;
	for (float x=-0.5f ; x<=0.5f ; x+=LEARNING_POINTS_STEPS)
	{
		valuesWithoutNoise.push_back( tanh(0.9*tanh(0.2*x) + 0.2) ); //tanh(0.9*tanh(0.2*x) + 0.2) is the function we want the NN to find
		const float noise = (((float)rand()) / ((float)RAND_MAX) * 2.f - 1.f) * maxNoise; //noise between -maxNoise and maxNoise
		nn.addLearningPoint( ENN::LearningVector({x}), ENN::LearningVector({valuesWithoutNoise.back() + noise}) );
	}
	
	//Let's train the neuron network and see what we get
	float learningRate = 0.1;
	std::cout << "The learning rate is automatically set to 0.1." << std::flush;
	
	std::cout << std::endl;
	std::cout << "Starting algorithm..." << std::endl;
	
	nn.setLearningRate(learningRate);
	unsigned cycles = nn.train(ENN::Verbose::Medium);
	std::cout << "After " << cycles << " training cycles, the neural network stabilized on this formula: " << nn.toString() << " (we want tanh(0.9*tanh(0.2*x0) + 0.2))." << std::endl;
	
	//Now let's compute some values with this NN and compare them to the original ones
	std::cout << std::endl;
	std::cout << "Let's compare the values of our network with the ones we should have..." << std::endl;
	
	std::vector<float> results;
	for (float x=-0.5f ; x<=0.5f ; x+=LEARNING_POINTS_STEPS)
	{
		results.push_back( nn.process(ENN::LearningVector({x}))[0] );
	}
	
	float error = 0.f;
	for (unsigned i=0 ; i<valuesWithoutNoise.size() ; i++)
	{
		error += pow(valuesWithoutNoise[i] - results[i], 2) / 2.f;
	}
	
	if (error < 0.01)
		std::cout << "The result is really good! (error: " << error << ")." << std::endl;
	else
		std::cout << "The result is not satisfying (error: " << error << "). You have probably added too much noise." << std::endl;
	
	std::cout << "Don't hesitate to run this example multiple times with different noise values!" << std::endl;
	
	return EXIT_SUCCESS;
}
