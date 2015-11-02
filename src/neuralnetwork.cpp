#include "neuralnetwork.hpp"

using namespace ENN;

NeuralNetwork::NeuralNetwork()
{
	srand(static_cast<unsigned>(time(0)));
}

void NeuralNetwork::addLayer(unsigned numberOfNeurons)
{
	std::list<Neuron> l;
	
	for (unsigned i=0 ; i<numberOfNeurons ; i++)
		l.emplace_back(this);
	
	_neurons.push_back(l);
}

unsigned NeuralNetwork::getNumberOfLayers() const
{
	return _neurons.size();
}

unsigned NeuralNetwork::getNumberOfNeuronsOnLayer(unsigned layer) const
{
	if (layer >= getNumberOfLayers())
	{
		ERROR_MSG("Cannot get number of neurons on layer " << layer << " because it does not exist");
		return 0;
	}
	
	auto it = _neurons.begin();
	std::advance(it, layer);
	return it->size();
}

bool NeuralNetwork::neuronExists(unsigned layer, unsigned index) const
{
	if (layer >= getNumberOfLayers())
		return false;
		
	if (index >= getNumberOfNeuronsOnLayer(layer))
		return false;
		
	return true;
}

Neuron * NeuralNetwork::getNeuron(unsigned layer, unsigned index)
{
	if (!neuronExists(layer, index))
	{
		ERROR_MSG("Cannot access neuron (" << layer << ", " << index << ") because it does not exist, returning nullptr");
		return nullptr;
	}
	
	auto layer_it = _neurons.begin();
	std::advance(layer_it, layer);
	
	auto neuron_it = layer_it->begin();
	std::advance(neuron_it, index);
	
	return &(*neuron_it);
}

Neuron const * const NeuralNetwork::getNeuron(unsigned layer, unsigned index) const
{
	if (!neuronExists(layer, index))
	{
		ERROR_MSG("Cannot access neuron (" << layer << ", " << index << ") because it does not exist, returning nullptr");
		return nullptr;
	}
	
	auto layer_it = _neurons.begin();
	std::advance(layer_it, layer);
	
	auto neuron_it = layer_it->begin();
	std::advance(neuron_it, index);
	
	return &(*neuron_it);
}

std::pair<unsigned, unsigned> NeuralNetwork::getNeuronPosition(Neuron const * const neuron) const
{
	unsigned layer = -1, index = -1;
	bool found = false;
	
	for (auto layer_it = _neurons.begin() ; layer_it != _neurons.end() ; layer_it++)
	{
		if (found)
			break;
		
		for (auto neuron_it = layer_it->begin() ; neuron_it != layer_it->end() ; neuron_it++)
		{
			if (neuron == &(*neuron_it))
			{
				layer = std::distance(_neurons.begin(), layer_it);
				index = std::distance(layer_it->begin(), neuron_it);
				found = true;
				break;
			}
		}
	}
	
	if (!found)
		WARNING_MSG("Could not find neuron " << neuron << " inside network");
	
	return std::make_pair(layer, index);
}

void NeuralNetwork::connect(unsigned sourceLayer, unsigned sourceIndex, unsigned destinationLayer, unsigned destinationIndex)
{
	if (destinationLayer == 0)
	{
		ERROR_MSG("Cannot connect a neuron to an input neuron (layer 0)");
		return;
	}

	if (sourceLayer == getNumberOfLayers()-1)
	{
		ERROR_MSG("Cannot create a connection from an output neuron (last layer)");
		return;
	}

	if (sourceLayer == destinationLayer)
	{
		ERROR_MSG("Cannot connect two neurons on the same layer");
		return;
	}

	if (destinationLayer < sourceLayer)
	{
		ERROR_MSG("Destination neuron layer must be superior to source neuron layer");
		return;
	}
	
	if (!neuronExists(sourceLayer, sourceIndex))
	{
		ERROR_MSG("Source neuron (" << sourceLayer << ", " << sourceIndex << ") does not exist");
		return;
	}
	
	if (!neuronExists(destinationLayer, destinationIndex))
	{
		ERROR_MSG("Destination neuron (" << destinationLayer << ", " << destinationIndex << ") does not exist");
		return;
	}
	
	if (connectionExists(sourceLayer, sourceIndex, destinationLayer, destinationIndex))
	{
		INFO_MSG("Connection already exists, not creating a new one");
		return;
	}
	
	Neuron * src = getNeuron(sourceLayer, sourceIndex);
	Neuron * dest = getNeuron(destinationLayer, destinationIndex);

	connect(src, dest);
}

void NeuralNetwork::connect(Neuron * source, Neuron * destination)
{
	ConnectionPtr connection = std::make_shared<Connection>(source, destination);
	source->addOutput(connection);
	destination->addInput(connection);
	_connections.push_back(connection);
}

void NeuralNetwork::setConnectionWeight(unsigned sourceLayer, unsigned sourceIndex, unsigned destinationLayer, unsigned destinationIndex, float weight)
{
	if (!connectionExists(sourceLayer, sourceIndex, destinationLayer, destinationIndex))
	{
		ERROR_MSG("Cannot change connection weight: connection does not exist");
		return;
	}
	
	getNeuron(sourceLayer, sourceIndex)->setConnectionWeight(getNeuron(destinationLayer, destinationIndex), weight);
}

bool NeuralNetwork::connectionExists(unsigned sourceLayer, unsigned sourceIndex, unsigned destinationLayer, unsigned destinationIndex) const
{
	return getNeuron(sourceLayer, sourceIndex)->connectedToDestination(getNeuron(destinationLayer, destinationIndex));
}

void NeuralNetwork::connectAllLayers()
{
	//For each neuron from the first layer to the second to last layer
	for (auto layer = _neurons.begin() ; layer != std::prev(_neurons.end()) ; layer++)
	{
		for (auto neuron = layer->begin() ; neuron != layer->end() ; neuron++)
		{
			//Connect the neuron to all the neurons of the next layer
			for (auto nextLayerNeuron = std::next(layer)->begin() ; nextLayerNeuron != std::next(layer)->end() ; nextLayerNeuron++)
			{
				connect(&(*neuron), &(*nextLayerNeuron));
			}
		}
	}
}	

void NeuralNetwork::addLearningPoint(LearningVector const & inputs, LearningVector const & outputs)
{
	if (getNumberOfNeuronsOnLayer(0) != inputs.size())
	{
		ERROR_MSG("Input learning vector size (" << inputs.size() << ") and number of input neurons (" << getNumberOfNeuronsOnLayer(0) << ") are not equal");
		return;
	}
	
	if (getNumberOfNeuronsOnLayer(getNumberOfLayers()-1) != outputs.size())
	{
		ERROR_MSG("Output learning vector size (" << outputs.size() << ") and number of output neurons (" << getNumberOfNeuronsOnLayer(getNumberOfLayers()-1) << ") are not equal");
		return;
	}
		
	_learningSet.push_back(std::make_pair(inputs, outputs));
}

void NeuralNetwork::clearLearningSet()
{
	_learningSet.clear();
}

void NeuralNetwork::appendLearningSet(LearningSet const & set)
{
	_learningSet.insert(_learningSet.begin(), set.begin(), set.end());
}

void NeuralNetwork::setLearningRate(float learningRate)
{
	for (ConnectionPtr c : _connections)
		c->setLearningRate(learningRate);
}

unsigned NeuralNetwork::train(Verbose verbose)
{
	/* Here is how we train a neural network:
	 *  - compute the output values (from the second layer to the last layer)
	 *  - compute the NN error ; if it is close enough to the last one (null derivative), stop the algorithm
	 *  - compute the derivative of the error with respect to the net value for each neuron (from the last layer to the second one)
	 *  - update the weight of each connection (no order)
	 *  - go back to first step
	 */
	 
	//Before all this we need to make sure that all bias neurons are a non zero value (let's say 1)
	setBiasNeurons(1.f);
	 
	//Shit's getting real now
	float error = std::numeric_limits<float>::max();
	float lastError = std::numeric_limits<float>::min();
	unsigned cycles = 0;
	
	while (std::abs(error - lastError) > 0.00001)
	{
		lastError = error;
		error = 0.f;
		
		if (verbose == Verbose::Full)
			DEBUG_MSG("STARTING CYCLE " << cycles);
		
		unsigned step = 0;
		for (LearningPoint const & p : _learningSet)
		{
			setInputs(p.first);
			computeOutputs();
			setDesiredOutputs(p.second);
			error += getError();
			
			if (verbose == Verbose::Full)
				DEBUG_MSG("   Learning point " << step << ": error = " << error);
			step++;
			
			computeDerivativesOfErrorToNets();
			updateWeights();
		}
		
		if (verbose >= Verbose::Medium)
			DEBUG_MSG("Cycle " << cycles << ": error = " << error);
		cycles++;
	}
	
	return cycles;
}

LearningVector NeuralNetwork::process(LearningVector const & inputs)
{
	setInputs(inputs);
	computeOutputs();
	
	LearningVector result;
	
	for (auto neuron = _neurons.back().begin() ; neuron != _neurons.back().end() ; neuron++)
	{
		result.push_back(neuron->getOutputValue());
	}
	
	return result;
}

void NeuralNetwork::setBiasNeurons(float constantValue)
{
	for (auto layer = std::next(_neurons.begin()) ; layer != std::prev(_neurons.end()) ; layer++) //Bias neurons cannot be inside the first or last layers
	{
		for (auto neuron = layer->begin() ; neuron != layer->end() ; neuron++)
		{
			if (neuron->getNumberOfInputs() == 0) //Means it's a bias neuron
				neuron->setOutputValue(constantValue);
		}
	}
}

void NeuralNetwork::setInputs(LearningVector const & values)
{
	if (getNumberOfNeuronsOnLayer(0) != values.size())
	{
		ERROR_MSG("Input learning vector size (" << values.size() << ") and number of input neurons (" << getNumberOfNeuronsOnLayer(0) << ") are not equal");
		return;
	}

	std::list<Neuron> & layer = _neurons.front();
	for (auto neuron = layer.begin() ; neuron != layer.end() ; neuron++)
		neuron->setOutputValue(values[std::distance(layer.begin(),neuron)]);
}
	
void NeuralNetwork::setDesiredOutputs(LearningVector const & values)
{
	if (getNumberOfNeuronsOnLayer(getNumberOfLayers()-1) != values.size())
	{
		ERROR_MSG("Output learning vector size (" << values.size() << ") and number of output neurons (" << getNumberOfNeuronsOnLayer(getNumberOfLayers()-1) << ") are not equal");
		return;
	}

	std::list<Neuron> & layer = _neurons.back();
	for (auto neuron = layer.begin() ; neuron != layer.end() ; neuron++)
		neuron->setDesiredOutputValue(values[std::distance(layer.begin(),neuron)]);
}

void NeuralNetwork::computeOutputs()
{
	for (auto layer = std::next(_neurons.begin()) ; layer != _neurons.end() ; layer++) //We should never compute the input layer (it is fixed by the user)
	{
		#ifdef ENABLE_OPENMP
		__gnu_parallel::for_each(layer->begin(), layer->end(), [](Neuron & neuron){ neuron.compute(); });
		#else
		for (auto neuron = layer->begin() ; neuron != layer->end() ; neuron++)
		{
			neuron->compute();
		}
		#endif
	}
}

float NeuralNetwork::getError() const
{
	float error = 0.f;
	
	for (auto neuron = _neurons.back().begin() ; neuron != _neurons.back().end() ; neuron++)
	{
		error += neuron->getError();
	}
	
	return error;
}

void NeuralNetwork::computeDerivativesOfErrorToNets()
{
	for (auto layer = _neurons.rbegin() ; layer != std::prev(_neurons.rend()) ; layer++) //We're going backward from the last layer to the second layer (intput neurons cannot have any contribution to the network error)
	{
		#ifdef ENABLE_OPENMP
		__gnu_parallel::for_each(layer->begin(), layer->end(), [](Neuron & neuron){ neuron.computeDerativeOfErrorToNetValue(); });
		#else
		for (auto neuron = layer->begin() ; neuron != layer->end() ; neuron++)
		{
			neuron->computeDerativeOfErrorToNetValue();
		}
		#endif
	}
}

void NeuralNetwork::updateWeights()
{
	for (auto connection = _connections.begin() ; connection != _connections.end() ; connection++)
	{
		(*connection)->updateWeight();
	}
}

std::string NeuralNetwork::toString() const
{
	std::stringstream ss;
	
	for (auto neuron = _neurons.back().begin() ; neuron != _neurons.back().end() ; neuron++)
	{
		ss << neuron->toString();
		if (neuron != std::prev(_neurons.back().end()))
			ss << " + ";
	}
	
	return ss.str();
}

