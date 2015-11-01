#include "neuron.hpp"
#include "connection.hpp"
#include "neuralnetwork.hpp"

using namespace ENN;

Neuron::Neuron(NeuralNetwork * network)
 : _network(network)
{
	_inputNeurons.clear();
	_outputNeurons.clear();
	
	_netValue = 0.f;
	_outputValue = 0.f;
	_derivativeOfErrorToNetValue = 0.f;
}

Neuron::Type Neuron::getType() const
{
	std::pair<unsigned, unsigned> position = _network->getNeuronPosition(this);
	
	if (position.first == 0)
		return Neuron::Type::Input;
	
	if (position.first == _network->getNumberOfLayers() - 1)
		return Neuron::Type::Output;
		
	if (getNumberOfInputs() == 0)
		return Neuron::Type::Bias;
		
	return Neuron::Type::Hidden;
}

void Neuron::addInput(ConnectionPtr connection)
{
	_inputNeurons.push_back(connection);
}

void Neuron::addOutput(ConnectionPtr connection)
{
	_outputNeurons.push_back(connection);
}

unsigned Neuron::getNumberOfInputs() const
{
	return _inputNeurons.size();
}

unsigned Neuron::getNumberOfOutputs() const
{
	return _outputNeurons.size();
}

bool Neuron::connectedToDestination(Neuron const * const destination) const
{
	bool result = false;
	
	for (ConnectionPtr c : _outputNeurons)
	{
		if (c->getDestination() == destination)
		{
			result = true;
			break;
		}
	}
		
	return result;
}

void Neuron::setConnectionWeight(Neuron * destination, float newWeight)
{
	for (ConnectionPtr c : _outputNeurons)
	{
		if (c->getDestination() == destination)
		{
			c->setWeight(newWeight);
			return;
		}
	}
	
	ERROR_MSG("Cannot change connection weight because connection can not be found");
}
		
void Neuron::compute()
{
	//If it doesn't have inputs, then it's a bias neuron and we shouldn't do anything
	if (getNumberOfInputs() == 0)
		return;
	
	//Compute net value
	_netValue = 0.f;
	
	for (ConnectionPtr c : _inputNeurons)
		_netValue += c->getSource()->getOutputValue() * c->getWeight();
	
	//Compute output value
	_outputValue = tanh(_netValue);
}

float Neuron::getOutputValue() const
{
	return _outputValue;
}

float Neuron::getNetValue() const
{
	return _netValue;
}

void Neuron::setOutputValue(float outputValue)
{
	if (getType() != Neuron::Type::Input && getType() != Neuron::Type::Bias)
	{
		ERROR_MSG("Cannot set the output value of a neuron which is neither an input nor a bias");
		return;
	}
	
	_outputValue = outputValue;
}

void Neuron::setDesiredOutputValue(float desiredOutputValue)
{
	if (getType() != Neuron::Type::Output)
	{
		ERROR_MSG("Cannot set the desired output value of a neuron which is not an output neuron");
		return;
	}
	
	_desiredOutput = desiredOutputValue;
}
		
void Neuron::computeDerativeOfErrorToNetValue()
{
	if (getType() == Neuron::Type::Output) //Output neuron, simple case
	{
		_derivativeOfErrorToNetValue = (_outputValue - _desiredOutput) * (1 - pow(tanh(_netValue), 2));
	}
	else //Hidden neuron, that's were the backpropagation algorithm kicks in
	{
		//The first step is to sum up the derivative of error with respect to the net value of neurons of the next layer, mutliplied by the connections weights
		for (ConnectionPtr c : _outputNeurons)
			_derivativeOfErrorToNetValue += c->getDestination()->getDerativeOfErrorToNetValue() * c->getWeight();
			
		//The second and last step is to multiply this partial error derivative by the derivative of the activation function applied to the net value
		_derivativeOfErrorToNetValue *= (1 - pow(tanh(_netValue), 2));
	}
}
	
float Neuron::getDerativeOfErrorToNetValue() const
{
	return _derivativeOfErrorToNetValue;
}

float Neuron::getError() const
{
	if (getType() != Neuron::Type::Output)
	{
		ERROR_MSG("This neuron is not an output neuron and therefore cannot have an error value");
		return 0.f;
	}
		
	return pow(_desiredOutput - _outputValue, 2) / 2.0;
}

std::string Neuron::toString() const
{
	std::stringstream ss;
	
	if (getType() == Neuron::Type::Bias)
	{
		ss << _outputValue;
	}
	else if (getType() == Neuron::Type::Input)
	{
		ss << "x" << _network->getNeuronPosition(this).second;
	}
	else
	{
		ss << "tanh(";

		for (auto c = _inputNeurons.begin() ; c != _inputNeurons.end() ; c++)
		{
			ss << (*c)->getWeight() << "*" << (*c)->getSource()->toString();
			if (c != std::prev(_inputNeurons.end()))
				ss << " + ";
		}

		ss << ")";
	}
	
	return ss.str();
}

