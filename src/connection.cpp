#include "connection.hpp"
#include "neuron.hpp"

using namespace ENN;

float Connection::_defaultLearningRate = 0.001; //Default learning rate is set to 0.001.

Connection::Connection(Neuron * from, Neuron * to)
 : _source(from), _destination(to)
{
	_weight = ((float)rand()) / ((float)RAND_MAX) - 0.5f; //Generate random weight between -0.5 and 0.5.
	_learningRate = _defaultLearningRate;
}

Neuron * Connection::getSource()
{
	return _source;
}

Neuron * Connection::getDestination()
{
	return _destination;
}

void Connection::setWeight(float weight)
{
	_weight = weight;
}

float Connection::getWeight() const
{
	return _weight;
}

void Connection::setLearningRate(float learningRate)
{
	_learningRate = learningRate;
}

void Connection::updateWeight()
{
	/* We are going to update the weight of the connection using the gradient descent algorithm. */
	
	/* First, we need to compute the partial derivative of the error with respect to the weight of this connection.
	 * It's the product of 2 things, thanks to the chain rule:
	 *   - the partial derivative of the error with respect to the destination net (a bit complicated -- is calculated by the destination neuron),
	 *   - the partial derivative of the destination net with respect to the weight of this connection (simply equals the output of the source neuron).
	 * 
	 * Then the only thing left to do is to multiply this product by the learning rate and substract the whole to the weight.
	 */
	 
	 //DEBUG_MSG("Update weight " << _weight << " with variation " << _destination->getDerativeOfErrorToNetValue() *  _source->getOutputValue());
	 _weight -= _learningRate * (_destination->getDerativeOfErrorToNetValue() *  _source->getOutputValue());
}
