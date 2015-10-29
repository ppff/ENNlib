#pragma once

#include "general.hpp"

namespace ENN
{

typedef float (*ActivationFunction)(float);
typedef float (*ActivationFunctionDerivative)(float);

typedef std::pair<

///This class describes the usual model of a neuron which you can easily modify to suit your needs.
class Neuron
{
	public:
		
		Neuron(); ///<constructor
		~Neuron(); ///<descructor

	private:

		std::list< std::shared_ptr<Neuron> > _inputNeurons;
		std::list< std::weak_ptr<Neuron> > _outputNeurons;

		

}

} //namespace ENN
