#pragma once

#include "general.hpp"

namespace ENN
{
	
class NeuralNetwork;
class Connection;
typedef std::shared_ptr<Connection> ConnectionPtr;

///This class describes the usual model of a neuron which you can easily modify to suit your needs.
class Neuron
{
	public:
	
		enum class Type { Input, Output, Hidden, Bias };
		
		Neuron(NeuralNetwork * network); ///<constructor
		
		Type getType() const;
		
		void addInput(ConnectionPtr connection);
		void addOutput(ConnectionPtr connection);
		unsigned getNumberOfInputs() const;
		unsigned getNumberOfOutputs() const;
		bool connectedToDestination(Neuron const * const destination) const;
		void setConnectionWeight(Neuron * destination, float newWeight);
		
		void compute();
		float getOutputValue() const;
		float getNetValue() const;
		
		void setOutputValue(float outputValue);
		void setDesiredOutputValue(float desiredOutputValue);
				
		void computeDerativeOfErrorToNetValue();
		float getDerativeOfErrorToNetValue() const;
		float getError() const;
		
		std::string toString() const;
		

	private:

		NeuralNetwork * _network;

		std::list<ConnectionPtr> _inputNeurons;
		std::list<ConnectionPtr> _outputNeurons;

		float _netValue;
		float _outputValue;
		float _desiredOutput;
		
		float _derivativeOfErrorToNetValue;

};

} //namespace ENN
