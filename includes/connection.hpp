#pragma once

#include "general.hpp"

namespace ENN
{

class Neuron;

///This class 
class Connection
{
	public:
	
		Connection(Neuron * from = nullptr, Neuron * to = nullptr); ///<constructor
		
		Neuron * getSource();
		Neuron * getDestination();
		
		void setWeight(float weight);
		float getWeight() const;
		
		void setLearningRate(float learningRate);
		void updateWeight();
		

	private:

		Neuron * _source;
		Neuron * _destination;

		float _weight;
		float _learningRate;
		static float _defaultLearningRate;

};

typedef std::shared_ptr<Connection> ConnectionPtr;

} //namespace ENN
