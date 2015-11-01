#pragma once

#include "general.hpp"

#include "neuron.hpp"
#include "connection.hpp"

namespace ENN
{
	
typedef std::vector<float> 							LearningVector;
typedef std::pair<LearningVector, LearningVector> 	LearningPoint;
typedef std::vector<LearningPoint> 					LearningSet;

///This class
class NeuralNetwork
{	
	public:
	
		NeuralNetwork();
	
		void addLayer(unsigned numberOfNeurons);
		unsigned getNumberOfLayers() const;
		unsigned getNumberOfNeuronsOnLayer(unsigned layer) const;
		bool neuronExists(unsigned layer, unsigned index) const;
		Neuron * getNeuron(unsigned layer, unsigned index);
		Neuron const * const getNeuron(unsigned layer, unsigned index) const;
		std::pair<unsigned, unsigned> getNeuronPosition(Neuron const * const neuron) const;
		
		void connect(unsigned sourceLayer, unsigned sourceIndex, unsigned destinationLayer, unsigned destinationIndex);
		void setConnectionWeight(unsigned sourceLayer, unsigned sourceIndex, unsigned destinationLayer, unsigned destinationIndex, float weight);
		bool connectionExists(unsigned sourceLayer, unsigned sourceIndex, unsigned destinationLayer, unsigned destinationIndex) const;
		
		void addLearningPoint(LearningVector const & inputs, LearningVector const & outputs);
		void setLearningRate(float learningRate);
		unsigned train(Verbose verbose = Verbose::None);
		
		LearningVector process(LearningVector const & inputs);
		
		std::string toString() const;
		
		
	private:
	
		std::list< std::list<Neuron> > _neurons;
		std::list<ConnectionPtr> _connections;
		LearningSet _learningSet;
		
		void setBiasNeurons(float constantValue);
		void setInputs(LearningVector const & values);
		void setDesiredOutputs(LearningVector const & values);
		void computeOutputs();
		float getError() const;
		void computeDerivativesOfErrorToNets();
		void updateWeights();

};

} //namespace ENN
