#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>

#include "enn.hpp"

//#define DEBUG_PARSING

//Useful global variables
std::vector<std::string> chordRoots {"C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"};
std::vector<std::string> chordCompositions {"7", "maj7", "m7", "m7b5"}; //We could add: "dim7", "7sus4", "7b9", "9", "6", "M", "m", "dim", "sus4", "#9", "#11"...
const unsigned numberOfOutputNeurons = chordRoots.size() + chordCompositions.size();
const unsigned numberOfInputNeurons = 24;
const unsigned numberOfNeuronsPerHiddenLayer = 32;
const unsigned numberOfHiddenLayers = 1;
const std::string learningSetFileName("learning_set");
const float learningRate = 0.01;

const float LOW = -0.5f;
const float HIGH = 0.5f;

//Parsing functions for the learning set
std::vector<std::string> split(std::string const & string, char token)
{
	std::vector<std::string> result;
	std::stringstream ss(string);
	std::string item;

	while (std::getline(ss, item, token))
		result.push_back(item);

	return result;
}

ENN::LearningVector parseNotesIntoInput(std::string const & notes)
{
	ENN::LearningVector inputs(numberOfInputNeurons);
	
	for (float & input : inputs)
		input = LOW;
	
	std::vector<std::string> splitNotes = split(notes, '-');
	for (std::string const & note : splitNotes)
	{
		const unsigned noteNumber = std::stoi(note);
		inputs[noteNumber] = HIGH;
	}
	
	#ifdef DEBUG_PARSING
	std::cout << "Converted notes '" << notes << "' to : ";
	for (float input : inputs)
		std::cout << input << " ";
	std::cout << std::endl;
	#endif
	
	return inputs;
}	
	
ENN::LearningVector parseChordsIntoOutput(std::string const & chord)
{
	ENN::LearningVector outputs(numberOfOutputNeurons);
	
	for (float & output : outputs)
		output = LOW;
	
	std::vector<std::string> splitChord = split(chord, '_');
	
	auto rootIterator = std::find(chordRoots.begin(), chordRoots.end(), splitChord.front());
	auto compIterator = std::find(chordCompositions.begin(), chordCompositions.end(), splitChord.back());

	const unsigned rootNumber = std::distance(chordRoots.begin(), rootIterator);
	const unsigned compNumber = std::distance(chordCompositions.begin(), compIterator);
	
	outputs[rootNumber] = HIGH;
	outputs[chordRoots.size() + compNumber] = HIGH;
	
	#ifdef DEBUG_PARSING
	std::cout << "Converted chord '" << chord << "' to : ";
	for (float output : outputs)
		std::cout << output << " ";
	std::cout << std::endl;
	#endif
	
	return outputs;
}
	
ENN::LearningPoint parseLineIntoLearningPoint(std::string const & line)
{
	if (line.find_first_not_of(" \n") != 0) //the line starts with a space or a newline character
	{
		#ifdef DEBUG_PARSING
		std::cout << "The line '" << line << "' is empty." << std::endl;
		#endif
		return ENN::LearningPoint();
	}
	
	if (line[0] == '#') //comment
	{
		#ifdef DEBUG_PARSING
		std::cout << "The line '" << line << "' is a comment." << std::endl;
		#endif
		return ENN::LearningPoint();
	}
	
	//Remove spaces
	std::string lineWithoutSpaces;
	for (unsigned i=0 ; i<line.size() ; i++)
	{
		if (line[i] != ' ')
		{
			lineWithoutSpaces.push_back(line[i]);
		}
	}
	
	#ifdef DEBUG_PARSING
	std::cout << "Changed line '" << line << "' into '" << lineWithoutSpaces << "'." << std::endl;
	#endif
	
	//Split the two parts of the line	
	std::vector<std::string> lineParts = split(lineWithoutSpaces, ';');
	
	//Parse the parts and return the learning point
	return std::make_pair(parseNotesIntoInput(lineParts[0]),  parseChordsIntoOutput(lineParts[1]));
}

ENN::LearningSet parseFileIntoLearningSet(std::string const & fileName)
{
	std::ifstream file(fileName);
	
	if (!file.is_open())
	{
		std::cerr << "Cannot open file " << fileName << "\n";
		exit(EXIT_FAILURE);
	}
	
	ENN::LearningSet result;
	std::string line;
	
	while (std::getline(file,line))
	{
		ENN::LearningPoint point = parseLineIntoLearningPoint(line);
		
		if (!point.first.empty() && !point.second.empty())
		{
			result.push_back(point);
		}
	}
	
	std::cout << "Sucessfully parsed learning set." << std::endl;
	return result;
}

//Reverse parsing operation for chord name
std::pair<std::string, float> convertOutputVectorToChordName(ENN::LearningVector const & outputs)
{
	std::stringstream chord;
	
	auto highestRootOutput = std::max_element(outputs.begin(), outputs.begin()+chordRoots.size());
	chord << chordRoots[std::distance(outputs.begin(), highestRootOutput)];
	const float rootCertainty = (*highestRootOutput - LOW) / (HIGH - LOW);
	
	chord << "_";
	
	auto highestCompOutput = std::max_element(outputs.begin()+chordRoots.size(), outputs.end());
	chord << chordCompositions[std::distance(outputs.begin()+chordRoots.size(), highestCompOutput)];
	const float compCertainty = (*highestCompOutput - LOW) / (HIGH - LOW);
	
	float globalCertainty = (rootCertainty + compCertainty)/2.f;
	if (globalCertainty > 1.f)
		globalCertainty = 1.f;
	
	return std::make_pair(chord.str(), globalCertainty);
}
	
//Program
int main()
{
	srand(static_cast<unsigned>(time(0)));
	std::cout << "ENNlib example n4 : chords recognition." << std::endl;
	std::cout << "We'll feed notes to a neural network and get the corresponding chord name." << std::endl;
	std::cout << std::endl;
	std::cout << "The question is: how can we represent notes and chords for a network? Everything else is just basic supervised training with gradient descent and backprop." << std::endl;
	std::cout << "Processing symbolic knowledge is a rather vast question for NNs. To keep things simple, here is what we will do:" << std::endl;
	std::cout << " - The input layer will consist of 24 neurons representing 2 octaves (for example if we want to send C0 and C1 we'll set neurons 0 and 12 to 0.5 and the others to -0.5)." << std::endl;
	std::cout << " - The output layer is a bit more tricky because there are many possible chords existing (Cmaj7, Cm7, Cm7b5, Cdim7, D7, E...). Let's break this in two components: the chord's root (12 possible notes) and the chord's \"composition\" (7, maj7, m7...). There will be 12 output neurons for each root and one neuron for each possible composition." << std::endl;
	
	
	//Create the network
	std::cout << std::endl;
	std::cout << "Let's first create the network: 24 input neurons, " << numberOfHiddenLayers << " hidden layers of " << numberOfNeuronsPerHiddenLayer << " neurons (completely arbitrary) and the output layer. But how many neurons for the output layer?" << std::endl;
	
	std::cout << "First of all, there are " << chordRoots.size() << " notes for chord names: " << std::flush;
	for (unsigned i=0 ; i<chordRoots.size() ; i++)
	{
		std::cout << chordRoots[i] << std::flush;
		if (i < chordRoots.size()-1)
			std::cout << ", " << std::flush;
	}
	std::cout << std::endl;
	
	std::cout << "Then, there are " << chordCompositions.size() << " compositions for chord names: " << std::flush;
	for (unsigned i=0 ; i<chordCompositions.size() ; i++)
	{
		std::cout << chordCompositions[i] << std::flush;
		if (i < chordCompositions.size()-1)
			std::cout << ", " << std::flush;
	}
	std::cout << std::endl;
	
	std::cout << "Hence, there are " << numberOfOutputNeurons << " neurons on the output layer." << std::endl;
		  
	ENN::NeuralNetwork nn;
	
	nn.addLayer(24); //input layer
	for (unsigned i=0 ; i<numberOfHiddenLayers ; i++) //hidden layers
		nn.addLayer(numberOfNeuronsPerHiddenLayer);
	nn.addLayer(numberOfOutputNeurons); //output layer
	
	std::cout << "To keep things simple, we'll just connect every neuron to all the neurons of the next layer (typical feed forward neuron network)." << std::endl;
	
	nn.connectAllLayers();
	
	//Create the training set
	std::cout << std::endl;
	std::cout << "In order to train the network, we need a set of corresponding notes and chord names. The file 'learning_set' contains a few \"learning points\"." << std::endl;
	std::cout << "Each line represents a learning point. The first part of the line describes the notes with the format X-X-X where X is the index of the note (0-4-7-10 is C E G Bb)." << std::endl;
	std::cout << "The second part of the line (separated with ';') indicates the chord name with the format R_Comp where R is the root and Comp the composition." << std::endl;
	std::cout << "The file will be parsed and used to train the neural network (lines starting with '#' are ignored)." << std::endl;
	
	std::cout << std::endl;
	std::cout << "Parsing file..." << std::endl;
	
	ENN::LearningSet learningSet = parseFileIntoLearningSet(learningSetFileName);
	nn.appendLearningSet(learningSet);
	
	//Let's train the neuron network
	std::cout << "The learning rate is set to " << learningRate << "." << std::endl;
	
	do 
	{
		std::cout << "Press enter to train the algorithm (be patient)..." << std::flush;
	} while (std::cin.get() != '\n');
	
	std::cout << "Starting algorithm..." << std::endl;
	
	
	nn.setLearningRate(learningRate);
	unsigned cycles = nn.train(ENN::Verbose::Medium);
	
	std::cout << "The neural network stabilized after " << cycles << " training cycles." << std::endl;
		
	//Let's see what this network can do
	std::cout << std::endl;
	std::cout << "Now let's feed some notes to the network and see if it recognizes the chords." << std::endl;
	
	std::pair<std::string, float> outputChord;
		
	//Test 1
	std::cout << std::endl;
	std::cout << "(1) Let's start by something very basic: C_7. We'll feed 0-4-7-10 to the network." << std::endl;
	do 
	{
		std::cout << "Press enter to see the result..." << std::flush;
	} while (std::cin.get() != '\n');
	
	outputChord = convertOutputVectorToChordName(nn.process(parseNotesIntoInput(std::string("0-4-7-10"))));
	
	std::cout << "The network outputs: '" << outputChord.first << "' with " << outputChord.second*100.f << "% of certainty. ";
	
	if (outputChord.first == std::string("C_7"))
		std::cout << "Great!";
	else
		std::cout << "The result isn't good. That's really bad news for the whole network. Maybe we need a larger learning set.";
	std::cout << std::endl;
	
	//Test 2
	std::cout << std::endl;
	std::cout << "(2) Now let's try a reversal of C_7. We'll feed 10-12-16-19 to the network." << std::endl;
	do 
	{
		std::cout << "Press enter to see the result..." << std::flush;
	} while (std::cin.get() != '\n');
	
	outputChord = convertOutputVectorToChordName(nn.process(parseNotesIntoInput(std::string("10-12-16-19"))));
	
	std::cout << "The network outputs: '" << outputChord.first << "' with " << outputChord.second*100.f << "% of certainty. ";
	
	if (outputChord.first == std::string("C_7"))
		std::cout << "Great again.";
	else
		std::cout << "That's not good. The network should not fail on a value used for its learning. Maybe we need a larger learning set.";
	std::cout << std::endl;
		
	//Test 3
	std::cout << std::endl;
	std::cout << "(3) We will now feed the network with an unknown reversal of C_7 (0-7-10-16)." << std::endl;
	do 
	{
		std::cout << "Press enter to see the result..." << std::flush;
	} while (std::cin.get() != '\n');
	
	outputChord = convertOutputVectorToChordName(nn.process(parseNotesIntoInput(std::string("0-7-10-16"))));
	
	std::cout << "The network outputs: '" << outputChord.first << "' with " << outputChord.second*100.f << "% of certainty. ";
	
	if (outputChord.first == std::string("C_7"))
		std::cout << "Awesome! The network just recognized a chord from a note pattern he didn't know. We can fairly say he \"learned\" something.";
	else
		std::cout << "Too bad. The network is not satisfying. Possible workarounds are to add more layers and hidden neurons and increase the learning set.";
	std::cout << std::endl;
	
	//Test 4
	std::cout << std::endl;
	std::cout << "(4) Alright. Now let's try to make the network recognize a chord he has never seen (Eb_7) with a special reversal: 1-3-7-10" << std::endl;
	do 
	{
		std::cout << "Press enter to see the result..." << std::flush;
	} while (std::cin.get() != '\n');
	
	outputChord = convertOutputVectorToChordName(nn.process(parseNotesIntoInput(std::string("1-3-7-10"))));
	
	std::cout << "The network outputs: '" << outputChord.first << "' with " << outputChord.second*100.f << "% of certainty. ";
	
	if (outputChord.first == std::string("Eb_7"))
		std::cout << "Yeah!! The network just recognized a chord he had never seen. That's a damn clever network we got here (just kidding).";
	else
		std::cout << "Well. Maybe we were too hopeful. The problem we address might need a more sophisticated type of neural network.";
	std::cout << std::endl;
	
	//Test 5
	/*std::cout << std::endl;
	std::cout << "(5) Final test. Let's see what the network does with 0-4-7-9. This is C_6... but also A_m7. We'll run this 100 times and see how many of each we get." << std::endl;
	do 
	{
		std::cout << "Press enter to see the result..." << std::flush;
	} while (std::cin.get() != '\n');
	
	std::map<std::string, unsigned> outputChords;
	
	for (unsigned i=0 ; i<100 ; i++)
	{
		outputChords[convertOutputVectorToChordName(nn.process(parseNotesIntoInput(std::string("2-5-7-11")))).first]++;
	}
		
	std::cout << "We obtain: " << std::endl;
	
	for (auto it = outputChords.begin() ; it != outputChords.end() ; it++)
	{
		std::cout << " - " << it->second << " of " << it->first << std::endl;
	}*/
	
	//User tests
	std::cout << std::endl;
	std::cout << "Now if you'd like to, you can enter some notes and see which chord returns the neural network." << std::endl;
	std::cout << "Enter a string with the format x-x-x to feed the network and q to quit." << std::endl;
	
	std::string userInput("");
	while (userInput != std::string("q"))
	{
		if (!userInput.empty())
		{
			std::pair<std::string, float> res = convertOutputVectorToChordName(nn.process(parseNotesIntoInput(userInput)));
			std::cout << "The network returns " << res.first << " with " << res.second*100.f << "% of certainty." << std::endl;
		}
		
		std::cout << std::endl;
		std::cout << "Type something here: " << std::flush;
		std::cin >> userInput;
	}
	
	std::cout << "Byebye" << std::endl;
	
	return EXIT_SUCCESS;
}
