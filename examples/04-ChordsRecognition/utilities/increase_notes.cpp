#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>

std::vector<std::string> split(std::string const & string, char token)
{
	std::vector<std::string> result;
	std::stringstream ss(string);
	std::string item;

	while (std::getline(ss, item, token))
		result.push_back(item);

	return result;
}

std::string increaseNumbers(std::string const & line)
{
	if (line.find_first_not_of(" \n") != 0 || line[0] == '#')
		return line;
	
	std::vector<std::string> lineParts = split(line, ';');
	std::vector<std::string> numbers = split(lineParts.front(), '-');
	
	std::stringstream newNumbers;
	for (unsigned i=0 ; i<numbers.size() ; i++)
	{
		newNumbers << std::stoi(numbers[i])+1;
		if (i < numbers.size()-1)
			newNumbers << "-";
	}
	newNumbers << " ;";
	newNumbers << lineParts.back();
	
	return newNumbers.str();
}

int main()
{
	//Get lines from pipe, treat them and send them to cout
	std::string input;
	bool gotInput = false;
	while (getline(std::cin, input))
	{
		std::cout << increaseNumbers(input) << std::endl;
		gotInput = true;
	}
	
	return EXIT_SUCCESS;
}
