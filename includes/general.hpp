#pragma once

#include <iostream>
#include <memory>
#include <list>
#include <vector>
#include <map>
#include <utility>
#include <cmath>
#include <limits>
#include <string>
#include <sstream>

#define ENABLE_DEBUG

namespace ENN
{
	enum class Verbose { None = 0, Medium = 1, Full = 2 };
}

//Macros
extern "C"
{
	//debug < info < warning < error < fatal (error_quit)
	
	#ifdef ENABLE_DEBUG
	#define DEBUG_MSG(...)     \
	{ 	   \
		do      \
		{	   \
			std::cout     \
			<< "[DEBUG]"		\
			<< " (" << __FILE__ << ":" << __FUNCTION__ << ":" << __LINE__ << ") "	\
			<< __VA_ARGS__		\
			<< "." << std::endl;		\
		} while( 0 );     \
	}
	#else
	#define DEBUG_MSG(...)
	#endif
	
	#define INFO_MSG(...)    \
	{     \
		do     \
		{    \
			std::cout      \
			<< "[INFO]"		\
			<< " (" << __FILE__ << ":" << __FUNCTION__ << ":" << __LINE__ << ") "	\
			<< __VA_ARGS__		\
			<< "." << std::endl;		\
		} while( 0 );   \
	}

	#define WARNING_MSG(...)    \
	{     \
		do     \
		{    \
			std::cout      \
			<< "[WARNING]"		\
			<< " (" << __FILE__ << ":" << __FUNCTION__ << ":" << __LINE__ << ") "	\
			<< __VA_ARGS__		\
			<< "." << std::endl;		\
		} while( 0 );   \
	}
	
	#define ERROR_MSG(...)    \
	{     \
		do     \
		{    \
			std::cout      \
			<< "[ERROR]"		\
			<< " (" << __FILE__ << ":" << __FUNCTION__ << ":" << __LINE__ << ") "	\
			<< __VA_ARGS__		\
			<< "." << std::endl;		\
		} while( 0 );   \
	}

	#define ERROR_QUIT(...)   \
	{   \
		do    \
		{    \
			ERROR_MSG(__VA_ARGS__); \
			std::cout     \
			<< "EXITING" << std::endl;		\
			std::exit(EXIT_FAILURE);	\
		} while( 0 );   \
	} 
		
} //extern C
