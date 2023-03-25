#ifndef STANDALONE_MAIN_FUNCTIONS
#define STANDALONE_MAIN_FUNCTIONS

#include <iostream>
using namespace std;

//The default values used to be located in the main file
//However, these should always be declared in the headerfile
//This requires the change of the function definition in the new file with where the default should be removed
void option2(string ID_antigen, string repertoireFile, int nThreads = 1, string prefix = string(""), int startingLine = 0, int endingLine = 1000000000);



// TESTING
int testing_return();

#endif