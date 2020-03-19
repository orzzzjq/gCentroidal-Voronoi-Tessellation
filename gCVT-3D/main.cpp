//MIT License
//
//Copyright(c) 2020 Zheng Jiaqi @NUSComputing
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files(the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions :
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

#include <stdio.h>
#include <stdlib.h>

#include "gCVT\gCVT.h"

// Input parameters
int fboSize     = 512;
int nVertices   = 10000;

int depth       = 1;
int maxIter     = 10000;

#define TOID(x, y, z, w)    ((z) * (w) * (w) + (y) * (w) + (x))

// Global Vars
int *inputPoints, *inputVoronoi, *outputVoronoi; 
float *inputDensity;

// Initialization
void initialization()
{
    gCVTInitialization(fboSize); 

    outputVoronoi   = (int *) malloc(fboSize * fboSize * fboSize * sizeof(int));
    inputDensity    = (float *) malloc(fboSize * fboSize * fboSize * sizeof(float));
}

// Deinitialization
void deinitialization()
{
    gCVTDeinitialization(); 

    free(inputDensity);
    free(outputVoronoi); 
}

void initializeDensity()
{
    for(int i = 0; i < fboSize; ++i) {
        for(int j = 0; j < fboSize; ++j) {
            for(int k = 0; k < fboSize; ++k) {
				inputDensity[TOID(i, j, k, fboSize)] = 1.0;
            }
        }
    }
}

// Run the tests
void runTests()
{
    initializeDensity();

	gCVT(inputDensity, nVertices, outputVoronoi, depth, maxIter); 
}

int main(int argc, char **argv)
{
	initialization();

	runTests(); 
	
	deinitialization();

	return 0;
}