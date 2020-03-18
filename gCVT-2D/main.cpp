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

#include "gCVT/gCVT.h"

// Input parameters
int fboSize     = 8192;
int nVertices   = 10000;

int depth       = 3;
int maxIter		= 10000;

short *inputPoints, *inputVoronoi, *outputVoronoi; 
float *density;

void generateDensity(int size, int nPoints)
{
    int tx, ty;

    for (int i = 0; i < size * size; i++) {
        tx = i % size, ty = i / size;
        float x = tx * 2.0 / size - 1.0;
        float y = ty * 2.0 / size - 1.0;

		density[i] = 1.0;
	}
}

// Deinitialization
void deinitialization()
{
    gCVTDeinitialization(); 

    free(inputPoints); 
    free(density); 
    free(inputVoronoi); 
    free(outputVoronoi); 
}

// Initialization                                                                           
void initialization()
{
    gCVTInitialization(fboSize); 

    inputPoints     = (short *) malloc(nVertices * 2 * sizeof(short));
    density         = (float *) malloc(fboSize * fboSize * sizeof(float));
    inputVoronoi    = (short *) malloc(fboSize * fboSize * 2 * sizeof(short));
    outputVoronoi   = (short *) malloc(fboSize * fboSize * 2 * sizeof(short));
}

void runTests()
{
    generateDensity(fboSize, nVertices); 

    gCVT(inputVoronoi, density, nVertices, outputVoronoi, depth, maxIter);
}

int main(int argc,char **argv)
{
	initialization();

	runTests();

	deinitialization();

    return 0;
}