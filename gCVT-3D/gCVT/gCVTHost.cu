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

#include <device_functions.h>
#include <cuda_runtime.h>
#include <stdio.h>
//#include <helper_timer.h>

#include "gCVT.h"

// Parameters for CUDA kernel executions
#define BLOCKX      32
#define BLOCKY      4
#define BLOCKSIZE 	32

int **cvtTextures, *cvtVoronoi, *cvtTemp;
float **cvtDensity, *cvtTotalX, *cvtTotalY, *cvtTotalZ, *cvtTotalW;
float cvtOmiga, *cvtEnergy_d, *cvtEnergyTex, cvtEnergy_h;
int cvtScale;

size_t cvtMemSize; 
int cvtBuffer; 
int cvtTexSize;

#include "gCVTKernel.h"

void gCVTInitialization(int fboSize)
{
   	cvtTexSize = fboSize; 

	cvtTextures = (int **) malloc(2 * sizeof(int *)); 
	cvtDensity = (float **) malloc(10 * sizeof(float *));

	cvtMemSize = cvtTexSize * cvtTexSize * cvtTexSize * sizeof(int); 

	cudaMalloc((void **) &cvtEnergy_d, sizeof(float));
	cudaMalloc((void **) &cvtEnergyTex, cvtTexSize * cvtTexSize * cvtTexSize * sizeof(float));

	cudaMalloc((void **) &cvtTextures[0], cvtMemSize);
	cudaMalloc((void **) &cvtTextures[1], cvtMemSize);

	cudaMalloc((void **) &cvtTotalX, cvtTexSize * cvtTexSize * cvtTexSize * sizeof(float));
	cudaMalloc((void **) &cvtTotalY, cvtTexSize * cvtTexSize * cvtTexSize * sizeof(float));
	cudaMalloc((void **) &cvtTotalZ, cvtTexSize * cvtTexSize * cvtTexSize * sizeof(float));
	cudaMalloc((void **) &cvtTotalW, cvtTexSize * cvtTexSize * cvtTexSize * sizeof(float));

	for(int i = 0; i < 10; ++i) {
	if((cvtTexSize >> i) < 64) break;
	cudaMalloc((void **) &cvtDensity[i], (cvtTexSize>>i) * (cvtTexSize>>i) * (cvtTexSize>>i) * sizeof(float));
	}
}

// Deallocate all allocated memory
void gCVTDeinitialization()
{
	cudaFree(cvtTextures[0]);
	cudaFree(cvtTextures[1]);
	cudaFree(cvtTotalX);
	cudaFree(cvtTotalY);
	cudaFree(cvtTotalZ);
	cudaFree(cvtTotalW);
	cudaFree(cvtEnergyTex);
	cudaFree(cvtEnergy_d);

	for(int i = 0; i < 3; ++i) {
		if((cvtTexSize >> i) < 64) break;
		cudaFree(cvtDensity[i]);
	}

	free(cvtDensity);
	free(cvtTextures); 
}

// Random Point Generator
// Random number generator, obtained from http://oldmill.uchicago.edu/~wilder/Code/random/
unsigned long z, w, jsr, jcong; // Seeds
void randinit(unsigned long x_) 
{ z =x_; w = x_; jsr = x_; jcong = x_; }
unsigned long znew() 
{ return (z = 36969 * (z & 0xfffful) + (z >> 16)); }
unsigned long wnew() 
{ return (w = 18000 * (w & 0xfffful) + (w >> 16)); }
unsigned long MWC()  
{ return ((znew() << 16) + wnew()); }
unsigned long SHR3()
{ jsr ^= (jsr << 17); jsr ^= (jsr >> 13); return (jsr ^= (jsr << 5)); }
unsigned long CONG() 
{ return (jcong = 69069 * jcong + 1234567); }
unsigned long rand_int()         // [0,2^32-1]
{ return ((MWC() ^ CONG()) + SHR3()); }
double random()     // [0,1)
{ return ((double) rand_int() / (double(ULONG_MAX)+1)); }

void generateRandomPoints(int *hostTexture, int nPoints)
{	
	cvtTexSize <<= 1;

    int tx, ty, tz, id; 

    randinit(0);

    for (int i = 0; i < cvtTexSize * cvtTexSize * cvtTexSize; i++)
        hostTexture[i] = MARKER; 

	for (int i = 0; i < nPoints; i++)
	{
        do { 
            tx = int(random() * cvtTexSize); 
            ty = int(random() * cvtTexSize); 
            tz = int(random() * cvtTexSize); 
            id = TOID(tx, ty, tz, cvtTexSize); 
		} while (hostTexture[id] != MARKER); 

        hostTexture[id] = ENCODE(tx, ty, tz, 0, 0); 
    }

    cudaMemcpy(cvtTextures[cvtBuffer], hostTexture, cvtTexSize * cvtTexSize * cvtTexSize * sizeof(int), cudaMemcpyHostToDevice);

    cvtTexSize >>= 1;
}

// Copy input to GPU 
void gCVTInitializeInput(float *density)
{
    cudaMemcpy(cvtDensity[0], density, cvtTexSize * cvtTexSize * cvtTexSize * sizeof(float), cudaMemcpyHostToDevice);

	cvtBuffer = 0;
}

void pba3DColorZAxis() 
{
   	dim3 block = dim3(BLOCKX, BLOCKY); 
    dim3 grid = dim3(cvtTexSize / block.x, cvtTexSize / block.y); 

    kernelFloodZ<<< grid, block >>>(cvtTextures[cvtBuffer], cvtTextures[1^cvtBuffer], cvtTexSize); 
    cvtBuffer = 1^cvtBuffer; 
}

void pba3DComputeProximatePointsYAxis() 
{
	dim3 block = dim3(BLOCKX, BLOCKY); 
    dim3 grid = dim3(cvtTexSize / block.x, cvtTexSize / block.y);

    kernelMaurerAxis<<< grid, block >>>(cvtTextures[cvtBuffer], cvtTextures[1^cvtBuffer], cvtTexSize);
}

// Phase 3 of PBA. m3 must divides texture size
// This method color along the Y axis
void pba3DColorYAxis() 
{
	dim3 block = dim3(BLOCKSIZE, 2); 
    dim3 grid = dim3(cvtTexSize / block.x, cvtTexSize); 

    kernelColorAxis<<< grid, block >>>(cvtTextures[1^cvtBuffer], cvtTextures[cvtBuffer], cvtTexSize);
}

void pba3DCompute()
{
	pba3DColorZAxis();

	pba3DComputeProximatePointsYAxis();

	pba3DColorYAxis();

	pba3DComputeProximatePointsYAxis();

	pba3DColorYAxis();

	cvtVoronoi = cvtTextures[cvtBuffer];
	cvtTemp    = cvtTextures[1^cvtBuffer];
}

void gCVTCalcDensity()
{
	dim3 block(BLOCKX, BLOCKY);

	for(int i = 1; i < 3; ++i) {
		int size = cvtTexSize >> i;
		dim3 grid(size / block.x, size / block.y, size);
		kernelCalcDensity<<< grid, block >>>(cvtDensity[i - 1], cvtDensity[i], size);
	}
}

void gCVTZoomIn()
{
	dim3 block(BLOCKX, BLOCKY);
	dim3 grid((cvtTexSize<<1) / block.x, (cvtTexSize<<1) / block.y, (cvtTexSize<<1));

	kernelFill<<< grid, block >>>(cvtTemp, MARKER, cvtTexSize<<1);

	grid = dim3(cvtTexSize / block.x, cvtTexSize / block.y, cvtTexSize);
	kernelZoomIn<<< grid, block >>>(cvtVoronoi, cvtTemp, cvtTexSize);

	int *tmp_ptr = cvtVoronoi;
	cvtVoronoi = cvtTemp;
	cvtTemp = tmp_ptr;
	cvtBuffer = 1^cvtBuffer;
}

void gCVTComputeCentroid()
{
	dim3 block(BLOCKX, BLOCKY);
	dim3 grid(cvtTexSize / block.x, cvtTexSize / block.y, cvtTexSize);

	kernelCalcProduct<<< grid, block >>>(cvtDensity[cvtScale], cvtTotalX, cvtTotalY, cvtTotalZ, cvtTotalW, cvtTexSize);

	block = dim3(BLOCKSIZE);
	grid = dim3(cvtTexSize, cvtTexSize);

	int ns = cvtTexSize * 4;
	kernel2DVoronoi<<< grid, block, ns >>>(cvtVoronoi, cvtTemp, cvtTexSize);

	kernelTotalX<<< grid, block >>>(cvtVoronoi, cvtTotalX, cvtTotalY, cvtTotalZ, cvtTotalW, cvtTexSize);

	block = dim3(BLOCKX, BLOCKY);
	grid = dim3(cvtTexSize / block.x, cvtTexSize / block.y, cvtTexSize);
	kernelTotalYZ<<< grid, block >>>(cvtTemp, cvtTotalX, cvtTotalY, cvtTotalZ, cvtTotalW, cvtTexSize);
}

void gCVTUpdateCentroid()
{
	dim3 block(BLOCKX, BLOCKY);
	dim3 grid(cvtTexSize / block.x, cvtTexSize / block.y, cvtTexSize);

	kernelFill<<< grid, block >>>(cvtTemp, MARKER, cvtTexSize);

	kernelUpdateCentroid<<< grid, block >>>(cvtVoronoi, cvtTemp, cvtTotalX, cvtTotalY, cvtTotalZ, \
											cvtTotalW, cvtOmiga, cvtTexSize);

	int *tmp_ptr = cvtVoronoi;
	cvtVoronoi = cvtTemp;
	cvtTemp = tmp_ptr;
	cvtBuffer = 1^cvtBuffer;
}

float gCVTEnergy()
{
	dim3 block(BLOCKX, BLOCKY);
	dim3 grid(cvtTexSize / block.x, cvtTexSize / block.y, cvtTexSize);

	kernelEnergyValue<<< grid, block >>>(cvtVoronoi, cvtDensity[cvtScale], cvtEnergyTex, cvtTexSize);

    const int blockSize = 512;
    int n = cvtTexSize * cvtTexSize * cvtTexSize;
    int blocksPerGrid;

    do {
        blocksPerGrid = min(int(std::ceil((1.*n) / blockSize)), 32768);
        kernelReduce<blockSize><<< blocksPerGrid, blockSize >>>(cvtEnergyTex, cvtEnergyTex, n);
        n = blocksPerGrid;
    } while (n > blockSize);

    if (n > 1) {
        kernelReduce<blockSize><<< 1, blockSize >>>(cvtEnergyTex, cvtEnergyTex, n);
    }

    cudaMemcpy(&cvtEnergy_h, cvtEnergyTex, sizeof(float), cudaMemcpyDeviceToHost);

    return cvtEnergy_h * powf(8, cvtScale);
}


int gCVTIteration;

void gCVT(float *density, int nPoints, int *output, int depth, int maxIter) 
{
	for(int i = 0; i < depth; ++i) if((cvtTexSize >> i) < 64) { depth = i; break; }

	gCVTInitializeInput(density);

	gCVTCalcDensity();

	cvtTexSize >>= depth;

	generateRandomPoints(output, nPoints);

	float Energy, lastEnergy = 1e18, diffEnergy, gradEnergy;

	cvtOmiga = 2.0;
	gCVTIteration = 0;

	for(cvtScale = depth - 1; ~cvtScale; --cvtScale) {
		cvtTexSize <<= 1;

 		do {
 			gCVTIteration++;

 		    pba3DCompute();

 			if(gCVTIteration % 10 == 0) {
 				Energy = gCVTEnergy();
 			}

 		    gCVTComputeCentroid();

 		    gCVTUpdateCentroid();

 		    if(gCVTIteration % 10 == 0) {
 		    	diffEnergy = lastEnergy - Energy;
 		    	gradEnergy = diffEnergy / 10.0;

 		    	cvtOmiga = min(2.0, 1.0 + 1e-4 * diffEnergy);

 		    	if(cvtScale) {
 		    		if(gradEnergy < 5) break;
 		    	} else {
 			    	if(gradEnergy < 1e-4) break;
 		    	} 
 		    	lastEnergy = Energy;
 		    }

 		} while(gCVTIteration < maxIter);

		if(cvtScale) gCVTZoomIn();
	}

	pba3DCompute();

    cudaMemcpy(output, cvtVoronoi, cvtMemSize, cudaMemcpyDeviceToHost); 
}

