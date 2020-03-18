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
#include <cuda_runtime.h>
#include <device_functions.h>
#include <unordered_map>
#include <vector>
//#include <helper_timer.h>

#include "gCVT.h"

#define BLOCKX		16
#define BLOCKY		16
#define BLOCKSIZE	64
#define BAND		256			// For simplicity, just assume we never need to work with a smaller texture.
#define THRESHOLD	1e-5

// Global Variables
short2 **cvtTextures, *cvtMargin;
short2 *cvtVoronoi, *cvtTemp;
float **cvtDensity;
float **cvtPrefixX, **cvtPrefixY, **cvtPrefixW;
float *cvtTotalX, *cvtTotalY, *cvtTotalW;
float *cvtEnergyTex, cvtEnergy_h;
float cvtOmega;

int cvtScale;
int cvtBuffer;              // Current buffer
int cvtMemSize;             // Size (in bytes) of a texture
int cvtTexSize;             // Texture size (squared texture)

#include "gCVTKernel.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// Initialization ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

std::unordered_map<int,int> m_1, m_2, m_3, m_4;

void gCVTInitialization(int textureSize)
{
    m_1[256]  = 4,  m_2[256]  = 32, m_3[256]  = 16;
    m_1[512]  = 8,  m_2[512]  = 32, m_3[512]  = 16;
    m_1[1024] = 16, m_2[1024] = 32, m_3[1024] = 16;
    m_1[2048] = 32, m_2[2048] = 32, m_3[2048] = 8;
    m_1[4096] = 64, m_2[4096] = 32, m_3[4096] = 8;
    m_1[8192] = 128,m_2[8192] = 32, m_3[8192] = 4;

    cvtTexSize  = textureSize;

    cvtMemSize  = cvtTexSize * cvtTexSize * sizeof(short2);

    cvtTextures = (short2 **) malloc(2 * sizeof(short2 *));
    cvtDensity  = (float  **) malloc(10 * sizeof(float  *));
    cvtPrefixX  = (float  **) malloc(10 * sizeof(float  *));
    cvtPrefixY  = (float  **) malloc(10 * sizeof(float  *));
    cvtPrefixW  = (float  **) malloc(10 * sizeof(float  *));

    cudaMalloc((void **) &cvtTextures[0],   cvtMemSize);
    cudaMalloc((void **) &cvtTextures[1],   cvtMemSize);
    cudaMalloc((void **) &cvtMargin,        m_1[cvtTexSize] * cvtTexSize * sizeof(short2));
    cudaMalloc((void **) &cvtTotalX,        cvtTexSize * cvtTexSize * sizeof(float)); 
    cudaMalloc((void **) &cvtTotalY,        cvtTexSize * cvtTexSize * sizeof(float)); 
    cudaMalloc((void **) &cvtTotalW,        cvtTexSize * cvtTexSize * sizeof(float)); 
    cudaMalloc((void **) &cvtEnergyTex,     cvtTexSize * cvtTexSize * sizeof(float));

    for(int i = 0; i < 10; ++i) {
    if((cvtTexSize>>i) < 256) break;
    cudaMalloc((void **) &cvtDensity[i],    (cvtTexSize>>i) * (cvtTexSize>>i) * sizeof(float));
    cudaMalloc((void **) &cvtPrefixX[i],    (cvtTexSize>>i) * (cvtTexSize>>i) * sizeof(float)); 
    cudaMalloc((void **) &cvtPrefixY[i],    (cvtTexSize>>i) * (cvtTexSize>>i) * sizeof(float)); 
    cudaMalloc((void **) &cvtPrefixW[i],    (cvtTexSize>>i) * (cvtTexSize>>i) * sizeof(float)); 
    }
    // gpuErrchk(cudaPeekAtLastError());gpuErrchk(cudaDeviceSynchronize());
}

// Deallocate all allocated memory
void gCVTDeinitialization()
{
    cudaFree(cvtTextures[0]);
    cudaFree(cvtTextures[1]);
    cudaFree(cvtMargin);

    for(int i = 0; i < 10; ++i) {
    if((cvtTexSize>>i) < 256) break;
    cudaFree(cvtDensity[i]);
    cudaFree(cvtPrefixX[i]);
    cudaFree(cvtPrefixY[i]);
    cudaFree(cvtPrefixW[i]);
    }
    
    cudaFree(cvtTotalX);
    cudaFree(cvtTotalY);
    cudaFree(cvtTotalW);
    cudaFree(cvtEnergyTex);

    free(cvtTextures);
    free(cvtDensity);
    free(cvtPrefixX);
    free(cvtPrefixY);
    free(cvtPrefixW);
}

// Copy input to GPU 
void gCVTInitializeInput(float *density)
{
    cudaMemcpy(cvtDensity[0], density, cvtTexSize * cvtTexSize * sizeof(float), cudaMemcpyHostToDevice); 

    cvtVoronoi = cvtTextures[0];
    cvtTemp    = cvtTextures[1];
    cvtBuffer  = 0;
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

void gCVTInitSites(short *input, int num)
{
    cvtTexSize <<= 1;

    randinit(0);
    int tx, ty; 
    for (int i = 0; i < cvtTexSize * cvtTexSize; i++)
        input[i * 2] = input[i * 2 + 1] = MARKER;

    for (int i = 0; i < num; i++) {
        do { 
            tx = random() * cvtTexSize; 
            ty = random() * cvtTexSize; 
        } while (input[(ty * cvtTexSize + tx) * 2] != MARKER); 
        input[(ty * cvtTexSize + tx) * 2    ] = tx; 
        input[(ty * cvtTexSize + tx) * 2 + 1] = ty; 
    }

    cudaMemcpy(cvtTextures[0], input, cvtTexSize * cvtTexSize * sizeof(short2), cudaMemcpyHostToDevice);

    cvtTexSize >>= 1;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Parallel Banding Algorithm plus //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

// Phase 1 of PBA. m1 must divides texture size and equal or less than size / 64
void pba2DPhase1(int m1) 
{
    dim3 block = dim3(BLOCKSIZE);   
    dim3 grid = dim3(cvtTexSize / block.x, m1); 

    kernelFloodDown<<< grid, block >>>(cvtTextures[cvtBuffer], cvtTextures[cvtBuffer], cvtTexSize, cvtTexSize / m1); 

    kernelFloodUp<<< grid, block >>>(cvtTextures[cvtBuffer], cvtTextures[cvtBuffer], cvtTexSize, cvtTexSize / m1); 

    kernelPropagateInterband<<< grid, block >>>(cvtTextures[cvtBuffer], cvtMargin, cvtTexSize, cvtTexSize / m1);

    kernelUpdateVertical<<< grid, block >>>(cvtTextures[cvtBuffer], cvtMargin, cvtTextures[1^cvtBuffer], cvtTexSize, cvtTexSize / m1);
}

// Phase 2 of PBA. m2 must divides texture size
void pba2DPhase2(int m2) 
{
    // Compute proximate points locally in each band
    dim3 block = dim3(BLOCKSIZE);
    dim3 grid = dim3(cvtTexSize / block.x, m2);

    kernelProximatePoints<<< grid, block >>>(cvtTextures[1^cvtBuffer], cvtTextures[cvtBuffer], cvtTexSize, cvtTexSize / m2); 

    kernelCreateForwardPointers<<< grid, block >>>(cvtTextures[cvtBuffer], cvtTextures[cvtBuffer], cvtTexSize, cvtTexSize / m2); 

    // Repeatly merging two bands into one
    for (int noBand = m2; noBand > 1; noBand /= 2) {
        grid = dim3(cvtTexSize / block.x, noBand / 2); 
        kernelMergeBands<<< grid, block >>>(cvtTextures[1^cvtBuffer], cvtTextures[cvtBuffer], cvtTextures[cvtBuffer], cvtTexSize, cvtTexSize / noBand); 
    }

    // Replace the forward link with the X coordinate of the seed to remove
    // the need of looking at the other texture. We need it for coloring.
    grid = dim3(cvtTexSize / block.x, cvtTexSize); 
    kernelDoubleToSingleList<<< grid, block >>>(cvtTextures[1^cvtBuffer], cvtTextures[cvtBuffer], cvtTextures[cvtBuffer], cvtTexSize); 
}

// Phase 3 of PBA. m3 must divides texture size and equal or less than 64
void pba2DPhase3(int m3) 
{
    dim3 block = dim3(BLOCKSIZE, m3); 
    dim3 grid = dim3(cvtTexSize / block.x);
    
    kernelColor<<< grid, block >>>(cvtTextures[cvtBuffer], cvtTextures[1^cvtBuffer], cvtTexSize); 
}

void pba2DCompute(int m1, int m2, int m3)
{
    pba2DPhase1(m1);  

    pba2DPhase2(m2); 

    pba2DPhase3(m3); 

    cvtVoronoi = cvtTextures[1^cvtBuffer]; 
    cvtTemp = cvtTextures[cvtBuffer]; 
    cvtBuffer = 1^cvtBuffer;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Centroidal Voronoi Tessellation ////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

void gCVTDensityScaling(int k)
{
    dim3 block(BLOCKX, BLOCKY);

    for(int i = 1; i < k; ++i) {
        dim3 grid((cvtTexSize >> i) / block.x, (cvtTexSize >> i) / block.y);
        kernelDensityScaling<<< grid, block >>>(cvtDensity[i - 1], cvtDensity[i], cvtTexSize >> i);
    }
}

void gCVTComputeWeightedPrefix(int k) 
{
    dim3 block(BLOCKSIZE); 

    int ns = BLOCKSIZE * 3 * sizeof(float); 
    for(int i = 0; i < k; ++i) {
        dim3 grid(cvtTexSize >> i);
        kernelComputeWeightedPrefixX<<< grid, block, ns >>>(cvtPrefixW[i], cvtPrefixX[i], cvtPrefixY[i], cvtDensity[i], cvtTexSize >> i); 
    }
}

void gCVTComputeCentroid()
{
	dim3 block(BLOCKSIZE);
    dim3 grid(cvtTexSize);

    int ns = cvtTexSize * sizeof(short); 

    kernelVoronoi1D<<< grid, block, ns >>>(cvtVoronoi, (int *) cvtTemp, cvtTexSize); 

	kernelTotal_X<<< grid, block >>>(cvtVoronoi, cvtPrefixX[cvtScale], cvtPrefixY[cvtScale], cvtPrefixW[cvtScale], \
                                     cvtTotalX, cvtTotalY, cvtTotalW, cvtTexSize); 

    block = dim3(BLOCKSIZE); 
    grid = dim3(cvtTexSize / block.x, cvtTexSize / block.x);
	kernelScan_Y<<< grid, block >>>((short *) cvtTemp, cvtTotalX, cvtTotalY, cvtTotalW, cvtTexSize);
}

void gCVTUpdateSites() 
{
    dim3 block(BLOCKX, BLOCKY); 
    dim3 grid(cvtTexSize / block.x, cvtTexSize / block.y); 

    kernelFillShort<<< grid, block >>>(cvtVoronoi, MARKER, cvtTexSize);

	kernelUpdateSites<<< grid, block >>>((short *) cvtTemp, cvtTotalX, cvtTotalY, cvtTotalW, \
                                         cvtVoronoi, cvtTexSize, cvtOmega);
}

void gCVTZoomIn()
{
    dim3 block(BLOCKX, BLOCKY);
    dim3 grid1(cvtTexSize / block.x, cvtTexSize / block.y);
    dim3 grid2((cvtTexSize << 1) / block.x, (cvtTexSize << 1) / block.y);

    kernelFillShort<<< grid2, block >>>(cvtTemp, MARKER, cvtTexSize << 1);

    kernelZoomIn<<< grid1, block >>>(cvtVoronoi, cvtTemp, cvtTexSize, 1);

    cvtBuffer = 1^cvtBuffer;

    short2 *tmp_ptr = cvtVoronoi;
    cvtVoronoi = cvtTemp;
    cvtTemp = tmp_ptr;
}

float gCVTCalcEnergy()
{
    dim3 block(BLOCKX, BLOCKY);
    dim3 grid(cvtTexSize / block.x, cvtTexSize / block.y);

    kernelCalcEnergy<<< grid, block >>>(cvtVoronoi, cvtDensity[cvtScale], cvtEnergyTex, cvtTexSize);

    const int blockSize = 512;
    int n = cvtTexSize * cvtTexSize;
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

    return cvtEnergy_h * powf(2.0, cvtScale * 2.0);
}


int gCVTIterations; 

void gCVT(short *input, float *density, int noSites, short *output, int depth, int maxIter)
{
    for(int i = 0; i < depth; ++i) if((cvtTexSize >> i) < 256) { depth = i; break; }

    gCVTInitializeInput(density);

    gCVTDensityScaling(depth);

	gCVTComputeWeightedPrefix(depth); 

    cvtScale = 0;

    gCVTIterations = 0; 

    cvtTexSize >>= depth;

    gCVTInitSites(input, noSites);

    float Energy, lastEnergy = 1e18, diffEnergy, gradientEnergy;

    cvtOmega = 2.0;

    for(cvtScale = depth - 1; ~cvtScale; --cvtScale) {
        cvtTexSize <<= 1;
        do { 
            pba2DCompute(m_1[cvtTexSize], m_2[cvtTexSize], m_3[cvtTexSize]);
            
            if(gCVTIterations % 10 == 0)
                Energy = gCVTCalcEnergy();
    
            gCVTComputeCentroid(); 

            gCVTUpdateSites();

            gCVTIterations++; 

            if(gCVTIterations % 10 == 0) {
                diffEnergy = lastEnergy - Energy;
                gradientEnergy = diffEnergy / 10.;
				cvtOmega = min(2.0, 1.0 + diffEnergy);

                if(cvtScale) {
                    if(gradientEnergy < 3e-1) break;
                } else {
                    if(gradientEnergy < THRESHOLD) break;
                } 

				lastEnergy = Energy;
			}

        } while (gCVTIterations < maxIter);

        if(cvtScale) gCVTZoomIn();
    }

    pba2DCompute(m_1[cvtTexSize], m_2[cvtTexSize], m_3[cvtTexSize]); 

    cudaMemcpy(output, cvtVoronoi, cvtMemSize, cudaMemcpyDeviceToHost); 
}

