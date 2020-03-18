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

__global__ void kernelFillShort(short2* arr, short value, int texSize) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    arr[__mul24(y, texSize) + x] = make_short2(value, value); 
}

__global__ void kernelFloodDown(short2 *input, short2 *output, int size, int bandSize) 
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * bandSize; 
    int id = TOID(tx, ty, size); 

    short2 pixel1, pixel2; 

    pixel1 = make_short2(MARKER, MARKER); 

    for (int i = 0; i < bandSize; i++, id += size) {
        pixel2 = input[id]; 

        if (pixel2.x != MARKER) 
            pixel1 = pixel2; 

        output[id] = pixel1; 
    }
}

__global__ void kernelFloodUp(short2 *input, short2 *output, int size, int bandSize) 
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = (blockIdx.y+1) * bandSize - 1; 
    int id = TOID(tx, ty, size); 

    short2 pixel1, pixel2; 
    int dist1, dist2; 

    pixel1 = make_short2(MARKER, MARKER); 

    for (int i = 0; i < bandSize; i++, id -= size) {
        dist1 = abs(pixel1.y - ty + i); 

        pixel2 = input[id]; 
        dist2 = abs(pixel2.y - ty + i); 

        if (dist2 < dist1) 
            pixel1 = pixel2; 

        output[id] = pixel1; 
    }
}

__global__ void kernelPropagateInterband(short2 *input, short2 *output, int size, int bandSize) 
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int inc = bandSize * size; 
    int ny, nid, nDist; 
    short2 pixel; 

    // Top row, look backward
    int ty = blockIdx.y * bandSize; 
    int topId = TOID(tx, ty, size); 
    int bottomId = TOID(tx, ty + bandSize - 1, size); 
    int tid = blockIdx.y * size + tx;
    int bid = tid + (size * size / bandSize);

    pixel = input[topId]; 
    int myDist = abs(pixel.y - ty); 
    output[tid] = pixel;

    for (nid = bottomId - inc; nid >= 0; nid -= inc) {
        pixel = input[nid]; 

        if (pixel.x != MARKER) { 
            nDist = abs(pixel.y - ty); 

            if (nDist < myDist)
                output[tid] = pixel;

            break;  
        }
    }

    // Last row, look downward
    ty = ty + bandSize - 1; 
    pixel = input[bottomId]; 
    myDist = abs(pixel.y - ty); 
    output[bid] = pixel;

    for (ny = ty + 1, nid = topId + inc; ny < size; ny += bandSize, nid += inc) {
        pixel = input[nid]; 

        if (pixel.x != MARKER) { 
            nDist = abs(pixel.y - ty); 

            if (nDist < myDist)
                output[bid] = pixel;

            break; 
        }
    }
}

__global__ void kernelUpdateVertical(short2 *color, short2 *margin, short2 *output, int size, int bandSize) 
{
    __shared__ short2 block[BLOCKSIZE][BLOCKSIZE];

    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * bandSize; 

    short2 top = margin[blockIdx.y * size + tx]; 
    short2 bottom = margin[(blockIdx.y + size / bandSize) * size + tx]; 
    short2 pixel; 

    int dist, myDist; 

    int id = TOID(tx, ty, size); 

    int n_step = bandSize / blockDim.x;
    for(int step = 0; step < n_step; ++step) {
        int y_start = blockIdx.y * bandSize + step * blockDim.x;
        int y_end = y_start + blockDim.x;

        for (ty = y_start; ty < y_end; ++ty, id += size) {
            pixel = color[id]; 
            myDist = abs(pixel.y - ty); 

            dist = abs(top.y - ty);
            if (dist < myDist) { myDist = dist; pixel = top; }

            dist = abs(bottom.y - ty); 
            if (dist < myDist) pixel = bottom; 

            block[threadIdx.x][ty - y_start] = make_short2(pixel.y, pixel.x);
        }

        __syncthreads();

        int tid = TOID(blockIdx.y * bandSize + step * blockDim.x + threadIdx.x, \
                        blockIdx.x * blockDim.x, size);

        for(int i = 0; i < blockDim.x; ++i, tid += size) {
            output[tid] = block[i][threadIdx.x];
        }

        __syncthreads();
    }
}

#define LL long long
__device__ bool dominate(LL x1, LL y1, LL x2, LL y2, LL x3, LL y3, LL x0)
{
    LL k1 = y2 - y1, k2 = y3 - y2;
    return (k1 * (y1 + y2) + (x2 - x1) * ((x1 + x2) - (x0 << 1))) * k2 > \
            (k2 * (y2 + y3) + (x3 - x2) * ((x2 + x3) - (x0 << 1))) * k1;
}
#undef LL

__global__ void kernelProximatePoints(short2 *input, short2 *stack, int size, int bandSize) 
{
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; 
    int ty = __mul24(blockIdx.y, bandSize); 
    int id = TOID(tx, ty, size); 
    int lasty = -1; 
    short2 last1, last2, current; 

    last1.y = -1; last2.y = -1; 

    for (int i = 0; i < bandSize; i++, id += size) {
        current = input[id];

        if (current.x != MARKER) {
            while (last2.y >= 0) {
                if (!dominate(last1.x, last2.y, last2.x, \
                    lasty, current.x, current.y, tx))
                    break;

                lasty = last2.y; last2 = last1; 

                if (last1.y >= 0)
                    last1 = stack[TOID(tx, last1.y, size)]; 
            }

            last1 = last2; last2 = make_short2(current.x, lasty); lasty = current.y; 

            stack[id] = last2;
        }
    }

    // Store the pointer to the tail at the last pixel of this band
    if (lasty != ty + bandSize - 1) 
        stack[TOID(tx, ty + bandSize - 1, size)] = make_short2(MARKER, lasty); 
}

__global__ void kernelCreateForwardPointers(short2 *input, short2 *output, int size, int bandSize) 
{
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; 
    int ty = __mul24(blockIdx.y+1, bandSize) - 1; 
    int id = TOID(tx, ty, size); 
    int lasty = -1, nexty; 
    short2 current; 

    // Get the tail pointer
    current = input[id]; 

    if (current.x == MARKER)
        nexty = current.y; 
    else
        nexty = ty; 

    for (int i = 0; i < bandSize; i++, id -= size)
        if (ty - i == nexty) {
            current = make_short2(lasty, input[id].y);
            output[id] = current; 

            lasty = nexty; 
            nexty = current.y; 
        }

    // Store the pointer to the head at the first pixel of this band
    if (lasty != ty - bandSize + 1) 
        output[id + size] = make_short2(lasty, MARKER);  
}

__global__ void kernelMergeBands(short2 *color, short2 *link, short2 *output, int size, int bandSize)
{
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; 
    int band1 = blockIdx.y * 2; 
    int band2 = band1 + 1; 
    int firsty, lasty; 
    short2 last1, last2, current; 
    // last1 and last2: x component store the x coordinate of the site, 
    // y component store the backward pointer
    // current: y component store the x coordinate of the site, 
    // x component store the forward pointer

    // Get the two last items of the first list
    lasty = __mul24(band2, bandSize) - 1; 
    last2 = make_short2(color[TOID(tx, lasty, size)].x, 
        link[TOID(tx, lasty, size)].y); 

    if (last2.x == MARKER) {
        lasty = last2.y; 

        if (lasty >= 0) 
            last2 = make_short2(color[TOID(tx, lasty, size)].x, 
            link[TOID(tx, lasty, size)].y); 
        else
            last2 = make_short2(MARKER, MARKER); 
    }

    if (last2.y >= 0) {
        // Second item at the top of the stack
        last1 = make_short2(color[TOID(tx, last2.y, size)].x, 
            link[TOID(tx, last2.y, size)].y); 
    }

    // Get the first item of the second band
    firsty = __mul24(band2, bandSize); 
    current = make_short2(link[TOID(tx, firsty, size)].x, 
        color[TOID(tx, firsty, size)].x); 

    if (current.y == MARKER) {
        firsty = current.x; 

        if (firsty >= 0) 
            current = make_short2(link[TOID(tx, firsty, size)].x, 
            color[TOID(tx, firsty, size)].x); 
        else
            current = make_short2(MARKER, MARKER); 
    }

    // Count the number of item in the second band that survive so far. 
    // Once it reaches 2, we can stop. 
    int top = 0; 

    while (top < 2 && current.y >= 0) {
        // While there's still something on the left
        while (last2.y >= 0) {

            if (!dominate(last1.x, last2.y, last2.x, \
                lasty, current.y, firsty, tx)) 
                break; 

            lasty = last2.y; last2 = last1; 
            top--; 

            if (last1.y >= 0) 
                last1 = make_short2(color[TOID(tx, last1.y, size)].x, 
                link[TOID(tx, last1.y, size)].y); 
        }

        // Update the current pointer 
        output[TOID(tx, firsty, size)] = make_short2(current.x, lasty); 

        if (lasty >= 0) 
            output[TOID(tx, lasty, size)] = make_short2(firsty, last2.y); 

        last1 = last2; last2 = make_short2(current.y, lasty); lasty = firsty; 
        firsty = current.x; 

        top = max(1, top + 1); 

        // Advance the current pointer to the next one
        if (firsty >= 0) 
            current = make_short2(link[TOID(tx, firsty, size)].x, 
            color[TOID(tx, firsty, size)].x); 
        else
            current = make_short2(MARKER, MARKER); 
    }

    // Update the head and tail pointer. 
    firsty = __mul24(band1, bandSize); 
    lasty = __mul24(band2, bandSize); 
    current = link[TOID(tx, firsty, size)]; 

    if (current.y == MARKER && current.x < 0) { // No head?
        last1 = link[TOID(tx, lasty, size)]; 

        if (last1.y == MARKER)
            current.x = last1.x; 
        else
            current.x = lasty; 

        output[TOID(tx, firsty, size)] = current; 
    }

    firsty = __mul24(band1, bandSize) + bandSize - 1; 
    lasty = __mul24(band2, bandSize) + bandSize - 1; 
    current = link[TOID(tx, lasty, size)]; 

    if (current.x == MARKER && current.y < 0) { // No tail?
        last1 = link[TOID(tx, firsty, size)]; 

        if (last1.x == MARKER) 
            current.y = last1.y; 
        else
            current.y = firsty; 

        output[TOID(tx, lasty, size)] = current; 
    }
}

__global__ void kernelDoubleToSingleList(short2 *color, short2 *link, short2 *output, int size)
{
    int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; 
    int ty = blockIdx.y; 
    int id = TOID(tx, ty, size); 

    output[id] = make_short2(color[id].x, link[id].y); 
}

__global__ void kernelColor(short2 *input, short2 *output, int size) 
{
    __shared__ short2 block[BLOCKSIZE][BLOCKSIZE];

    int col = threadIdx.x; 
    int tid = threadIdx.y; 
    int tx = __mul24(blockIdx.x, blockDim.x) + col; 
    int dx, dy, lasty; 
    unsigned int best, dist; 
    short2 last1, last2; 

    lasty = size - 1; 

    last2 = input[TOID(tx, lasty, size)];

    if (last2.x == MARKER) {
        lasty = last2.y; 
        last2 = input[TOID(tx, lasty, size)];
    }

    if (last2.y >= 0) 
        last1 = input[TOID(tx, last2.y, size)];

    int y_start, y_end, n_step = size / blockDim.x;
    for(int step = 0; step < n_step; ++step) {
        y_start = size - step * blockDim.x - 1;
        y_end = size - (step + 1) * blockDim.x;

        for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
            dx = last2.x - tx; dy = lasty - ty; 
            best = dist = __mul24(dx, dx) + __mul24(dy, dy); 

            while (last2.y >= 0) {
                dx = last1.x - tx; dy = last2.y - ty; 
                dist = __mul24(dx, dx) + __mul24(dy, dy); 

                if (dist > best) 
                    break; 

                best = dist; lasty = last2.y; last2 = last1;

                if (last2.y >= 0) 
                    last1 = input[TOID(tx, last2.y, size)];
            }

            block[threadIdx.x][ty - y_end] = make_short2(lasty, last2.x);
        }

        __syncthreads();

        int iinc = size * blockDim.y;
        int id = TOID(y_end + threadIdx.x, blockIdx.x * blockDim.x + tid, size);
        for(int i = tid; i < blockDim.x; i += blockDim.y, id += iinc) {
            output[id] = block[i][threadIdx.x];
        }

        __syncthreads();
    }
}

__global__ void kernelZoomIn(short2 *input, short2 *output, int size, int scale)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int id = TOID(tx, ty, size);
    int tid = TOID(tx<<scale, ty<<scale, size<<scale);

    short2 pixel = input[id];

    output[tid] = (pixel.x == MARKER) ? make_short2(MARKER, MARKER) : make_short2(pixel.x<<scale, pixel.y<<scale);
}

__global__ void kernelDensityScaling(float *input, float *output, int size)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    float density = 0;

    for(int x = (tx << 1); x < (tx << 1) + 2; ++x) {
        for(int y = (ty << 1); y < (ty << 1) + 2; ++y) {
            density += input[TOID(x, y, size<<1)];
        }
    }

    output[TOID(tx, ty, size)] = density / 4.0;
}

// compute the prefix sum of weight, x*weight and y*weight for each row
extern __shared__ float tmpScan[]; 
__global__ void kernelComputeWeightedPrefixX(float *prefixW, float *prefixX, float *prefixY, float *density, int texWidth)
{
    float *tmpX = tmpScan; 
    float *tmpY = tmpX + blockDim.x; 
    float *tmpWeight = tmpY + blockDim.x; 

    float pW, pX, pY; 

    int tid = threadIdx.x; 
    int tx, ty = blockIdx.x; 
    float lastX = 0.0f, lastY = 0.0f, lastW = 0.0f; 
    int id = __mul24(ty, texWidth); 

    for (int xx = 0; xx < texWidth; xx += blockDim.x) {
        tx = xx + tid; 
        pW = density[id + tx]; 
        pX = lastX + tx * pW; 
        pY = lastY + ty * pW;
        pW = lastW + pW; 
        tmpWeight[tid] = pW; tmpX[tid] = pX; tmpY[tid] = pY; 
        __syncthreads(); 

        for (int step = 1; step < blockDim.x; step *= 2) { // parallel prefix sum within a block
            if (tid >= step) {
                pW += tmpWeight[tid - step]; 
                pX += tmpX[tid - step]; 
                pY += tmpY[tid - step]; 
            }

            __syncthreads(); 
            tmpWeight[tid] = pW; tmpX[tid] = pX; tmpY[tid] = pY; 
            __syncthreads(); 
        }
        
        prefixX[id + tx] = tmpX[tid]; 
        prefixY[id + tx] = tmpY[tid]; 
        prefixW[id + tx] = tmpWeight[tid]; 

        if (tid == 0) {
            lastX = tmpX[blockDim.x-1]; 
            lastY = tmpY[blockDim.x-1]; 
            lastW = tmpWeight[blockDim.x-1]; 
        }
        __syncthreads(); 
    }
}

// 2D -> 1D Voronoi Diagram
extern __shared__ short sharedVor[]; 
__global__ void kernelVoronoi1D(short2 *input, int *output, int texWidth) 
{
    int tid = threadIdx.x; 
    int tx, ty = blockIdx.x; 
    int id = __mul24(ty, texWidth); 

    // Initialize
    for (tx = tid; tx < texWidth; tx += blockDim.x)
        sharedVor[tx] = MARKER; 

    __syncthreads(); 
    
    // Mark
    for (tx = tid; tx < texWidth; tx += blockDim.x) { 
        short2 pixel = input[id + tx]; 

        sharedVor[pixel.x] = pixel.y; 
    }

    __syncthreads(); 

    // Write
    id /= 2; 
    for (tx = tid; tx < texWidth / 2; tx += blockDim.x)
         output[id + tx] = ((int *)sharedVor)[tx]; 
}

__global__ void kernelTotal_X(short2 *voronoi, float *prefixX, float *prefixY, float *prefixW,\
                              float *totalX, float *totalY, float *totalW, int texWidth) 
{
	// Shared array to store the sums
	__shared__ float sharedTotalX[BAND];  // BAND = 256
	__shared__ float sharedTotalY[BAND]; 
	__shared__ float sharedTotalW[BAND]; 
	__shared__ int startBlk[100], endBlk[100];	// 100 blocks is more than enough

	int count; 
    int tid = threadIdx.x; 
    int tx, ty = blockIdx.x, offset; 
    int id = __mul24(ty, texWidth); 
    short2 me, other; 

	int margin = tid * BAND; 

	if (margin < texWidth) {
		startBlk[tid] = 0; 
		endBlk[tid] = texWidth; 

		for (tx = 0; tx < texWidth; tx += blockDim.x) {
			me = voronoi[id + tx]; 

			if (me.x >= margin) {
				startBlk[tid] = max(0, tx - int(blockDim.x)); 
				break; 
			}
		}

		for (; tx < texWidth; tx += blockDim.x) {
			me = voronoi[id + tx]; 

			if (me.x >= margin + BAND) {
				endBlk[tid] = tx; 
				break; 
			}
		}
	}

	__syncthreads(); 
    
	count = 0; 

	// We process one BAND at a time. 
	for (margin = 0; margin < texWidth; margin += BAND, count++) {
		// Only for the first iteration of tx
		// Make sure we detect the boundary at tx = 0
		other.x = -1;

	    // Left edge, scan through the row
		for (tx = startBlk[count] + tid; tx < endBlk[count]; tx += blockDim.x) { 
			if (tx > 0) 
	            other = voronoi[id + tx - 1]; 

		    me = voronoi[id + tx]; 
			offset = me.x - margin; 

            // margin <= me.x < margin + BAND  &&  the closest site of the previous pixel is different
			if (offset >= 0 && offset < BAND && other.x < me.x) {
				if (tx > 0) {
					sharedTotalX[offset] = prefixX[id + tx - 1];
					sharedTotalY[offset] = prefixY[id + tx - 1];
					sharedTotalW[offset] = prefixW[id + tx - 1];
				} else {
					sharedTotalX[offset] = 0.0f;
					sharedTotalY[offset] = 0.0f;
					sharedTotalW[offset] = 0.0f;
				}
			}
		}

		__syncthreads(); 

		// Right edge
		for (tx = startBlk[count] + tid; tx < endBlk[count]; tx += blockDim.x) { 
			me = voronoi[id + tx]; 
			offset = me.x - margin; 

			if (tx < texWidth - 1)
				other = voronoi[id + tx + 1]; 
			else
				other.x = texWidth;

            // margin <= me.x < margin + BAND  &&  the closest site of the next pixel is different
			if (offset >= 0 && offset < BAND && me.x < other.x) {
				sharedTotalX[offset] = prefixX[id + tx] - sharedTotalX[offset]; 
				sharedTotalY[offset] = prefixY[id + tx] - sharedTotalY[offset]; 
				sharedTotalW[offset] = prefixW[id + tx] - sharedTotalW[offset]; 
			}
		}

	    __syncthreads(); 

		// Write
		for (tx = tid; tx < BAND; tx += blockDim.x) 
			if (margin + tx < texWidth) {
				totalX[id + margin + tx] = sharedTotalX[tx]; 
				totalY[id + margin + tx] = sharedTotalY[tx]; 
				totalW[id + margin + tx] = sharedTotalW[tx]; 
			}
	}
}

__global__ void kernelScan_Y(short *voronoi, float *totalX, float *totalY, float *totalW, int size)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * BLOCKSIZE;
    int id = TOID(tx, ty, size), tid;
    short pixel, last = MARKER;
    float tmpX = 0.0, tmpY = 0.0, tmpW = 0.0;

    for(int i = 0; i < BLOCKSIZE; ++i, ++ty, id += size) {
        __syncthreads();
        pixel = voronoi[id];

        if(pixel != last) {
            if(last != MARKER) {
                tid = TOID(tx, last, size);
                atomicAdd(totalX + tid, tmpX);
                atomicAdd(totalY + tid, tmpY);
                atomicAdd(totalW + tid, tmpW);
            }
            tmpX = tmpY = tmpW = 0.0;
            last = pixel;
        }

        if(pixel != MARKER && pixel != ty) {
            tmpX += totalX[id];
            tmpY += totalY[id];
            tmpW += totalW[id];
        }
    }

    if(last != MARKER) {
        tid = TOID(tx, last, size);
        atomicAdd(totalX + tid, tmpX);
        atomicAdd(totalY + tid, tmpY);
        atomicAdd(totalW + tid, tmpW);
    }
}

__global__ void kernelUpdateSites(short *voronoi, float *totalX, float *totalY, float *totalW,
                                  short2 *output, int size, float omega)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 
	int ty = blockIdx.y * blockDim.y + threadIdx.y; 

	float pX, pY, pW; 

    int id = TOID(tx, ty, size); 
    short seed = voronoi[id]; 

    if (seed != ty) return ; 

	pX = totalX[id]; 
	pY = totalY[id]; 
	pW = totalW[id]; 

    float _x = pX / pW, _y = pY / pW;

    short2 rc = make_short2( tx + (_x - tx) * omega + 0.5f, ty + (_y - ty) * omega + 0.5f );

    rc.x = max(min(rc.x, size-1), 0); 
    rc.y = max(min(rc.y, size-1), 0); 

    id = TOID(rc.x, rc.y, size); 
    output[id] = rc;
}

__global__ void kernelCalcEnergy(short2 *voronoi, float *density, float *nrgTex, int size)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int id = TOID(tx, ty, size);

    short2 site = voronoi[id];

    float dx = (site.x - tx) * 1.0f / size;
    float dy = (site.y - ty) * 1.0f / size;

    float dist = dx * dx + dy * dy;

    nrgTex[id] = density[id] * dist;
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if(blockSize >= 16) sdata[tid] += sdata[tid +  8];
    if(blockSize >=  8) sdata[tid] += sdata[tid +  4];
    if(blockSize >=  4) sdata[tid] += sdata[tid +  2];
    if(blockSize >=  2) sdata[tid] += sdata[tid +  1];
}

template <unsigned int blockSize>
__global__ void kernelReduce(float *input, float *output, unsigned int n)
{
    __shared__ float sdata[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;

    while(i < n) { sdata[tid] += input[i] + input[i+blockSize]; i += gridSize; }
    __syncthreads();

    if(blockSize >= 512) { if(tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if(blockSize >= 256) { if(tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if(blockSize >= 128) { if(tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if(tid < 32) warpReduce<blockSize>(sdata, tid);
    if(tid == 0) output[blockIdx.x] = sdata[0];
}
