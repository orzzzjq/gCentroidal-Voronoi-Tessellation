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

#define TOID(x, y, z, size)    ((((z) * (size)) + (y)) * (size) + (x))

// Flood along the Z axis
__global__ void kernelFloodZ(int *input, int *output, int size) 
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 
	int ty = blockIdx.y * blockDim.y + threadIdx.y; 
	int tz = 0; 

    int plane = size * size; 
    int id = TOID(tx, ty, tz, size); 
    int pixel1, pixel2; 

    pixel1 = ENCODE(0,0,0,1,0); 

    // Sweep down
    for (int i = 0; i < size; i++, id += plane) {
        pixel2 = input[id];

        if (!NOTSITE(pixel2))
            pixel1 = pixel2;

        output[id] = pixel1;
    }

	int dist1, dist2, nz;

	id -= plane + plane;

    // Sweep up
    for (int i = size - 2; i >= 0; i--, id -= plane) {
        nz = GET_Z(pixel1);
        dist1 = abs(nz - (tz + i));

        pixel2 = output[id];
        nz = GET_Z(pixel2);
        dist2 = abs(nz - (tz + i));

        if (dist2 < dist1)
            pixel1 = pixel2;

        output[id] = pixel1;
    }
}

#define LL long long
__device__ bool dominate(LL x_1, LL y_1, LL z_1, LL x_2, LL y_2, LL z_2, LL x_3, LL y_3, LL z_3, LL x_0, LL z_0)
{
	LL k_1 = y_2 - y_1, k_2 = y_3 - y_2;

	return (((y_1 + y_2) * k_1 + ((x_2 - x_1) * (x_1 + x_2 - (x_0 << 1)) + (z_2 - z_1) * (z_1 + z_2 - (z_0 << 1)))) * k_2 > \
			((y_2 + y_3) * k_2 + ((x_3 - x_2) * (x_2 + x_3 - (x_0 << 1)) + (z_3 - z_2) * (z_2 + z_3 - (z_0 << 1)))) * k_1);
}
#undef LL

__global__ void kernelMaurerAxis(int *input, int *stack, int size) 
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int tz = blockIdx.y * blockDim.y + threadIdx.y;
	int ty = 0;

	int id = TOID(tx, ty, tz, size);

    int lasty = 0;
    int x1, y1, z1, x2, y2, z2, nx, ny, nz;
    int p = ENCODE(0,0,0,1,0), s1 = ENCODE(0,0,0,1,0), s2 = ENCODE(0,0,0,1,0);
    int flag = 0;

    for (ty = 0; ty < size; ++ty, id += size) {
        p = input[id];

        if (!NOTSITE(p)) {

            while (HASNEXT(s2)) {
                DECODE(s1, x1, y1, z1);
                DECODE(s2, x2, y2, z2);
                DECODE(p, nx, ny, nz);

                if (!dominate(x1, y2, z1, x2, lasty, z2, nx, ty, nz, tx, tz))
                	break;

                lasty = y2; s2 = s1; y2 = y1;

                if (HASNEXT(s2))
                    s1 = stack[TOID(tx, y2, tz, size)];
            }

            DECODE(p, nx, ny, nz);
            s1 = s2;
            s2 = ENCODE(nx, lasty, nz, 0, flag);
            y2 = lasty;
            lasty = ty;

            stack[id] = s2;

            flag = 1;
        }
    }

    if (NOTSITE(p))
        stack[TOID(tx, ty - 1, tz, size)] = ENCODE(0, lasty, 0, 1, flag); 
}

__global__ void kernelColorAxis(int *input, int *output, int size) 
{
	__shared__ int block[BLOCKSIZE][BLOCKSIZE];

	int col = threadIdx.x;
	int tid = threadIdx.y;
	int tx = blockIdx.x * blockDim.x + col; 
	int tz = blockIdx.y;
 
    int x1, y1, z1, x2, y2, z2;
    int last1 = ENCODE(0,0,0,1,0), last2 = ENCODE(0,0,0,1,0), lasty;
    long long dx, dy, dz, best, dist;

	lasty = size - 1;

	last2 = input[TOID(tx, lasty, tz, size)]; 
	DECODE(last2, x2, y2, z2);

	if (NOTSITE(last2)) {
		lasty = y2;
		if(HASNEXT(last2)) {
			last2 = input[TOID(tx, lasty, tz, size)];
			DECODE(last2, x2, y2, z2);
		}
	}

    if (HASNEXT(last2)) {
		last1 = input[TOID(tx, y2, tz, size)];
		DECODE(last1, x1, y1, z1);
	}

	int y_start, y_end, n_step = size / blockDim.x;
	for(int step = 0; step < n_step; ++step) {
		y_start = size - step * blockDim.x - 1;
		y_end = size - (step + 1) * blockDim.x;

	    for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
	    	dx = x2 - tx; dy = lasty - ty; dz = z2 - tz;
			best = dx * dx + dy * dy + dz * dz;

			while (HASNEXT(last2)) {
				dx = x1 - tx; dy = y2 - ty; dz = z1 - tz;
				dist = dx * dx + dy * dy + dz * dz;

				if(dist > best) break;

				best = dist; lasty = y2; last2 = last1;
				DECODE(last2, x2, y2, z2);

				if (HASNEXT(last2)) {
					last1 = input[TOID(tx, y2, tz, size)];
					DECODE(last1, x1, y1, z1); 
				}
	        }

	        block[threadIdx.x][ty - y_end] = ENCODE(lasty, x2, z2, NOTSITE(last2), 0);
	    }

	    __syncthreads();

	    if(!threadIdx.y) {
	    	int id = TOID(y_end + threadIdx.x, blockIdx.x * blockDim.x, tz, size);
	    	for(int i = 0; i < blockDim.x; i++, id+=size) {
	    		output[id] = block[i][threadIdx.x];
	    	}
	    }

	    __syncthreads();
	}
}

__global__ void kernelFill(int *array, int value, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z;
	int id = TOID(tx, ty, tz, size);

	array[id] = value;
}

__global__ void kernelCalcDensity(float *last, float *output, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z;
	int id = TOID(tx, ty, tz, size);
	int lx = tx << 1, ly = ty << 1, lz = tz << 1, ls = size << 1;

	float rho = 0.0;

	rho += last[TOID(lx,     ly,     lz,     ls)];
	rho += last[TOID(lx,     ly,     lz + 1, ls)];
	rho += last[TOID(lx,     ly + 1, lz,     ls)];
	rho += last[TOID(lx,     ly + 1, lz + 1, ls)];
	rho += last[TOID(lx + 1, ly,     lz,     ls)];
	rho += last[TOID(lx + 1, ly,     lz + 1, ls)];
	rho += last[TOID(lx + 1, ly + 1, lz,     ls)];
	rho += last[TOID(lx + 1, ly + 1, lz + 1, ls)];

	output[id] = rho / 8.0;
}

__global__ void kernelZoomIn(int *input, int *output, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z;
	int id = TOID(tx, ty, tz, size);
	int lx = tx << 1, ly = ty << 1, lz = tz << 1, ls = size << 1;
	int pixel, x, y, z;

	pixel = input[id];

	if(!NOTSITE(pixel)) {
		DECODE(pixel, x, y, z);
		pixel = ENCODE(x<<1, y<<1, z<<1, 0, 0);
	}

	output[TOID(lx, ly, lz, ls)] = pixel;
}

__global__ void kernelCalcProduct(float *density, float *totalX, float *totalY, float *totalZ, float *totalW, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z;
	int id = TOID(tx, ty, tz, size);

	float rho = density[id];
	totalX[id] = rho * tx;
	totalY[id] = rho * ty;
	totalZ[id] = rho * tz;
	totalW[id] = rho;
}

// 3D -> 2D Voronoi Diagram
extern __shared__ int sharedVoro[];
__global__ void kernel2DVoronoi(int *voronoi, int *output, int size)
{
	int tid = threadIdx.x;
	int tx, ty = blockIdx.x, tz = blockIdx.y;
	int id = TOID(0, ty, tz, size);

	for(tx = tid; tx < size; tx += blockDim.x) 
		sharedVoro[tx] = MARKER;

	__syncthreads();

	int pixel, x, y, z;
	for(tx = tid; tx < size; tx += blockDim.x) {
		pixel = voronoi[id + tx];
		DECODE(pixel, x, y, z);
		sharedVoro[x] = pixel;
	}

	__syncthreads();

	for(tx = tid; tx < size; tx += blockDim.x) {
		output[id + tx] = sharedVoro[tx];
	}
}

__global__ void kernelTotalX(int *voronoi, float *totalX, float *totalY, float *totalZ, float *totalW, int size)
{
	int tid = threadIdx.x;
	int tx, ty = blockIdx.x, tz = blockIdx.y;
	int id = TOID(0, ty, tz, size), toid;
	int pixel, x, y, z;

	for(tx = tid; tx < size; tx += blockDim.x) {
		__syncthreads();
		pixel = voronoi[id + tx];
		DECODE(pixel, x, y, z);

		if(x != tx) {
			toid = id + x;
			atomicAdd(totalX + toid, totalX[id + tx]);
			atomicAdd(totalY + toid, totalY[id + tx]);
			atomicAdd(totalZ + toid, totalZ[id + tx]);
			atomicAdd(totalW + toid, totalW[id + tx]);
		}
	}
}

__global__ void kernelTotalYZ(int *voronoi, float *totalX, float *totalY, float *totalZ, float *totalW, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z;
	int id = TOID(tx, ty, tz, size), tid;

	int x, y, z;
	int pixel = voronoi[id];
	DECODE(pixel, x, y, z);

	if(NOTSITE(pixel)) return;
	if(ty == y && tz == z) return;

	tid = TOID(x, y, z, size);
	atomicAdd(totalX + tid, totalX[id]);
	atomicAdd(totalY + tid, totalY[id]);
	atomicAdd(totalZ + tid, totalZ[id]);
	atomicAdd(totalW + tid, totalW[id]);
}

__global__ void kernelUpdateCentroid(int *voronoi, int *output, float *totalX, float *totalY, float *totalZ, \
									 float *totalW, float omiga, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z;
	int id = TOID(tx, ty, tz, size);

	int x, y, z;
	int pixel = voronoi[id];
	DECODE(pixel, x, y, z);

	if(tx != x || ty != y || tz != z) return;

	float pX = totalX[id], pY = totalY[id], pZ = totalZ[id], pW = totalW[id];

	float _x = pX / pW, _y = pY / pW, _z = pZ / pW;

	x = tx * 1.0 + (_x - tx * 1.0) * omiga + 0.5;
	y = ty * 1.0 + (_y - ty * 1.0) * omiga + 0.5;
	z = tz * 1.0 + (_z - tz * 1.0) * omiga + 0.5;

	x = max(min(x, size - 1), 0);
	y = max(min(y, size - 1), 0);
	z = max(min(z, size - 1), 0);

	pixel = ENCODE(x, y, z, 0, 0);
	id = TOID(x, y, z, size);
	output[id] = pixel;
}

__global__ void kernelEnergyValue(int *voronoi, float *density, float *energy, int size)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tz = blockIdx.z;
	int id = TOID(tx, ty, tz, size);

	int x, y, z;
	int pixel = voronoi[id];
	float rho = density[id];
	DECODE(pixel, x, y, z);

	float dx = (tx - x) * 2.0 / size, dy = (ty - y) * 2.0 / size, dz = (tz - z) * 2.0 / size;

	energy[id] = rho * (dx * dx + dy * dy + dz * dz);
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
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while(i < n) { sdata[tid] += input[i] + input[i+blockSize]; i += gridSize; }
    __syncthreads();

    if(blockSize >= 512) { if(tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if(blockSize >= 256) { if(tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if(blockSize >= 128) { if(tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if(tid < 32) warpReduce<blockSize>(sdata, tid);
    if(tid == 0) output[blockIdx.x] = sdata[0];
}


