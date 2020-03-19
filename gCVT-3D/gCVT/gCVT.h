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

#ifndef __CUDA_H__
#define __CUDA_H__

extern "C" void gCVTInitialization(int textureSize); 

extern "C" void gCVTDeinitialization(); 

extern "C" void gCVT(float *density, int nPoints, int *output, int depth, int maxIter);

#define MARKER	    -2147483648
#define MAX_INT 	2147483647

// Sites 	 : ENCODE(x, y, z, 0, 0)
// Not sites : ENCODE(0, 0, 0, 1, 0) or MARKER
#define ENCODE(x, y, z, a, b)  (((x) << 20) | ((y) << 10) | (z) | ((a) << 31) | ((b) << 30))
#define DECODE(value, x, y, z) \
    x = ((value) >> 20) & 0x3ff; \
    y = ((value) >> 10) & 0x3ff; \
    z = (value) & 0x3ff

#define NOTSITE(value)	(((value) >> 31) & 1)
#define HASNEXT(value) 	(((value) >> 30) & 1)

#define GET_X(value)	(((value) >> 20) & 0x3ff)
#define GET_Y(value)	(((value) >> 10) & 0x3ff)
#define GET_Z(value)	((NOTSITE((value))) ? MAX_INT : ((value) & 0x3ff))

#endif
