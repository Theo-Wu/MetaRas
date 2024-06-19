#ifndef MLPH
#define MLPH

#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

template <typename scalar_t>
__host__ __device__ __forceinline__ scalar_t mysigmoid(scalar_t x) {
    if(x<-10.0f){
        return 0.0f;
    }
	return 1.0f / (1.0f + exp(-x));
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void hardforward(scalar_t* x, int xlen) {
    for(int i=0;i<xlen;i++){
        // x[i] = (x[i] > 0)?1.0f:0.0f;
        x[i] = 1.0f / (1.0f + exp(-x[i]));
    }
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void sigmoidforward(scalar_t* x, int xlen) {
    for(int i=0;i<xlen;i++){
        x[i] = mysigmoid(x[i]);
    }
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void sigmoidbackward(scalar_t* x, scalar_t* dx, int xlen) {
    for(int i=0;i<xlen;i++){
	    dx[i] = mysigmoid(x[i])*(1.0-mysigmoid(x[i]));
    }
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void tanhforward(scalar_t* x, int xlen) {
    for(int i=0;i<xlen;i++){
        x[i] = tanh(x[i]);
    }
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void tanhbackward(scalar_t* x, scalar_t* dx, int xlen) {
    for(int i=0;i<xlen;i++){
        dx[i] = 1.0f-tanh(x[i])*tanh(x[i]);
    }
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void reluforward(scalar_t* x, int xlen) {
    for(int i=0;i<xlen;i++){
        x[i] = fmaxf(x[i],0);
    }
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void relubackward(scalar_t* x, scalar_t* dx, int xlen) {
    for(int i=0;i<xlen;i++){
	    if(x[i]>0){
            dx[i]=1.0f;
        }else{
            dx[i]=0.0f;
        }
    }
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void activationforward(scalar_t* x, int xlen, char mode) {
    if (mode == 'z')
        reluforward(x, xlen);
    else
        tanhforward(x, xlen);
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void activationbackward(scalar_t* x, scalar_t* dx, int xlen, char mode) {
    if (mode == 'z')
        relubackward(x, dx, xlen);
    else
        tanhbackward(x, dx, xlen);
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void linearforward(scalar_t* w, scalar_t* x, scalar_t* y, int wlen1, int wlen2, int xlen1, int xlen2) {
    for(int i = 0; i < wlen1; i++){
        for(int j = 0; j < xlen2; j++){
            for(int k = 0; k < wlen2; k++){
                y[i*wlen2+j] += w[i*wlen2+k] * x[k*xlen2+j];
            }
        }
    }         
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void linearbackward(scalar_t* w, scalar_t* x, scalar_t* dw, scalar_t* dx, scalar_t* dy, int wlen1, int wlen2, int xlen1, int xlen2) {
    for(int i = 0; i < wlen1; i++){
        for(int j = 0; j < wlen2; j++){
            for(int k = 0; k < xlen2; k++){
                dw[i*wlen2+j] += x[j*xlen2+k]*dy[i*xlen2+k];
            }
        }
    }
    for(int i = 0; i < xlen1; i++){
        for(int j = 0; j < xlen2; j++){
            for(int k = 0; k < wlen1; k++){
                dx[i*xlen2+j] += w[k*wlen2+i]*dy[k*xlen2+j];
            }
        }
    }
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void linearAddEqual(scalar_t* old, scalar_t* add, int len) {
    for(int i = 0; i < len; i++){
        old[i] += add[i];
    } 
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void linearMulEqual(scalar_t* old, scalar_t* add, int len) {
    for(int i = 0; i < len; i++){
        old[i] *= add[i];
    }
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void linearAdd(scalar_t* old, scalar_t* add, scalar_t* res, int len) {
    for(int i = 0; i < len; i++){
        res[i] = old[i] + add[i];
    } 
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void linearMul(scalar_t* old, scalar_t* add, scalar_t* res, int len) {
    for(int i = 0; i < len; i++){
        res[i] = old[i] * add[i];
    }
}

template <typename scalar_t>
__host__ __device__ __forceinline__ void linearDot(scalar_t* w, scalar_t* x, scalar_t* y, int wlen1, int wlen2, int xlen1, int xlen2) {
    for(int i = 0; i < wlen1; i++){
        for(int j = 0; j < xlen2; j++){
            for(int k = 0; k < wlen2; k++){
                y[i*wlen2+j] += w[i*wlen2+k] * x[k*xlen2+j];
            }
        }
    }         
}

#endif