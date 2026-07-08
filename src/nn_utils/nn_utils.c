#ifndef NNUTILS_H
#define NNUTILS_H
#include <stdlib.h>

#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../mem/arena.h"
#include "../ops/tensor.h"
#include "nn_utils.h"

int findMax(struct tk_tensor* t) {
    size_t n = shape_size_calc(t->shape, t->ndims);
    int index = -1;
    TK_DISPATCH_TYPES(t->dtype, "findMax", {
        scalar_t* data = (scalar_t*)t->data;
        scalar_t max = data[0];
        index = 0;
        for (size_t i = 1; i < n; ++i) {
            if (data[i] > max) { max = data[i]; index = (int)i; }
        }
    });
    return index;
}
// 5x5 for now
double* flatten(double** input, int colSize, int rowSize) {
	int i = 0;
	double* flatArr = (double*) malloc(colSize * rowSize * sizeof(double));
	for ( int y = 0 ; y < rowSize ; ++y ) {
		for( int x = 0 ; x < colSize ; ++x ) {
			flatArr[i] = input[y][x];
			i++;
		}
	}
	return flatArr;
}

// bias is added, n is the size that dose not add bias
double dotProd(size_t n, double* w, double* input, double b) {
	size_t i = 0;
	double result = 0;
	for ( i = 0 ; i < n ; ++i ) {
		result += (w[i] * input[i]);
	}
	result += b;
	return result;
}

double sigmoid(double x) {
	return (1.0/(1+exp(-x)));	
}

void softMax(struct tk_tensor* tk_output) {
    tk_ops_softmax(tk_output, tk_output);
}

// ans should be either 0 or 1(0 means not the answer, 1 means is the answer)
// eg. ans = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
// predict is the probability that the machine calculate if the choice is the answer.
double crossEntropyLoss( int ansIndex, struct tk_tensor* softmaxOut ) {
    double prob = 0.0;
    TK_DISPATCH_TYPES(softmaxOut->dtype, "crossEntropyLoss", {
        scalar_t* sfm_out = (scalar_t*)softmaxOut->data;
        prob = (double)sfm_out[ansIndex];
    });
    prob = fmax(prob, DBL_MIN);
    return -log(prob);
}

double totalLoss(int sampleCount, double* lossArr) {
	double total = 0.0;		
	for ( int i = 0 ; i < sampleCount ; ++i )
		total += lossArr[i];
	return total / (double)sampleCount;
}

// E = (1/2)*(y'-y)^2    Factor is used tto simplify expression.
double MSE( double ans, double trainedAns ) {
	return (0.5)*pow((trainedAns-ans),2);
}

double sigmoid_diff(double z) {
	double sig = sigmoid(z);
	return sig * ( 1 - sig );
}

// loss = ( y' - y )
double LossDiffWeight( double loss, double z, double weight, double input ) {
	return loss * sigmoid_diff(z) * input;
}

// calculate OutLayer error term
void calcOutLayerErr(size_t n, double* errTerm, double* softMaxOut, double* ans) {
	for ( size_t i = 0 ; i < n ; ++i ) {
		errTerm[i] = softMaxOut[i] - ans[i];
	}
}

void calcHidLayerErr(size_t outSize, double* errHidTerm, double* errOutTerm,
					 double* z, double** output_weight ) {
	for ( size_t j = 0 ; j < 2 ; ++j ) {
		double temp = 0.0; 	
		for ( size_t i = 0 ; i < outSize ; ++i ) {
			temp += errOutTerm[i] * output_weight[i][j];			
		}
		errHidTerm[j] = temp * sigmoid_diff(z[j]);
	}
}

void partialInHid(size_t errHid, size_t inCnt, double* errHidTerm, 
				  double* input, double** partial_weight) {
	for ( size_t j = 0 ; j < errHid ; ++j ) {
		for ( size_t k = 0 ; k < inCnt ; ++k ) {
			partial_weight[j][k] = errHidTerm[j] * input[k];
		}
	}
}

// weight is a precreated array that store partial derivative result
// are the weights between hidden layer and output layer
void partialHidOut(size_t errSize, size_t hiddenSize, double* errTerm, double* hiddenOut,
				 double** partial_weight) {
	for ( size_t i = 0 ; i < errSize ; ++i ) {
		for ( size_t j = 0 ; j < hiddenSize ; ++j ) {
			partial_weight[i][j] = errTerm[i] * hiddenOut[j];	
		}
	}
}
#endif
