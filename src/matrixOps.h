#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../mem/arena.h"
#ifndef MATRIXOPS
#define MATRIXOPS

double** alloc2DArr(size_t firstD, size_t secondD) {
	double** re = (double**)calloc(firstD, sizeof(double*));
	for ( size_t i = 0 ; i < firstD ; ++i )
		re[i] = (double*)calloc(secondD, sizeof(double));
	return re;	
}

void free2DArr(double** freeArr, size_t firstD) {
	for ( size_t i = 0 ; i < firstD; ++i )
		free(freeArr[i]);
	free(freeArr);	
}

double*** alloc3DArr(size_t firstD, size_t secondD, size_t thirdD) {

	double*** re = (double***)calloc(firstD, sizeof(double**));
	for ( size_t i = 0 ; i < firstD ; ++i ) {
		re[i] = alloc2DArr(secondD, thirdD);
	}		
	return re;
}		

void free3DArr(double*** freeArr, size_t firstD, size_t secondD) {
	for ( size_t i = 0 ; i < firstD ; ++i ) {
		free2DArr(freeArr[i], secondD);
	}		
	free(freeArr);
}		

double reLU(double x) {
	return ( x > 0.0 ) ? x : 0.0;
}		

double reLU_diff(double x) {
	return ( x > 0.0 ) ? 1 : 0.0;
}		

void reLU_pic( double** pic, int picRow, int picCol ) {
	for ( int i = 0 ; i < picRow; ++i ) {
		for ( int j = 0 ; j < picCol; ++j ) {
			if ( pic[i][j] < 0.000 )
				pic[i][j] = 0.000;
		}		
	}		
}		

int findMax(size_t outSize, double* prob) {
	double max = -99999;
	int index = -1;
	for ( size_t i = 0 ; i < outSize ; ++i ) {
		if (prob[i] > max) {
			index = i;
			max = prob[i];
		}		
	}		
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

void softMax(double* hid_layer_output, int n) {
    double max = hid_layer_output[0];
    double sum = 0;

    // Find the maximum value in the input array
    for (int i = 1; i < n; ++i) {
        if (hid_layer_output[i] > max) {
            max = hid_layer_output[i];
        }
    }

    // Calculate the exponentials and sum them up
    for (int i = 0; i < n; ++i) {
        hid_layer_output[i] = exp(hid_layer_output[i] - max);  // Subtract max for stability
        sum += hid_layer_output[i];
	//	printf("Exp Value[%d]: %f\n", i, hid_layer_output[i]); // Debug print
    }

    // Normalize to get probabilities
    for (int i = 0; i < n; ++i) {
        hid_layer_output[i] /= sum;
	//	printf("Softmax Output[%d]: %f\n", i, hid_layer_output[i]); // Debug print
    }
}

// ans should be either 0 or 1(0 means not the answer, 1 means is the answer)
// eg. ans = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
// predict is the probability that the machine calculate if the choice is the answer.
double crossEntropyLoss( int ansIndex, double* softmaxOut ) {
	double prob = fmax(softmaxOut[ansIndex], DBL_MIN);
	double singleSampleLoss = -1 * log(prob);
	return singleSampleLoss;
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
/*
void updateWeight( double** partial_weight, double** weight, size_t firstD, 
			       size_t secondD ) {
	for ( size_t i = 0 ; i < firstD ; ++i )
		for ( size_t j = 0 ; j < secondD ; ++j )
			weight[i][j] -= LEARNRATE * partial_weight[i][j];	
}
double gradDescent(double weight, double singleLDW) {
	return weight - ( LEARNRATE * singleLDW );
}
*/

#endif
