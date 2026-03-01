#ifndef MATRIXOPS_H
#define MATRIXOPS_H
#include <stdlib.h>

#include <math.h>
#include <float.h>
#include <stdio.h>
#include "../mem/arena.h"

double** alloc2DArr(size_t firstD, size_t secondD);
void free2DArr(double** freeArr, size_t firstD);
double*** alloc3DArr(size_t firstD, size_t secondD, size_t thirdD);
void free3DArr(double*** freeArr, size_t firstD, size_t secondD);
double reLU(double x);
double reLU_diff(double x);
void reLU_pic( double** pic, int picRow, int picCol );
int findMax(size_t outSize, double* prob);
double* flatten(double** input, int colSize, int rowSize);
double dotProd(size_t n, double* w, double* input, double b);
double sigmoid(double x);
void softMax(double* hid_layer_output, int n);
double crossEntropyLoss( int ansIndex, double* softmaxOut );
double totalLoss(int sampleCount, double* lossArr);
double MSE( double ans, double trainedAns );
double sigmoid_diff(double z);
double LossDiffWeight( double loss, double z, double weight, double input );
void calcOutLayerErr(size_t n, double* errTerm, double* softMaxOut, double* ans);
void calcHidLayerErr(size_t outSize, double* errHidTerm, double* errOutTerm,
					 double* z, double** output_weight );
void partialInHid(size_t errHid, size_t inCnt, double* errHidTerm, 
				  double* input, double** partial_weight);
void partialHidOut(size_t errSize, size_t hiddenSize, double* errTerm, double* hiddenOut,
				 double** partial_weight);
#endif
