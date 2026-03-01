#ifndef CONV_H
#define CONV_H

#include <stdio.h>
#include <stdlib.h>
#include "structDef.h"
#include "../mem/arena.h"

void initConv2D(Conv2D* conv, int num_filter, int fSize, int pSize,
				PoolingType pType);
void allocConv(Conv2D* conv, int input_rows, int input_cols);
void manualKernal(Conv2D* conv);
void freeConv(Conv2D* conv);
double** convolute( double** picture, int rows, int cols,
 					double** filter, int f_rows, int f_cols );
double** pooling( double** fmap, int fmapSize, int pSize );

#endif
