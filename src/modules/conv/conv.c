#include <stdio.h>
#include <stdlib.h>
#include "structDef.h"
#include "conv.h"
#include "../../nn_utils/nn_utils.h"
#include "../mem/arena.h"

void initConv2D(Conv2D* conv, int num_filter, int fSize, int pSize,
				PoolingType pType) {
    conv->in_channels = 1;
	conv->num_filter = num_filter;
    conv->pooling_h = pSize;
    conv->pooling_w = pSize;
	conv->pType = pType;
    conv->stride_h = 1;
    conv->stride_w = 1;
    conv->padding_h = 0;
    conv->padding_w = 0;
    conv->has_bias = 0;
    conv->kernel_h = fSize;
    conv->kernel_w = fSize;
}		

void allocConv(Conv2D* conv, int input_rows, int input_cols) {
	conv->filters = alloc3DArr(conv->num_filter, conv->kernel_h, conv->kernel_w);
}		

// called after initConv and allocConv
void manualKernal(Conv2D* conv) {

	int num_filter = conv->num_filter;
	double kernels[10][3][3] = {
			// Horizontal edge detection
			{
				{-1, -1, -1},
				{0, 0, 0},
				{1, 1, 1}
			},
			// Vertical edge detection
			{
				{-1, 0, 1},
				{-1, 0, 1},
				{-1, 0, 1}
			},
			// Diagonal edge detection (135)
			{
				{-1, -1, 0},
				{-1, 0, 1},
				{0, 1, 1}
			},
			// Diagonal edge detection (45)
			{
				{0, -1, -1},
				{1, 0, -1},
				{1, 1, 0}
			},
			// Sharpen Filter
			{
				{0, -1, 0},
				{-1, 5, -1},
				{0, -1, 0}
			},
			// Sobel Horizontal 
			{
				{-1, -2, -1},
				{0, 0, 0},
				{1, 2, 1}
			},
			// Sobel Vertical 
			{
				{-1, 0, 1},
				{-2, 0, 2},
				{-1, 0, 1}
			},
			// Horizontal line detection
			{
				{1, 1, 1},
				{0, 0, 0},
				{-1, -1, -1}
			},
			// Vertical line detection
			{
				{1, 0, -1},
				{1, 0, -1},
				{1, 0, -1}
			},
			// 45 line detection
			{
				{-2, -1, 0},
				{-1, 0, 1},
				{0, 1, 2}
			}
	};

	for ( int filterCnt = 0 ; filterCnt < num_filter ; ++filterCnt ) {
		for ( int fRowSize = 0 ; fRowSize < conv->kernel_h ; ++fRowSize ) {  // square filter assumed
			for ( int fColSize = 0 ; fColSize < conv->kernel_w ; ++fColSize ) {  // square filter assumed
				conv->filters[filterCnt][fRowSize][fColSize] = kernels[filterCnt][fRowSize][fColSize];
			}		
		}		
	}		

}		

void freeConv(Conv2D* conv) {
	free3DArr(conv->filters, conv->num_filter, conv->kernel_h);
    free(conv);
}

// assume picture and filter are square
// stride = 1
double** convolute( double** picture, int rows, int cols,
 					double** filter, int f_rows, int f_cols ) {
	int fmapColSize = cols - f_cols + 1;	// assume kernel = 3x3, fmapColSize = 26
	int fmapRowSize = rows - f_rows + 1;
	double** re = alloc2DArr(fmapRowSize, fmapColSize);
	for ( int y = 0 ; y < fmapRowSize ; ++y ) {
		for ( int x = 0 ; x < fmapColSize ; ++x ) {
			for ( int y_filter = 0 ; y_filter < f_rows ; ++y_filter) {
				for ( int x_filter = 0 ; x_filter < f_cols ; ++x_filter) {
					re[y][x] += picture[y + y_filter][x + x_filter] * filter[y_filter][x_filter];	
				}		
			}		
		}		
	}
	return re;
}		

// assume pooling filter is a square, feature map = 26x26, pSize = 2x2
// max pooling
double** pooling( double** fmap, int fmapSize, int pSize ) {
	int pMapSize = (fmapSize / pSize);
	double** pooledMap = alloc2DArr(pMapSize, pMapSize);	
	for ( int y = 0 ; y < fmapSize ; y += 2 ) {
		for ( int x = 0 ; x < fmapSize ; x += 2 ) {
			double biggest = -99999;		
			for ( int y_pool = 0 ; y_pool < pSize ; ++y_pool ) {
				for ( int x_pool = 0 ; x_pool < pSize ; ++x_pool ) {
					if (fmap[y + y_pool][x + x_pool] > biggest) {
						biggest = fmap[y + y_pool][x + x_pool];
					}		
				}		
			}
			pooledMap[y/pSize][x/pSize] = biggest;
		}		
	}		
	return pooledMap;
}

double** padding() {


}
