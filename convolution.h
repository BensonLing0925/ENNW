#include <stdio.h>
#include <stdlib.h>
#include "matrixOps.h"

typedef struct {
	int rowSize;
	int colSize;
	double** filter;
} Filter;		

// assume picture and filter are square
// stride = 1
double** convolute( double** picture, int rows, int cols,
 					double filter[][3], int f_rows, int f_cols ) {
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
// no need for now
/*
double** padding() {


}
*/
