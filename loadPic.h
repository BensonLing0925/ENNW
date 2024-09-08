#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "structDef.h"

uint32_t littleToBigEndian(uint32_t little) {
	return ((little >> 24) & 0x000000FF) |
		   ((little >> 8)  & 0x0000FF00) |
		   ((little << 8)  & 0x00FF0000) |
		   ((little << 24) & 0xFF000000);
}
		   
void loadImgLabel(const char filePath[], DataSet* dataset) {
	FILE* fptr = fopen(filePath, "rb");
	if (!fptr) {
		printf("File does not exist\n");
		return;
	}		

	int magic, num_sample;

	fread(&magic, 4, 1, fptr);
	fread(&num_sample, 4, 1, fptr);

	magic = littleToBigEndian(magic);
	num_sample = littleToBigEndian(num_sample);

	unsigned char temp;
	for ( int i = 0 ; i < dataset->num_sample ; ++i ) {
		fread(&temp, 1, 1, fptr);	
		dataset->samples[i].answer = temp;
	}
	fclose(fptr);
}		

// if sampleCnt = -1, load all samples
void loadImgFile(const char filePath[], DataSet* dataset, int sampleCnt){
	FILE* fptr = fopen(filePath, "rb");
	if (!fptr) {
		printf("File does not exist\n");
		return;
	}		
	int magic, num_sample;
	int rows, cols;

	/*             image info            */
	fread(&magic, 4, 1, fptr);
	fread(&num_sample, 4, 1, fptr);
	fread(&rows, 4, 1, fptr);
	fread(&cols, 4, 1, fptr);

	magic = littleToBigEndian(magic);
	num_sample = littleToBigEndian(num_sample);
	rows = littleToBigEndian(rows);
	cols = littleToBigEndian(cols);

	if ( sampleCnt == -1 ) {
		dataset->num_sample = num_sample;
	}	
	else {
		dataset->num_sample = sampleCnt;
	}		

	dataset->rows = rows;
	dataset->cols = cols;

	Sample* samples = (Sample*)malloc(dataset->num_sample * sizeof(Sample));	
	dataset->samples = samples;
	/*             read image            */
	unsigned char* buffer = (unsigned char*)malloc( rows * cols );
	// i < dataset->num_sample
	// just pick one sample to see
	for ( int i = 0 ; i < dataset->num_sample ; ++i ) {
		dataset->samples[i].picture = alloc2DArr(dataset->rows, dataset->cols);
		fread(buffer, rows * cols, 1, fptr);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                dataset->samples[i].picture[r][c] = (double)((int)buffer[r * cols + c]/(double)255);
            }
        }
	}

	free(buffer);
	fclose(fptr);
}		
