#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "structDef.h"


void writeData(PlotInfo plot) {
	FILE* fptr = fopen(plot.filePath, "w");

	fprintf(fptr, "# Title: %s\n", plot.title);
	fprintf(fptr, "# Xlabel: %s\n", plot.xlabel);
	fprintf(fptr, "# Ylabel: %s\n", plot.ylabel);
	fprintf(fptr, "# X Data - Y Data\n");

	fprintf(fptr, "$DataBlock1 << EOD\n");
	for ( size_t i = 0 ; i < plot.xdata.dataSize ; ++i ) {
		switch(plot.xdata.type) {
			case INT_TYPE:
				fprintf(fptr, "%d ", plot.xdata.intData[i]);
				break;
			case FLOAT_TYPE:
				fprintf(fptr, "%f ", plot.xdata.floatData[i]);
				break;
			case DOUBLE_TYPE:
				fprintf(fptr, "%lf ", plot.xdata.doubleData[i]);
				break;
		}		

		switch(plot.ydata.type) {
			case INT_TYPE:
				fprintf(fptr, "%d\n", plot.ydata.intData[i]);
				break;
			case FLOAT_TYPE:
				fprintf(fptr, "%f\n", plot.ydata.floatData[i]);
				break;
			case DOUBLE_TYPE:
				fprintf(fptr, "%lf\n", plot.ydata.doubleData[i]);
				break;
		}		
	}
	
	fprintf(fptr, "EOD\n");

	fprintf(fptr, "reset\n");
	fprintf(fptr, "set xlabel \"%s\"\n", plot.xlabel);
	fprintf(fptr, "set ylabel \"%s\"\n", plot.ylabel);
	fprintf(fptr, "set title \"%s\"\n", plot.title);
	fprintf(fptr, "set style line 1 lc rgb \'#f2b50f\' lt 1 lw 2 pt 7\n");
	fprintf(fptr, "plot $DataBlock1 using 1:2 with linespoints linestyle 1\n");

	fclose(fptr);
}		

void initPlot(const char filePath[], const char xlabel[], 
			  const char ylabel[],   const char title[],
			  PlotInfo* plot) {
	
	plot->filePath = (char*)malloc(strlen(filePath)+1);
	plot->xlabel = (char*)malloc(strlen(xlabel)+1);
	plot->ylabel = (char*)malloc(strlen(ylabel)+1);
	plot->title = (char*)malloc(strlen(title)+1);

	strcpy(plot->filePath, filePath);
	strcpy(plot->xlabel, xlabel);
	strcpy(plot->ylabel, ylabel);
	strcpy(plot->title, title);

}		

void freePlt(PlotInfo* plot) {
	free(plot->filePath);
	free(plot->xlabel);
	free(plot->ylabel);
	free(plot->title);
	free(plot);
}		
