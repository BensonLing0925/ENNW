#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "structDef.h"

#define BUFSIZE 2048
#define FILELEN 256
#define H_OFFSET 5
#define W_OFFSET 7

#define SOS 0xDA
#define SOF0 0xC0
#define DQT 0xDB
#define DHT 0xC4
#define SOI 0xD8
#define EOI 0xD9

const uint8_t SOS_MARKER[2] = { 0xFF, 0xDA };
const uint8_t SOF0_MARKER[2] = { 0xFF, 0xC0 };
const uint8_t DQT_MARKER[2] = { 0xFF, 0xDB };
const uint8_t DHT_MARKER[2] = { 0xFF, 0xC4 };
const uint8_t SOI_MARKER[2] = { 0xFF, 0xD8 };
const uint8_t EOI_MARKER[2] = { 0xFF, 0xD9 };

typedef enum {
	LUMINANCE = 0,	
	CHROMINANCE = 1
} Component;

typedef enum {
	DC = 0,	
	AC = 1
} TableClass;

typedef struct {
	Component component;	
	uint8_t precision;
	int dqt_len;
	union {
		uint8_t* table8;
		uint16_t* table16;
	};
} DQT_struct;

typedef struct {
	TableClass tClass;
	uint8_t precision;
	int dht_len;
	union {
		uint8_t* table8;
		uint16_t* table16;
	};
} DHT_struct;	

typedef struct {
	int pic_width;
	int pic_height;
	uint8_t*** picture;
	int num_DQT;
	int num_DHT;
	DQT_struct* DQTs;
	DHT_struct* DHTs;
} Img;	

uint32_t littleToBigEndian32(uint32_t little) {
	return ((little >> 24) & 0x000000FF) |
		   ((little >> 8)  & 0x0000FF00) |
		   ((little << 8)  & 0x00FF0000) |
		   ((little << 24) & 0xFF000000);
}

uint16_t littleToBigEndian16(uint16_t little) {
    return ((little >> 8) & 0x00FF) |
           ((little << 8) & 0xFF00);
}

uint16_t byteConcat(uint8_t byte1, uint8_t byte2) {
	return (byte1 << 8 | byte2);
}		

void readTableInfo(FILE* fptr, Img* image) {
	size_t size_read;
	uint8_t marker;
	uint8_t read_byte;
	uint16_t read_2bytes;
	int i_dqt = 0;
	int i_dht = 0;
	while ( (size_read = fread(&read_byte, 1, 1, fptr)) > 0 ) {
		if ( read_byte == 0xFF ) {
			fread(&marker, 1, 1, fptr);
			fread(&read_2bytes, 1, 2, fptr);
			read_2bytes = littleToBigEndian16(read_2bytes);
			fread(&read_byte, 1, 1, fptr);
			switch (marker) {
				case DQT:
					image->DQTs[i_dqt].dqt_len = read_2bytes - 3;
					image->DQTs[i_dqt].component = (Component)((read_byte << 4) & 0xF0);
					image->DQTs[i_dqt].precision = ((read_byte >> 4) & 0x01);
					i_dqt++;
					break;
				case DHT:		
					image->DHTs[i_dht].dht_len = read_2bytes - 3;
					image->DHTs[i_dht].tClass = (TableClass)((read_byte << 4) & 0xF0);
					image->DHTs[i_dht].precision = ((read_byte >> 4) & 0x0F);
					i_dht++;
					break;
				}
		}	
	}
	rewind(fptr);
}	

void readMarkers(FILE* fptr, Img* image) {
	size_t size_read;
	uint8_t marker = 0;
	uint8_t read_byte;
	while ( (size_read = fread(&read_byte, 1, 1, fptr)) > 0 ) {
		if ( read_byte == 0xFF ) {
			fread(&marker, 1, 1, fptr);
			switch (marker) {
				case DQT:
					image->num_DQT++;
					break;
				case DHT:
					image->num_DHT++;
					break;
			}		
		}			
	}
	rewind(fptr);
}	

void freeTable(Img* image) {
	for ( int i = 0 ; i < image->num_DQT ; ++i ) {
		if ( image->DQTs[i].precision == 0 )
			free(image->DQTs[i].table8);
		else if ( image->DQTs[i].precision == 1 )
			free(image->DQTs[i].table16);
	}	
	free(image->DQTs);

	for ( int i = 0 ; i < image->num_DHT ; ++i ) {
		if ( image->DHTs[i].precision == 0 )
			free(image->DHTs[i].table8);
		else if ( image->DHTs[i].precision == 1 )
			free(image->DHTs[i].table16);
	}	
	free(image->DHTs);
}	

void allocTable(Img* image) {
	for ( int i = 0 ; i < image->num_DQT ; ++i ) {
		if ( image->DQTs[i].precision == 0 )
			image->DQTs[i].table8 = (uint8_t*)malloc(image->DQTs[i].dqt_len * sizeof(uint8_t));
		else if ( image->DQTs[i].precision == 1 )
			image->DQTs[i].table16 = (uint16_t*)malloc(image->DQTs[i].dqt_len * sizeof(uint16_t));
		else
			printf("undefined size!\n");
	}	

	for ( int i = 0 ; i < image->num_DHT ; ++i ) {
		if ( image->DHTs[i].precision == 0 )
			image->DHTs[i].table8 = (uint8_t*)malloc(image->DHTs[i].dht_len * sizeof(uint8_t));
		else if ( image->DHTs[i].precision == 1 )
			image->DHTs[i].table16 = (uint16_t*)malloc(image->DHTs[i].dht_len * sizeof(uint16_t));
		else
			printf("undefined size!\n");
	}	
}	 

// use DQT, DHT macro to distinguish which table will be filled in
void bufferRead(FILE* fptr, Img* image, uint8_t type, int index) {
	size_t readSize;
	int precision;	
	if ( type == DQT ) {
		readSize = image->DQTs[index].dqt_len;
		precision = image->DQTs[index].precision;
		if ( precision == 0 ) fread(image->DQTs[index].table8, 1, readSize, fptr );
		else fread(image->DQTs[index].table16, readSize, 1, fptr );
	}	
	else if ( type == DHT ){
		readSize = image->DHTs[index].dht_len;
		precision = image->DQTs[index].precision;
		if ( precision == 0 ) fread(image->DHTs[index].table8, 1, readSize, fptr );
		else fread(image->DHTs[index].table16, readSize, 1, fptr );
	}	
}	

void readTables(FILE* fptr, Img* image) {
	size_t size_read;
	uint8_t marker;
	uint8_t read_byte;
	int i_dqt = 0;
	int i_dht = 0;
	while ( (size_read = fread(&read_byte, 1, 1, fptr)) > 0 ) {
		if ( read_byte == 0xFF ) {
			fread(&marker, 1, 1, fptr);
			fseek(fptr, 3, SEEK_CUR);
			switch (marker) {
				case DQT:
					bufferRead(fptr, image, DQT, i_dqt);
					i_dqt++;
					break;
				case DHT:
					bufferRead(fptr, image, DHT, i_dht);
					i_dht++;
					break;
			}	
		}	
	}
	rewind(fptr);
}	

void printDebug(Img* image) {
	printf("num_DQT: %d\n", image->num_DQT);
	printf("num_DHT: %d\n", image->num_DHT);

	for ( int i = 0 ; i < image->num_DQT ; ++i ) {
		printf("DQT LEN: %d\n", image->DQTs[i].dqt_len);
		for ( int j = 0 ; j < image->DQTs[i].dqt_len ; ++j ) {
			if (image->DQTs[i].precision == 0) {
				printf("%d ", image->DQTs[i].table8[j]);
			}
			else if (image->DQTs[i].precision == 1) {
				printf("%d ", image->DQTs[i].table16[j]);
			}
			if ( j % 8 == 7 ) printf("\n");
		}	
		printf("\n");
	}
	for ( int i = 0 ; i < image->num_DHT ; ++i ) {
		printf("DHT LEN: %d\n", image->DHTs[i].dht_len);
		for ( int j = 0 ; j < image->DHTs[i].dht_len ; ++j ) {
			if (image->DHTs[i].precision == 0) {
				printf("%d ", image->DHTs[i].table8[j]);
			}
			else if (image->DHTs[i].precision == 1) {
				printf("%d ", image->DHTs[i].table16[j]);
			}
			if ( j % 8 == 7 ) printf("\n");
		}	
		printf("\n");
	}

	printf("===========================\n");
}	

void initImg(Img* image) {
	image->num_DQT = 0;
	image->num_DHT = 0;
}	

void imageProcess(const char filePath[]) {
	FILE* fptr = fopen(filePath, "rb");
	Img image;
	initImg(&image);
	readMarkers(fptr, &image);
	image.DQTs = (DQT_struct*)malloc(image.num_DQT * sizeof(DQT_struct));
	image.DHTs = (DHT_struct*)malloc(image.num_DHT * sizeof(DHT_struct));
	readTableInfo(fptr, &image);

	allocTable(&image);
	readTables(fptr, &image);
	printDebug(&image);
	freeTable(&image);
	fclose(fptr);
}	

/*
void findTableLen(FILE* fptr, Img* image) {
	int i_dqt = 0;
	int i_dht = 0;
	size_t size_read;
	uint8_t marker = 0;
	uint16_t temp = 0;
	while ((size_read = fread(&marker, 1, 1, fptr)) > 0) {
		if ( marker == 0xFF ) {
			fread(&marker, 1, 1, fptr);
			if (marker == DQT) {
				fread(&temp, 1, 2, fptr);		
				image->imgHeader.dqt_len[i_dqt++] = littleToBigEndian16(temp);
			}		
			if (marker == DHT) {
				fread(&temp, 1, 2, fptr);		
				image->imgHeader.dht_len[i_dht++] = littleToBigEndian16(temp);
			}		
		}		
	}		
	rewind(fptr);
}
void freeTable(Img* image) {
	for ( int i = 0 ; i < image->imgHeader.num_dqt ; ++i )
		free(image->imgData.dqt_tables[i]);
	for ( int i = 0 ; i < image->imgHeader.num_dht ; ++i )
		free(image->imgData.dht_tables[i]);
	free(image->imgData.dqt_tables);	
	free(image->imgData.dht_tables);	
	free(image->imgHeader.dqt_len);
	free(image->imgHeader.dht_len);
}		

void allocTable(Img* image) {
	image->imgData.dqt_tables = (uint8_t**)malloc(image->imgHeader.num_dqt * sizeof(uint8_t*));	
	image->imgData.dht_tables = (uint8_t**)malloc(image->imgHeader.num_dht * sizeof(uint8_t*));	
	for ( int i = 0 ; i < image->imgHeader.num_dqt ; ++i ) {
		image->imgData.dqt_tables[i] = (uint8_t*)malloc(image->imgHeader.dqt_len[i] * sizeof(uint8_t));
	}		
	for ( int i = 0 ; i < image->imgHeader.num_dht ; ++i ) {
		image->imgData.dht_tables[i] = (uint8_t*)malloc(image->imgHeader.dht_len[i] * sizeof(uint8_t));
	}		
}		

void readTable(FILE* fptr, Img* image) {
	int i_dqt = 0;
	int i_dht = 0;
	size_t size_read;
	uint8_t marker = 0;
	uint8_t precision, a_byte;
	uint16_t temp = 0;
	while ((size_read = fread(&marker, 1, 1, fptr)) > 0) {
		if ( marker == 0xFF ) {
			fread(&marker, 1, 1, fptr);
			fread(&temp, 1, 2, fptr);		// read table length
			for ( int i = 0 ; i < temp ; ++i ) {
				fread(&a_byte, 1, 1, fptr);
				if ( marker == DQT ) 
					image->imgData.dqt_tables[i_dqt][i] = a_byte;
				if ( marker == DHT )
					image->imgData.dht_tables[i_dht][i] = a_byte;
			}	
			printf("\n");
			if ( marker == DQT ) i_dqt++;
			else i_dht++;
		}		
	}		
}		
void getJPEGMarker(const char filePath[]) {
	FILE* fptr = fopen(filePath, "rb");
	if (!fptr) {
		printf("File does not exist\n");
		return;
	}		

	Img image;

	int iter = 0;
	uint8_t buffer[BUFSIZE];
	uint8_t last_byte = 0;
	size_t size_read;

	rewind(fptr);

	image.imgHeader.dqt_len = (int*)malloc(image.imgHeader.num_dqt * sizeof(int));
	image.imgHeader.dht_len = (int*)malloc(image.imgHeader.num_dht * sizeof(int));
	image.imgData.tableType = (int*)malloc(image.imgHeader.num_dqt * sizeof(int));

	findTableLen(fptr, &image);

	for ( int i = 0 ; i < image.imgHeader.num_dqt ; ++i )
		printf("dqt_len: %d\n", image.imgHeader.dqt_len[i]);
	for ( int i = 0 ; i < image.imgHeader.num_dht ; ++i )
		printf("dht_len: %d\n", image.imgHeader.dht_len[i]);

	allocTable(&image);
	readTable(fptr, &image);
	freeTable(&image);

	printf("=====================================================\n");
}		
	*/	   
void loadImgLabel(const char filePath[], DataSet* dataset) {
	FILE* fptr = fopen(filePath, "rb");
	if (!fptr) {
		printf("File does not exist\n");
		return;
	}		

	int magic, num_sample;

	fread(&magic, 4, 1, fptr);
	fread(&num_sample, 4, 1, fptr);

	magic = littleToBigEndian32(magic);
	num_sample = littleToBigEndian32(num_sample);

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

	magic = littleToBigEndian32(magic);
	num_sample = littleToBigEndian32(num_sample);
	rows = littleToBigEndian32(rows);
	cols = littleToBigEndian32(cols);

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

void loadKernal(const char filePath[], Conv2D* conv) {
	

}		
