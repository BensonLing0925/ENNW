#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "structDef.h"
#include "loadPic.h"
#include "Trie.h"

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


char* intToBit(int val, int bitLen) {
    char* result = (char*)malloc(bitLen+1);
    for ( int i = 0 ; i < bitLen ; ++i ) {
        result[bitLen-1-i] = (val&(1<<i)) ? '1':'0';
    }    
    result[bitLen] = '\0';
    return result;
}    

void symTobits(Img* image) {
    int i_huf = 0;
    int baseVal = 0;
    for ( int i_dht = 0 ; i_dht < image->num_DHT ; ++i_dht ) {
        baseVal = 0;
        size_t tableSize = image->DHTs[i_dht].dht_len;
        HufTable table[tableSize];
        i_huf = 0;
        int changed = 0;
        for ( int height = 1 ; height <= 16 ; ++height ) {
            int num_height = image->DHTs[i_dht].codeLen[height-1];
            while ( num_height != 0 ) {
                changed = 1;
                table[i_huf].bitLen = height; 
                table[i_huf].val = image->DHTs[i_dht].table8[i_huf];
                table[i_huf].bitStr = intToBit(baseVal, height);
                if (num_height != 1)
                    baseVal++;
                num_height--;
                i_huf++;
            }    
            if (changed) {
                baseVal = (baseVal+1)*2;
            }    
        }    
        Node* root = buildTrie(table, tableSize);
        image->DHTs[i_dht].root = root;
        // printTrie(image->DHTs[i_dht].root);
        // printf("===============================\n");
        freeTrie(root);
        for ( size_t tSize = 0 ; tSize < tableSize ; ++tSize )
            free(table[tSize].bitStr);
    }

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
					image->DHTs[i_dht].dht_len = read_2bytes - 19;  // 3(length field + type) + 16(code length)
					image->DHTs[i_dht].tClass = (TableClass)read_byte;
					i_dht++;
					break;
                case SOF:
                    image->SOF_info.precision = read_byte; 
			        fread(&read_2bytes, 1, 2, fptr);  // width
                    image->pic_width = littleToBigEndian16(read_2bytes);
			        fread(&read_2bytes, 1, 2, fptr);  // height
                    image->pic_height = littleToBigEndian16(read_2bytes);
			        fread(&read_byte, 1, 1, fptr);
                    uint8_t num_component = read_byte;
                    image->SOF_info.num_component = num_component;
                    image->SOF_info.compInfo = (CompInfo*)malloc(num_component * sizeof(CompInfo));
                    for ( int comp = 0 ; comp < image->SOF_info.num_component; ++comp ) {
			            fread(&read_byte, 1, 1, fptr);
                        image->SOF_info.compInfo[comp].id = read_byte;
			            fread(&read_byte, 1, 1, fptr);
                        image->SOF_info.compInfo[comp].horzFactor = (read_byte >> 4) & 0x0F;
                        image->SOF_info.compInfo[comp].vertFactor = (read_byte & 0x0F) ;
			            fread(&read_byte, 1, 1, fptr);
                        image->SOF_info.compInfo[comp].component = (Component)read_byte;
                    }    
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
		free(image->DHTs[i].table8);
	}	
	free(image->DHTs);

    free(image->SOF_info.compInfo);
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
		image->DHTs[i].table8 = (uint8_t*)malloc(image->DHTs[i].dht_len * sizeof(uint8_t));
	}	
}	 

// use DQT, DHT macro to distinguish in which table will be filled 
void bufferRead(FILE* fptr, Img* image, uint8_t type, int index) {
	size_t readSize;
	int precision;	
	if ( type == DQT ) {
		readSize = image->DQTs[index].dqt_len;
		precision = image->DQTs[index].precision;
		if ( precision == 0 ) fread(image->DQTs[index].table8, 1, readSize, fptr );
		else fread(image->DQTs[index].table16, 1, readSize, fptr );
	}	
	else if ( type == DHT ){
		readSize = image->DHTs[index].dht_len;
		fread(image->DHTs[index].codeLen, 1, 16, fptr);
		fread(image->DHTs[index].table8, 1, readSize, fptr ); 
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
		// int sum = 0;
        printf("DHT codeLen: \n");
        for ( int m = 0 ; m < 16 ; ++m ) {
            printf("%d ",image->DHTs[i].codeLen[m]);
        }    
        printf("\n");
		for ( int j = 0 ; j < image->DHTs[i].dht_len ; ++j ) {
			printf("%d ", image->DHTs[i].table8[j]);
			if ( j % 8 == 7 ) printf("\n");
		}	
		printf("\n");
	}

    printf("pic_width: %d, pic_height: %d\n", image->pic_width, image->pic_height);
    printf("precision: %d\n", image->SOF_info.precision);
    for ( int i = 0 ; i < image->SOF_info.num_component ; ++i ) {
        printf("id: %d\n", image->SOF_info.compInfo[i].id);
        printf("horzFactor: %d\n", image->SOF_info.compInfo[i].horzFactor);
        printf("vertFactor: %d\n", image->SOF_info.compInfo[i].vertFactor);
        printf("component: %d\n", image->SOF_info.compInfo[i].component);
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
    symTobits(&image);
	printDebug(&image);
	freeTable(&image);
	fclose(fptr);
}	

/* ===============================bmp read=============================== */
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
