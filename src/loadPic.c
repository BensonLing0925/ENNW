#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "structDef.h"
#include "loadPic.h"
#include "ops/tensor.h"
#include "ops/tensor_ops.h"
#include "runtime/rt_context.h"
#include "runtime/workspaces/rt_workspaces.h"

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

/*
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
*/

/* ===============================bmp read=============================== */

struct Dataset* tk_dataset_create(struct tk_rt_ctx* ctx) {
    struct Dataset* dataset = arena_alloc(ctx->meta_arena, sizeof(struct Dataset));
    return dataset;
}

void loadImgLabel(struct tk_rt_ctx* ctx,
                  struct Dataset* dataset,
                  const char filePath[]) {
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

    dataset->labels = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_U8, (int[]){num_sample}, 1);
    dataset->num_samples = num_sample;

    if (ctx->rt_type != RT_DRYRUN) {
        struct tk_tensor* labels = dataset->labels;
        fread(labels->data, tk_get_dtype_size(labels->dtype), num_sample, fptr);
    }
	fclose(fptr);
}		

// if sampleCnt = -1, load all samples
void loadImgFile(struct tk_rt_ctx* ctx, Dataset* dataset, const char filePath[], int sampleCnt){
	FILE* fptr = fopen(filePath, "rb");
	if (!fptr) {
		printf("File does not exist\n");
		return;
	}		
	int magic, total_num;
	int rows, cols;
    int num_sample;

	/*             image info            */
	fread(&magic, 4, 1, fptr);
	fread(&total_num, 4, 1, fptr);
	fread(&rows, 4, 1, fptr);
	fread(&cols, 4, 1, fptr);

	magic = littleToBigEndian32(magic);
	total_num = littleToBigEndian32(total_num);
	rows = littleToBigEndian32(rows);
	cols = littleToBigEndian32(cols);

    num_sample = (sampleCnt == -1) ? total_num : sampleCnt;

    dataset->samples = tk_ws_tensor_alloc(ctx->ws, ctx->meta_arena, TK_U8, (int[]){num_sample, rows, cols}, 3);
    dataset->rows = rows;
    dataset->cols = cols;

    if (ctx->rt_type != RT_DRYRUN) {
        uint64_t total_size = num_sample * rows * cols;
        struct tk_tensor* samples = dataset->samples;
        fread(samples->data, tk_get_dtype_size(samples->dtype), total_size, fptr);
    }

	fclose(fptr);
}		
