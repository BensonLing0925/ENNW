#ifndef LOADPIC_H
#define LOADPIC_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "structDef.h"
#include "Trie.h"

#define BUFSIZE 2048
#define FILELEN 256
#define H_OFFSET 5
#define W_OFFSET 7

#define SOS 0xDA
#define SOF 0xC0
#define DQT 0xDB
#define DHT 0xC4
#define SOI 0xD8
#define EOI 0xD9

const uint8_t SOS_MARKER[2] = { 0xFF, 0xDA };
const uint8_t SOF_MARKER[2] = { 0xFF, 0xC0 };
const uint8_t DQT_MARKER[2] = { 0xFF, 0xDB };
const uint8_t DHT_MARKER[2] = { 0xFF, 0xC4 };
const uint8_t SOI_MARKER[2] = { 0xFF, 0xD8 };
const uint8_t EOI_MARKER[2] = { 0xFF, 0xD9 };

uint32_t littleToBigEndian32(uint32_t little);
uint16_t littleToBigEndian16(uint16_t little);
uint16_t byteConcat(uint8_t byte1, uint8_t byte2);
char* intToBit(int val, int bitLen);
void symTobits(Img* image);
void readTableInfo(FILE* fptr, Img* image);
void readMarkers(FILE* fptr, Img* image);
void freeTable(Img* image);
void allocTable(Img* image);
void bufferRead(FILE* fptr, Img* image, uint8_t type, int index);
void readTables(FILE* fptr, Img* image);
void printDebug(Img* image);
void initImg(Img* image);
void imageProcess(const char filePath[]);
/* ===============================bmp read=============================== */
void loadImgLabel(const char filePath[], DataSet* dataset);
void loadImgFile(const char filePath[], DataSet* dataset, int sampleCnt);
void loadKernal(const char filePath[], Conv2D* conv);

#endif
