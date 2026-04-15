#ifndef LOADPIC_H
#define LOADPIC_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "structDef.h"

uint32_t littleToBigEndian32(uint32_t little);
uint16_t littleToBigEndian16(uint16_t little);
uint16_t byteConcat(uint8_t byte1, uint8_t byte2);
char* intToBit(int val, int bitLen);
/* ===============================bmp read=============================== */
struct Dataset* tk_dataset_create(struct tk_rt_ctx* ctx);
void loadImgLabel(struct tk_rt_ctx* ctx,
                  struct Dataset* dataset,
                  const char filePath[]);
void loadImgFile(struct tk_rt_ctx* ctx, Dataset* dataset, const char filePath[], int sampleCnt);
#endif
