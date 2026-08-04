#ifndef TK_DISTILBERT_IO_H
#define TK_DISTILBERT_IO_H

#include <stdint.h>
#include "distilbert.h"
#include "distilbert_cls.h"

#define DBERT_MAGIC      "DBERT\x00\x00\x00"
#define DBERT_MAGIC_LEN  8
#define DBERT_VERSION        1   /* base model, float32 weights              */
#define DBERT_VERSION_SST2   2   /* SST-2 head appended, float32 weights     */
#define DBERT_VERSION_INT8   3   /* SST-2 head appended, int8 transformer weights
                                    + float32 scale per weight matrix        */
#define DBERT_HEADER_SIZE 40   /* bytes */

/*
 * Load DistilBERT base weights from a DBERT binary file (VERSION 1 or 2).
 * model must be configured + allocated before calling.
 * Returns 0 on success. On success, *version_out is set to the file version.
 */
int tk_distilbert_load_weights(const char* path,
                               const struct tk_distilbert_config* cfg,
                               struct tk_distilbert* model,
                               uint32_t* version_out);

/*
 * Load SST-2 classification head weights appended at end of a VERSION 2 file.
 * Must be called after tk_distilbert_load_weights with the same path.
 * head must be allocated via tk_sst2_cls_alloc before calling.
 */
int tk_distilbert_load_cls_weights(const char* path,
                                   const struct tk_distilbert_config* cfg,
                                   struct tk_sst2_cls_head* head);

#endif
