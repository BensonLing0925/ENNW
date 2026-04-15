#include <string.h>
#include "weightio.h"
#include "../mem/arena.h"
#include "../src/structDef.h"
#include "../src/error/rt_error.h"

/* ---- Low-level I/O helpers (non-static, declared in weightio.h) ---- */

int write_bytes(FILE* fp, const void* ptr, size_t n) {
    return fwrite(ptr, 1, n, fp) == n ? 0 : -1;
}

int write_u32(FILE* fp, uint32_t v) {
    return write_bytes(fp, &v, sizeof(v));
}

int write_u64(FILE* fp, uint64_t v) {
    return write_bytes(fp, &v, sizeof(v));
}

int write_f64(FILE* fp, double v) {
    return write_bytes(fp, &v, sizeof(v));
}

int read_bytes(FILE* fp, void* ptr, size_t n) {
    return (fread(ptr, 1, n, fp) == n) ? 0 : -1;
}

int read_u32(FILE* fp, uint32_t* v) {
    return read_bytes(fp, v, sizeof(*v));
}

int read_u64(FILE* fp, uint64_t* v) {
    return read_bytes(fp, v, sizeof(*v));
}

int read_f64(FILE* fp, double* v) {
    return read_bytes(fp, v, sizeof(*v));
}

/* ---- Header write ---- */

int header_write(FILE* fptr, const struct Binary_Header* bh) {
    if (!fptr || !bh) return -1;

    if (write_bytes(fptr, bh->magic, sizeof(bh->magic)) != 0) goto write_fail;
    if (write_u32(fptr, bh->ver)         < 0) goto write_fail;
    if (write_u32(fptr, bh->endian)      < 0) goto write_fail;
    if (write_u32(fptr, bh->dtype)       < 0) goto write_fail;
    if (write_u32(fptr, bh->model_type)  < 0) goto write_fail;
    if (write_u32(fptr, bh->layer_count) < 0) goto write_fail;
    if (write_u32(fptr, bh->input_h)     < 0) goto write_fail;
    if (write_u32(fptr, bh->input_w)     < 0) goto write_fail;
    if (write_u32(fptr, bh->input_c)     < 0) goto write_fail;
    if (write_bytes(fptr, bh->reserved, sizeof(bh->reserved)) != 0) goto write_fail;
    return 0;

write_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "header_write: write fail");
}

/* ---- Pool meta write/read ---- */

int pool_meta_write(FILE* fptr, const struct Binary_Pool_Layer_Meta* pm) {
    if (!fptr || !pm) return -1;
    if (write_u32(fptr, pm->stride_h)     < 0) goto write_fail;
    if (write_u32(fptr, pm->stride_w)     < 0) goto write_fail;
    if (write_u32(fptr, pm->kernel_h)     < 0) goto write_fail;
    if (write_u32(fptr, pm->kernel_w)     < 0) goto write_fail;
    if (write_u32(fptr, pm->padding_h)    < 0) goto write_fail;
    if (write_u32(fptr, pm->padding_w)    < 0) goto write_fail;
    if (write_u32(fptr, pm->pooling_type) < 0) goto write_fail;
    for (int i = 0; i < 4; i++)
        if (write_u32(fptr, pm->reserved[i]) < 0) goto write_fail;
    return 0;
write_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "pool_meta_write: write fail");
}

int pool_meta_load(FILE* fptr, struct Binary_Pool_Layer_Meta* pm) {
    if (!fptr || !pm) return -1;
    if (read_u32(fptr, &pm->stride_h)     < 0) goto read_fail;
    if (read_u32(fptr, &pm->stride_w)     < 0) goto read_fail;
    if (read_u32(fptr, &pm->kernel_h)     < 0) goto read_fail;
    if (read_u32(fptr, &pm->kernel_w)     < 0) goto read_fail;
    if (read_u32(fptr, &pm->padding_h)    < 0) goto read_fail;
    if (read_u32(fptr, &pm->padding_w)    < 0) goto read_fail;
    if (read_u32(fptr, &pm->pooling_type) < 0) goto read_fail;
    fseek(fptr, 4 * (long)sizeof(uint32_t), SEEK_CUR);  /* reserved */
    return 0;
read_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "pool_meta_load: read fail");
}

/* ---- FC meta write ---- */

int fc_meta_write(FILE* fptr, const struct Binary_FC_Layer_Meta* fc_blm) {
    if (!fptr || !fc_blm) return -1;
    if (write_u32(fptr, fc_blm->num_neurons) < 0) goto write_fail;
    if (write_u32(fptr, fc_blm->input_dim)   < 0) goto write_fail;
    if (write_u32(fptr, fc_blm->has_bias)    < 0) goto write_fail;
    if (write_u32(fptr, fc_blm->reserved)    < 0) goto write_fail;
    return 0;
write_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "fc_meta_write: write fail");
}

/* ---- Conv2D meta write ---- */

int conv2d_meta_write(FILE* fptr, const struct Binary_Conv2D_Layer_Meta* m) {
    if (!fptr || !m) RT_FAIL(RT_EINVAL, "conv2d_meta_write: NULL pointer");
    if (write_u32(fptr, m->num_filter)   < 0) goto write_fail;
    if (write_u32(fptr, m->in_channels)  < 0) goto write_fail;
    if (write_u32(fptr, m->kernel_h)     < 0) goto write_fail;
    if (write_u32(fptr, m->kernel_w)     < 0) goto write_fail;
    if (write_u32(fptr, m->stride_h)     < 0) goto write_fail;
    if (write_u32(fptr, m->stride_w)     < 0) goto write_fail;
    if (write_u32(fptr, m->padding_h)    < 0) goto write_fail;
    if (write_u32(fptr, m->padding_w)    < 0) goto write_fail;
    if (write_u32(fptr, m->has_bias)     < 0) goto write_fail;
    if (write_u32(fptr, m->pooling_type) < 0) goto write_fail;
    if (write_u32(fptr, m->pooling_h)    < 0) goto write_fail;
    if (write_u32(fptr, m->pooling_w)    < 0) goto write_fail;
    for (int i = 0; i < 4; ++i)
        if (write_u32(fptr, m->reserved[i]) < 0) goto write_fail;
    return 0;
write_fail:
    fclose(fptr);
    RT_FAIL(RT_EIO, "conv2d_meta_write: write fail");
}
