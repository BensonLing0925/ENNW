#include "fileio.h"
#include <stdio.h>
#include <stdlib.h>

struct tk_file_data* tk_file_create() {
    return malloc(sizeof(struct tk_file_data));
}

void tk_file_free(struct tk_file_data* file_data) {
    free(file_data->data);
    free(file_data);
}

// caller free
// use malloc and free since file_data's lifecycle is short
struct tk_file_data* tk_file_read_all(char* path) {
    FILE* fptr = fopen(path, "rb");
    if (!fptr) {
        fprintf(stderr, "failed to open file: %s\n", path);
        return NULL;
    }

    if (fseek(fptr, 0, SEEK_END) != 0) {
        fprintf(stderr, "fseek failed\n");
        fclose(fptr);
        return NULL;
    }

    long bytes = ftell(fptr);
    if (bytes < 0) {
        fclose(fptr);
        return NULL;
    }
    rewind(fptr);
    
    struct tk_file_data* file_data = tk_file_create();
    if (!file_data) {
        fprintf(stderr, "struct allocation failed\n");
        fclose(fptr);
        return NULL;
    }
    file_data->bytes = bytes;
    file_data->data = malloc(bytes);
    if (!file_data->data) {
        fprintf(stderr, "data allocation failed\n");
        fclose(fptr);
        return NULL;
    }
    fread(file_data->data, file_data->bytes, 1, fptr);
    fclose(fptr);
    return file_data;
}
