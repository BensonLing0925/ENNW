#ifndef TK_FILE_H
#define TK_FILE_H
#include <stddef.h>
#include <inttypes.h>

struct tk_file_data {
    size_t bytes;
    uint8_t* data;  // byte
};

struct tk_file_data* tk_file_create();
struct tk_file_data* tk_file_read_all(char* path);
#endif
