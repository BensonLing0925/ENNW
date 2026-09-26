#ifndef TK_FILE_H
#define TK_FILE_H

struct tk_file_data {
    long bytes;
    char* data;
};

struct tk_file_data* tk_file_create();
struct tk_file_data* tk_file_read_all(char* path);
#endif
