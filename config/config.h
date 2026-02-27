#ifndef CONFIG_H
#define CONFIG_H

#define MAX_STR_LEN 1024
#include "../mem/arena.h"

enum Mode {
    MODE_NONE,
    MODE_TRAIN,
    MODE_TEST
};

struct Config {
    enum Mode mode;
    int seed;
    double lr;
    char* imgPath;
    char* imgLabelPath;
    unsigned int max_iter;
    char* save_path;
};

void config_init(struct Config* c);
int load_json(const char* path, struct arena* a, struct Config* c);

#endif
