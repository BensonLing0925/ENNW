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

    /* Architecture (all optional; have defaults) */
    int num_filter;     /* conv2d filters, default 10 */
    int kernel_size;    /* conv2d square kernel, default 3 */
    int pool_size;      /* max-pool square kernel, default 2 */
    int tf_n_heads;     /* transformer heads, default 13 */
    int* fc_layers;     /* FC output-dim array, e.g. [100, 50, 10] */
    int fc_num_layers;  /* length of fc_layers, default 3 */

    /* Inference */
    char* weights_path; /* path to .bin weights (for TEST mode) */
};

void config_init(struct Config* c);
int load_json(const char* path, struct arena* a, struct Config* c);

#endif
