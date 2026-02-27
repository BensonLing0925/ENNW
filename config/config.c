#include "config.h"
#include "cJSON.h"
#include "arena.h"

void config_init(struct Config* c) {
    c->mode = MODE_NONE;
    c->seed = -1;   // seed = -1 will be time(NULL)
    c->lr = 0.01;
    c->imgPath = NULL;
    c->imgLabelPath = NULL;
    c->max_iter = 100;
    c->save_path = NULL;
}

static char* read_entire_file_arena(const char* path, struct arena* a) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }

    // +1 for NUL
    char* buf = arena_alloc(a, (size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }

    size_t nread = fread(buf, 1, (size_t)sz, f);
    fclose(f);

    buf[nread] = '\0'; // ensure NUL even if short read
    return buf;
}

int load_json(const char* path, struct arena* a,struct Config* c) {
    FILE* fptr = NULL;
    char read_buf[MAX_STR_LEN];
    memset(read_buf, 0, MAX_STR_LEN);
    fptr = fopen(path, "r");
    if (!fptr) {
        perror("Error when reading files\n");
        return -1;
    }
    char* text = read_entire_file_arena(path, a);
    if (!text) {
        fprintf(stderr, "Error reading config file '%s': %s\n", path, strerror(errno));
        goto cleanup;
    }
    cJSON* json = NULL;
    json = cJSON_Parse(text);
    if (!json) {
        fprintf(stderr, "Error when parsing JSON: %s\n", cJSON_GetErrorPtr());
        return -1;
    }

    /* setting struct Config */
    const cJSON* mode = NULL;
    const cJSON* seed = NULL;
    const cJSON* lr = NULL;
    const cJSON* imgPath = NULL;
    const cJSON* imgLabelPath = NULL;
    const cJSON* max_iter = NULL;
    const cJSON* save_path = NULL;
    int ret = -1;

    mode = cJSON_GetObjectItemCaseSensitive(json, "mode");
    if (cJSON_IsString(mode) && (mode->valuestring != NULL)) {
        if (!strcmp("TRAIN", mode->valuestring))
            c->mode = MODE_TRAIN;
        else if (!strcmp("TEST", mode->valuestring))
            c->mode = MODE_TEST;
    }
    else {
        perror("Error while reading JSON element \"mode\": invalid option");
        goto cleanup;
    }

    seed = cJSON_GetObjectItemCaseSensitive(json, "seed");
    if (cJSON_IsNumber(seed))
        c->seed = (int) seed->valuedouble;
    else
        printf("No seed specified, use time(NULL) as seed");

    lr = cJSON_GetObjectItemCaseSensitive(json, "lr");
    if (cJSON_IsNumber(lr))
        c->lr = lr->valuedouble;
    else {
        perror("Error while reading JSON element \"lr\": invalid option");
        goto cleanup;
    }

    imgPath = cJSON_GetObjectItemCaseSensitive(json, "imgPath");
    if (cJSON_IsString(imgPath) && (imgPath->valuestring != NULL))
        c->imgPath = arena_strdup(a, imgPath->valuestring);
    else {
        perror("Error while reading JSON element \"imgPath\": invalid option");
        goto cleanup;
    }

    imgLabelPath = cJSON_GetObjectItemCaseSensitive(json, "imgLabelPath");
    if (cJSON_IsString(imgLabelPath) && (imgLabelPath->valuestring != NULL)) {
        c->imgLabelPath = arena_strdup(a, imgLabelPath->valuestring);
        printf("raw imgLabelPath: [%s]\n", imgLabelPath->valuestring);
    }
    else {
        perror("Error while reading JSON element \"imgLabelPath\": invalid option");
        goto cleanup;
    }

    max_iter = cJSON_GetObjectItemCaseSensitive(json, "max_iter");
    if (cJSON_IsNumber(max_iter)) {
        if (max_iter->valuedouble < 0)
            perror("JSON element \"max_iter\" must be >= 0");
        c->max_iter = (unsigned int) max_iter->valuedouble;
    }
    else {
        printf("Invalid option for \"max_iter\", using default value: %d", c->max_iter);
        goto cleanup;
    }

    save_path = cJSON_GetObjectItemCaseSensitive(json, "save_path");
    if (cJSON_IsString(save_path) && (save_path->valuestring != NULL)) {
        c->save_path = arena_strdup(a, save_path->valuestring);
        printf("raw save_path:    [%s]\n", save_path->valuestring);
    }
    else {
        printf("[REMINDER] Not saving model weights\n");
    }

    ret = 0;
    fclose(fptr);

cleanup:
    if (json) cJSON_Delete(json);
    return ret;
}
