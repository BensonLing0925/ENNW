# =========================
# Portable Makefile (GNU Make)
# Windows (MinGW) + Linux
# =========================

CC := gcc

# ---- Platform detect ----
ifeq ($(OS),Windows_NT)
    EXEEXT := .exe
    RM := del /S /Q
else
    EXEEXT :=
    RM := rm -f
    NULLDEV := /dev/null
endif

# ---- Project layout ----
SRC_DIR     := src
MEM_DIR     := $(SRC_DIR)/mem
MODULES_DIR := $(SRC_DIR)/modules
RT_DIR		:= $(SRC_DIR)/runtime
RT_WS_DIR	:= $(RT_DIR)/workspaces
RT_SG_DIR	:= $(RT_DIR)/graph
WEIGHTIO_DIR 	:= $(SRC_DIR)/weightio
FC_DIR          := $(MODULES_DIR)/fc
CONV_DIR        := $(MODULES_DIR)/conv
PL_DIR          := $(MODULES_DIR)/pooling
TRANSFORMER_DIR := $(MODULES_DIR)/transformer
EMB_DIR         := $(TRANSFORMER_DIR)/embedding
DISTILBERT_DIR  := $(TRANSFORMER_DIR)/distilbert
GPT2_DIR  		:= $(TRANSFORMER_DIR)/gpt2
OPS_DIR         := $(SRC_DIR)/ops
ERROR_DIR       := $(SRC_DIR)/error
PROF_DIR		:= $(SRC_DIR)/profiler
PLATFORM_DIR	:= $(SRC_DIR)/platform

THIRD_PARTY_DIR	:= $(SRC_DIR)/third_party
CJSON_DIR    := $(THIRD_PARTY_DIR)/cJSON
CFG_DIR		 := $(SRC_DIR)/config

EXAMPLES_DIR 	:= examples
TEST_DIR		:= test

# --- storing test binaries ---
BIN_DIR		:= bin

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

DISTILBERT_INFER_TARGET := $(BIN_DIR)/distilbert_infer$(EXEEXT)
SST2_EVAL_TARGET        := $(BIN_DIR)/sst2_eval$(EXEEXT)
OPS_TEST_TARGET			:= $(BIN_DIR)/ops_test$(EXEEXT)
GPT2_TEST_TARGET			:= $(BIN_DIR)/gpt2_test$(EXEEXT)
GPT2_IO_TEST_TARGET			:= $(BIN_DIR)/gpt2_io_test$(EXEEXT)
OMP_TEST_TARGET			:= $(BIN_DIR)/omp_test$(EXEEXT)

# ---- Include paths ----
INCLUDES := -I$(SRC_DIR) -I$(MODULES_DIR) -I$(FC_DIR) -I$(CONV_DIR) \
            -I$(CFG_DIR) -I$(CJSON_DIR) -I$(MEM_DIR) -I$(ERROR_DIR) -I$(PLATFORM_DIR)\
            -I$(OPS_DIR) -I$(RT_DIR) -I$(RT_WS_DIR) -I$(PL_DIR) -I$(TRANSFORMER_DIR) \
            -I$(WEIGHTIO_DIR) -I$(EMB_DIR) -I$(DISTILBERT_DIR) -I$(EXAMPLES_DIR)	 \
			-I$(RT_SG_DIR) -I$(GPT2_DIR) -I$(PROF_DIR) 
			
# Math library (needed on Linux if using exp/sqrt/etc)
LDLIBS ?= -lm -fopenmp 
# Add PROF=1 to enable profiler
ifeq ($(PROF),1)
    PROF_FLAGS := -DPROF
	LDLIBS     += -Wl,--wrap=GOMP_parallel 
endif 

# ---- Common flags ----
CFLAGS_COMMON := -Wall -Wextra -Werror=implicit-function-declaration  $(INCLUDES) $(PROF_FLAGS)

ifeq ($(VERBOSE),1)
    CFLAGS_COMMON += -DTK_RT_VERBOSE=1
endif

# C standards by module
CFLAGS_C23 := $(CFLAGS_COMMON) -std=gnu17
CFLAGS_C89 := $(CFLAGS_COMMON) -std=c89

# Add DEBUG=1 to build debug version
ifeq ($(DEBUG),1)
    CFLAGS_C23 += -O0 -g -DDEBUG
    CFLAGS_C89 += -O0 -g -DDEBUG
else
    CFLAGS_C23 += -O3 -fopenmp
    CFLAGS_C89 += -O3 -fopenmp
endif


# ---- Shared library sources (no main, used by both targets) ----
LIB_SRC := $(wildcard $(FC_DIR)/*.c) \
           $(wildcard $(CONV_DIR)/*.c) \
           $(wildcard $(PL_DIR)/*.c) \
           $(wildcard $(OPS_DIR)/*.c) \
           $(wildcard $(RT_DIR)/*.c) \
           $(wildcard $(RT_WS_DIR)/*.c) \
		   $(wildcard $(RT_SG_DIR)/*.c) \
           $(wildcard $(TRANSFORMER_DIR)/*.c) \
           $(wildcard $(EMB_DIR)/*.c) \
           $(wildcard $(DISTILBERT_DIR)/*.c) \
           $(wildcard $(GPT2_DIR)/*.c) \
		   $(PLATFORM_DIR)/tk_time.c \
           $(CFG_DIR)/config.c \
           $(MEM_DIR)/arena.c \
           $(ERROR_DIR)/rt_error.c \
           $(filter-out $(WEIGHTIO_DIR)/test_weightio.c, $(wildcard $(WEIGHTIO_DIR)/*.c))

ifeq ($(PROF), 1)
    PROF_CORE_SRC := $(filter-out $(PROF_DIR)/test_%.c $(PROF_DIR)/tk_profiler_view.c, $(wildcard $(PROF_DIR)/*.c))
    LIB_SRC += $(PROF_CORE_SRC)
endif

# ---- Source groups ----
# distilbert_infer-specific source (its own main)
DISTILBERT_INFER_SRC := $(EXAMPLES_DIR)/distilbert_infer.c

# sst2_infer-specific source (its own main)
SST2_INFER_SRC := $(EXAMPLES_DIR)/sst2_infer.c

# sst2_eval-specific source (its own main)
SST2_EVAL_SRC := $(EXAMPLES_DIR)/sst2_eval.c

# ops_test-specific source
OPS_TEST_SRC := $(TEST_DIR)/ops_test.c

# gpt2_test-specific source
GPT2_TEST_SRC := $(TEST_DIR)/gpt2_test.c

GPT2_IO_TEST_SRC := $(TEST_DIR)/gpt2_io_test.c

OMP_TEST_SRC := $(TEST_DIR)/omp_test.c

# cJSON source (compile as C89)
SRC_C89 := $(CJSON_DIR)/cJSON.c

# Object files
LIB_OBJ              := $(LIB_SRC:.c=.o)
DISTILBERT_INFER_OBJ := $(DISTILBERT_INFER_SRC:.c=.o)
SST2_INFER_OBJ       := $(SST2_INFER_SRC:.c=.o)
SST2_EVAL_OBJ        := $(SST2_EVAL_SRC:.c=.o)
OPS_TEST_OBJ         := $(OPS_TEST_SRC:.c=.o)
GPT2_TEST_OBJ         := $(GPT2_TEST_SRC:.c=.o)
GPT2_IO_TEST_OBJ         := $(GPT2_IO_TEST_SRC:.c=.o)
OMP_TEST_OBJ			:= $(OMP_TEST_SRC:.c=.o)
OBJ_C89              := $(SRC_C89:.c=.o)

# ---- Default target ----
.PHONY: all
all: $(DISTILBERT_INFER_TARGET) $(SST2_INFER_TARGET) $(SST2_EVAL_TARGET) $(OPS_TEST_TARGET) $(GPT2_TEST_TARGET) $(GPT2_IO_TEST_TARGET) $(OMP_TEST_TARGET)

# ---- Link ----
$(DISTILBERT_INFER_TARGET): $(DISTILBERT_INFER_OBJ) $(LIB_OBJ) $(OBJ_C89) | $(BIN_DIR)
	$(CC) $^ -o $@ $(LDLIBS)

$(SST2_INFER_TARGET): $(SST2_INFER_OBJ) $(LIB_OBJ) $(OBJ_C89) | $(BIN_DIR)
	$(CC) $^ -o $@ $(LDLIBS)

$(SST2_EVAL_TARGET): $(SST2_EVAL_OBJ) $(LIB_OBJ) $(OBJ_C89) | $(BIN_DIR)
	$(CC) $^ -o $@ $(LDLIBS)

$(OPS_TEST_TARGET): $(OPS_TEST_OBJ) $(LIB_OBJ) $(OBJ_C89) | $(BIN_DIR)
	$(CC) $^ -o $@ $(LDLIBS)

$(GPT2_TEST_TARGET): $(GPT2_TEST_OBJ) $(LIB_OBJ) $(OBJ_C89) | $(BIN_DIR)
	$(CC) $^ -o $@ $(LDLIBS)

$(GPT2_IO_TEST_TARGET): $(GPT2_IO_TEST_OBJ) $(LIB_OBJ) $(OBJ_C89) | $(BIN_DIR)
	$(CC) $^ -o $@ $(LDLIBS)

$(OMP_TEST_TARGET): $(OMP_TEST_OBJ) $(LIB_OBJ) $(OBJ_C89) | $(BIN_DIR)
	$(CC) $^ -o $@ $(LDLIBS)

# ---- Pattern rules by standard ----
# All C23 objects (lib, nn, examples)
$(LIB_OBJ) $(DISTILBERT_INFER_OBJ) $(SST2_INFER_OBJ) $(SST2_EVAL_OBJ) $(OPS_TEST_OBJ) $(GPT2_TEST_OBJ) $(GPT2_IO_TEST_OBJ) $(OMP_TEST_OBJ): %.o: %.c
	$(CC) $(CFLAGS_C23) -c $< -o $@

# Compile cJSON with C89
$(OBJ_C89): %.o: %.c
	$(CC) $(CFLAGS_C89) -c $< -o $@

# ---- Helpers ----
.DEFAULT_GOAL := all

.PHONY: clean run run-distilbert run-sst2 run-sst2-eval print

run-distilbert: $(DISTILBERT_INFER_TARGET)
	./$(DISTILBERT_INFER_TARGET)

run-sst2: $(SST2_INFER_TARGET)
	./$(SST2_INFER_TARGET)

run-sst2-eval: $(SST2_EVAL_TARGET)
	./$(SST2_EVAL_TARGET)

run-ops-test: $(OPS_TEST_TARGET)
	./$(OPS_TEST_TARGET)

run-gpt2-test: $(GPT2_TEST_TARGET)
	./$(GPT2_TEST_TARGET)

run-omp-test: $(OMP_TEST_TARGET)
	./$(OMP_TEST_TARGET)

clean:
	-$(RM) $(LIB_OBJ) $(DISTILBERT_INFER_OBJ) $(SST2_INFER_OBJ) $(SST2_EVAL_OBJ) \
	       $(OPS_TEST_OBJ) $(GPT2_TEST_OBJ) $(GPT2_IO_TEST_OBJ) $(OMP_TEST_OBJ) $(OBJ_C89)
	-$(RM) -r $(BIN_DIR)

print:
	@echo DISTILBERT_INFER_TARGET=$(DISTILBERT_INFER_TARGET)
	@echo LIB_SRC=$(LIB_SRC)
	@echo SRC_C89=$(SRC_C89)

