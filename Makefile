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
MODULES_DIR := $(SRC_DIR)/modules
RT_DIR		:= $(SRC_DIR)/runtime
RT_WS_DIR	:= $(RT_DIR)/workspaces
RT_SG_DIR	:= $(RT_DIR)/graph
FC_DIR          := $(MODULES_DIR)/fc
CONV_DIR        := $(MODULES_DIR)/conv
PL_DIR          := $(MODULES_DIR)/pooling
TRANSFORMER_DIR := $(MODULES_DIR)/transformer
EMB_DIR         := $(TRANSFORMER_DIR)/embedding
DISTILBERT_DIR  := $(TRANSFORMER_DIR)/distilbert
OPS_DIR         := $(SRC_DIR)/ops
NNUTILS_DIR     := $(SRC_DIR)/nn_utils
ERROR_DIR       := $(SRC_DIR)/error
PROF_DIR		:= $(SRC_DIR)/profiler
RAYLIB_DIR  := $(SRC_DIR)/raylib/src
RAYLIB_LIB  := $(RAYLIB_DIR)/libraylib.a

CFG_DIR      := config
CJSON_DIR    := $(CFG_DIR)/cJSON
MEM_DIR      := mem
WEIGHTIO_DIR := weightio
EXAMPLES_DIR := examples

TARGET                  := nn$(EXEEXT)
DISTILBERT_INFER_TARGET := distilbert_infer$(EXEEXT)
SST2_EVAL_TARGET        := sst2_eval$(EXEEXT)

# ---- Include paths ----
INCLUDES := -I$(SRC_DIR) -I$(MODULES_DIR) -I$(FC_DIR) -I$(CONV_DIR) -I$(NNUTILS_DIR) \
            -I$(CFG_DIR) -I$(CJSON_DIR) -I$(MEM_DIR) -I$(ERROR_DIR) \
            -I$(OPS_DIR) -I$(RT_DIR) -I$(RT_WS_DIR) -I$(PL_DIR) -I$(TRANSFORMER_DIR) \
            -I$(WEIGHTIO_DIR) -I$(EMB_DIR) -I$(DISTILBERT_DIR) -I$(EXAMPLES_DIR)	 \
			-I$(RT_SG_DIR)
			
# Math library (needed on Linux if using exp/sqrt/etc)
LDLIBS ?= -lm -fopenmp 
# Add PROF=1 to enable profiler
ifeq ($(PROF),1)
    PROF_FLAGS := -DPROF
    INCLUDES   += -I$(PROF_DIR) -I$(RAYLIB_DIR)
	LDLIBS     += $(RAYLIB_LIB) -Wl,--wrap=GOMP_parallel 
	ifeq ($(OS),Windows_NT)
        LDLIBS += -lgdi32 -lwinmm -lopengl32   # ← 加 -lopengl32
    else
        LDLIBS += -lGL -ldl -lpthread
    endif
endif 

# ---- Common flags ----
CFLAGS_COMMON := -Wall -Wextra $(INCLUDES) $(PROF_FLAGS)

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
           $(wildcard $(NNUTILS_DIR)/*.c) \
           $(wildcard $(OPS_DIR)/*.c) \
           $(wildcard $(RT_DIR)/*.c) \
           $(wildcard $(RT_WS_DIR)/*.c) \
		   $(wildcard $(RT_SG_DIR)/*.c) \
           $(wildcard $(TRANSFORMER_DIR)/*.c) \
           $(wildcard $(EMB_DIR)/*.c) \
           $(wildcard $(DISTILBERT_DIR)/*.c) \
           $(CFG_DIR)/config.c \
           $(MEM_DIR)/arena.c \
           $(ERROR_DIR)/rt_error.c \
           $(filter-out $(WEIGHTIO_DIR)/test_weightio.c, $(wildcard $(WEIGHTIO_DIR)/*.c))

ifeq ($(PROF), 1)
    # 這裡過濾掉 test_ 開頭的檔案，只取核心實作
    PROF_CORE_SRC := $(filter-out $(PROF_DIR)/test_%.c, $(wildcard $(PROF_DIR)/*.c))
    LIB_SRC += $(PROF_CORE_SRC)
endif

# ---- Source groups ----
# nn-specific sources (has main in NN.c), exclude Trie.c
NN_SRC := $(filter-out $(SRC_DIR)/Trie.c, $(wildcard $(SRC_DIR)/*.c))

# distilbert_infer-specific source (its own main)
DISTILBERT_INFER_SRC := $(EXAMPLES_DIR)/distilbert_infer.c

# sst2_infer-specific source (its own main)
SST2_INFER_SRC := $(EXAMPLES_DIR)/sst2_infer.c

# sst2_eval-specific source (its own main)
SST2_EVAL_SRC := $(EXAMPLES_DIR)/sst2_eval.c

# cJSON source (compile as C89)
SRC_C89 := $(CJSON_DIR)/cJSON.c

# Object files
LIB_OBJ              := $(LIB_SRC:.c=.o)
NN_OBJ               := $(NN_SRC:.c=.o)
DISTILBERT_INFER_OBJ := $(DISTILBERT_INFER_SRC:.c=.o)
SST2_INFER_OBJ       := $(SST2_INFER_SRC:.c=.o)
SST2_EVAL_OBJ        := $(SST2_EVAL_SRC:.c=.o)
OBJ_C89              := $(SRC_C89:.c=.o)

# ---- Default target ----
.PHONY: all
all: $(TARGET) $(DISTILBERT_INFER_TARGET) $(SST2_INFER_TARGET) $(SST2_EVAL_TARGET)

# ---- Link ----
$(TARGET): $(NN_OBJ) $(LIB_OBJ) $(OBJ_C89)
	$(CC) $^ -o $@ $(LDLIBS)

$(DISTILBERT_INFER_TARGET): $(DISTILBERT_INFER_OBJ) $(LIB_OBJ) $(OBJ_C89)
	$(CC) $^ -o $@ $(LDLIBS)

$(SST2_INFER_TARGET): $(SST2_INFER_OBJ) $(LIB_OBJ) $(OBJ_C89)
	$(CC) $^ -o $@ $(LDLIBS)

$(SST2_EVAL_TARGET): $(SST2_EVAL_OBJ) $(LIB_OBJ) $(OBJ_C89)
	$(CC) $^ -o $@ $(LDLIBS)

# ---- Pattern rules by standard ----
# All C23 objects (lib, nn, examples)
$(LIB_OBJ) $(NN_OBJ) $(DISTILBERT_INFER_OBJ) $(SST2_INFER_OBJ) $(SST2_EVAL_OBJ): %.o: %.c
	$(CC) $(CFLAGS_C23) -c $< -o $@

# Compile cJSON with C89
$(OBJ_C89): %.o: %.c
	$(CC) $(CFLAGS_C89) -c $< -o $@
	

# 編 raylib 靜態庫（只在 libraylib.a 不存在時觸發）
$(RAYLIB_LIB):
	$(MAKE) -C $(RAYLIB_DIR) PLATFORM=PLATFORM_DESKTOP

# ---- Helpers ----
.PHONY: clean run run-distilbert run-sst2 run-sst2-eval print

run: $(TARGET)
	./$(TARGET)

run-distilbert: $(DISTILBERT_INFER_TARGET)
	./$(DISTILBERT_INFER_TARGET)

run-sst2: $(SST2_INFER_TARGET)
	./$(SST2_INFER_TARGET)

run-sst2-eval: $(SST2_EVAL_TARGET)
	./$(SST2_EVAL_TARGET)

clean:
	-$(RM) $(LIB_OBJ) $(NN_OBJ) $(DISTILBERT_INFER_OBJ) $(SST2_INFER_OBJ) $(SST2_EVAL_OBJ) $(OBJ_C89) \
	       $(TARGET) $(DISTILBERT_INFER_TARGET)	$(SST2_EVAL_TARGET)

print:
	@echo TARGET=$(TARGET)
	@echo DISTILBERT_INFER_TARGET=$(DISTILBERT_INFER_TARGET)
	@echo LIB_SRC=$(LIB_SRC)
	@echo NN_SRC=$(NN_SRC)
	@echo SRC_C89=$(SRC_C89)

