# =========================
# Portable Makefile (GNU Make)
# Windows (MinGW) + Linux
# =========================

CC := gcc

# ---- Platform detect ----
ifeq ($(OS),Windows_NT)
    EXEEXT := .exe
    RM := del /Q
    NULLDEV := nul
else
    EXEEXT :=
    RM := rm -f
    NULLDEV := /dev/null
endif

# ---- Project layout ----
SRC_DIR    := src
CFG_DIR    := config
CJSON_DIR  := $(CFG_DIR)/cJSON
MEM_DIR    := mem

TARGET := nn$(EXEEXT)

# ---- Include paths ----
INCLUDES := -I$(SRC_DIR) -I$(CFG_DIR) -I$(CJSON_DIR) -I$(MEM_DIR)

# ---- Common flags ----
CFLAGS_COMMON := -Wall -Wextra $(INCLUDES)

# C standards by module
CFLAGS_C23 := $(CFLAGS_COMMON) -std=gnu17
CFLAGS_C89 := $(CFLAGS_COMMON) -std=c89

# Add DEBUG=1 to build debug version
ifeq ($(DEBUG),1)
    CFLAGS_C23 += -O0 -g -DDEBUG
    CFLAGS_C89 += -O0 -g -DDEBUG
else
    CFLAGS_C23 += -O2
    CFLAGS_C89 += -O2
endif

# Math library (needed on Linux if using exp/sqrt/etc)
LDLIBS ?= -lm

# ---- Source groups ----
# 1) Your CNN / main and other app sources (C23)
SRC_C23 := $(wildcard $(SRC_DIR)/*.c) \
           $(CFG_DIR)/config.c \
           $(MEM_DIR)/arena.c

# 2) cJSON source (compile as C89)
SRC_C89 := $(CJSON_DIR)/cJSON.c

OBJ_C23 := $(SRC_C23:.c=.o)
OBJ_C89 := $(SRC_C89:.c=.o)
OBJS    := $(OBJ_C23) $(OBJ_C89)

# ---- Default target ----
.PHONY: all
all: $(TARGET)

# ---- Link ----
$(TARGET): $(OBJS)
	$(CC) $^ -o $@ $(LDLIBS)

# ---- Pattern rules by standard ----
# Compile listed C23 objects with C23
$(OBJ_C23): %.o: %.c
	$(CC) $(CFLAGS_C23) -c $< -o $@

# Compile cJSON with C89
$(OBJ_C89): %.o: %.c
	$(CC) $(CFLAGS_C89) -c $< -o $@

# ---- Helpers ----
.PHONY: clean run print

run: $(TARGET)
	./$(TARGET)

clean:
	-$(RM) $(OBJS) $(TARGET) 2>$(NULLDEV)

print:
	@echo TARGET=$(TARGET)
	@echo SRC_C23=$(SRC_C23)
	@echo SRC_C89=$(SRC_C89)
	@echo OBJ_C23=$(OBJ_C23)
	@echo OBJ_C89=$(OBJ_C89)
