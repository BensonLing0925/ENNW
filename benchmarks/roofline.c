#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <tk_time.h>
#include <arm_neon.h>
#include <assert.h>


#define ACC_INIT(num) \
	float32x4_t a##num = vdupq_n_f32(num * 0.1 + 0.05f); \
	float32x4_t b##num = vdupq_n_f32(0.25f); \
	float32x4_t acc##num = vdupq_n_f32(num * 0.1 + 0.5f)

#define ACC_DO(num) \
	acc##num = vfmaq_f32(acc##num, a##num, b##num)

#define ACC_CLEAN(num) \
	acc##num = vdupq_n_f32(0.0f)

#define INIT_ONE(n)  ACC_INIT(n);
#define DO_ONE(n)    ACC_DO(n);
#define CLEAN_ONE(n) ACC_CLEAN(n);
#define PRINT_ONE(n) printf("%f ", (double)vaddvq_f32(acc##n));
#define COUNT_ONE(n) + 1

#define PRINT_NL printf("\n")

#define ACC_LIST(OP) \
    OP(0) OP(1) OP(2) OP(3)

enum {NUM_ACC = 0 ACC_LIST(COUNT_ONE)};

#define DEFAULT_ITER (UINT64_C(2) << 31)
#define DEFAULT_RUNS 10
#define DEFAULT_WARMUP 5
#define DEFAULT_SIZE (UINT64_C(2) << 12)

// compute
void fma_acc_run(uint64_t iteration, uint64_t runs, uint64_t warmup) {
	uint64_t iter = iteration;

	printf("Number of iterations: %ld\n", iter);
	printf("Number of accumulators: %d\n", NUM_ACC);
	printf("Warmup runs: %ld\n", warmup);

	ACC_LIST(INIT_ONE);
	ACC_LIST(CLEAN_ONE);

	for ( uint64_t w = 0 ; w < warmup ; ++w ) {
		for ( uint64_t i = 0 ; i < iter ; ++i ) {
			ACC_LIST(DO_ONE);
		}	
		printf("Warmup checksum %ld:\n", w);
		ACC_LIST(PRINT_ONE);
		ACC_LIST(CLEAN_ONE);
		PRINT_NL;
	}

	ACC_LIST(CLEAN_ONE);
	printf("Warmup finished\n");

	for ( uint64_t r = 0 ; r < runs ; ++r ) {
		uint64_t start = tk_get_now_ns();
		for ( uint64_t i = 0 ; i < iter ; ++i ) {
			ACC_LIST(DO_ONE);
		}	
		uint64_t elapsed_ns = tk_get_now_ns() - start;
		double gflops = (double)iter * NUM_ACC * 4.0 * 2.0 / (double)elapsed_ns;	
		ACC_LIST(PRINT_ONE);
		printf("Elapsed: %.3f ms\n", (double)elapsed_ns / 1e6);
		printf("Throughput: %.6f GFLOP/s\n", gflops);
		ACC_LIST(CLEAN_ONE);
	}
}

void data_transfer_run(uint64_t bytes) {
	float* mem = NULL;
	uint64_t elements = bytes / sizeof(float);

	mem = malloc(bytes);
	while (mem == NULL) {
		fprintf(stderr, "Not enough memory for size: %ld\n", bytes);
		bytes /= 2;
		fprintf(stderr, "Reduce memory allocation by half: %ld\n", bytes);
		mem = malloc(bytes);
	}
	fprintf(stdout, "Final size: %ld bytes\n", bytes);
	
	float sum0 = 0.0f;
	float sum1 = 0.0f;
	float sum2 = 0.0f;
	float sum3 = 0.0f;

	for ( uint64_t i = 0 ; i < elements ; ++i ) {
		mem[i] = (float)(i % 4) * 0.01f + 0.2f;
	}

	assert(elements % 4 == 0);

	uint64_t start = tk_get_now_ns();

	for ( uint64_t i = 0 ; i < elements ; i += 4 ) {
		sum0 += mem[i];
		sum1 += mem[i + 1];
		sum2 += mem[i + 2];
		sum3 += mem[i + 3];
	}

	uint64_t elapsed_ns = tk_get_now_ns() - start;
    printf("Checksum: %f %f %f %f\n",
           (double)sum0, (double)sum1,
           (double)sum2, (double)sum3);

	if (elapsed_ns != 0) {
        printf("Read bandwidth: %.3f GB/s\n",
               (double)bytes / (double)elapsed_ns);
    }

	free(mem);
}

uint64_t str_to_uint64(char* s) {
	char *end;

	if (s[0] < '0' || s[0] > '9') {
		fprintf(stderr, "Invalid iteration count: %s\n", s);
		return 1;
	}

	errno = 0;
	uintmax_t value = strtoumax(s, &end, 10);

	if (errno == ERANGE || *end != '\0' ||
		value == 0 || value > UINT64_MAX) {
		fprintf(stderr, "Invalid iteration count: %s\n", s);
		fprintf(stdout, "Return 1 instead\n");
		return 1;
	}
	return (uint64_t)value;
}

int main(int argc, char* argv[]) {
	
	uint64_t iter = DEFAULT_ITER;
	uint64_t runs = DEFAULT_RUNS;
	uint64_t warmup = DEFAULT_WARMUP;
	uint64_t size = DEFAULT_SIZE;

	if (argc == 1) {
		printf("usage: ./roofline [--compute] [--memory]\n");
		return 0;
	}
	if (argc >= 2) {
		if (strcmp(argv[1], "--compute") == 0) {
			for ( int i = 2 ; i < argc ; ++i ) {
				if (strcmp(argv[i], "--iter") == 0) iter = str_to_uint64(argv[++i]);
				else if (strcmp(argv[i], "--warmup") == 0) warmup = str_to_uint64(argv[++i]);
				else if (strcmp(argv[i], "--runs") == 0) runs = str_to_uint64(argv[++i]);
				else {
					printf("Unknown options: %s\n", argv[i]);
					return -1;
				}
			}
			fma_acc_run(iter, runs, warmup);
		}
		else if (strcmp(argv[1], "--memory") == 0) {
			for ( int i = 2 ; i < argc ; ++i ) {
				if (strcmp(argv[i], "--size") == 0) size = str_to_uint64(argv[++i]);
			}
			data_transfer_run(size);
		}
		else {
			printf("Unknown options: %s\n", argv[1]);
			return -1;
		}
	}
}
