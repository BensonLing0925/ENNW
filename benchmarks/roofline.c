#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tk_time.h>
#include <arm_neon.h>

#define RUNS 10

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

int main(void) {

	uint64_t iter = (2 << 20);
	uint64_t warmup = 5;

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

	for ( uint64_t r = 0 ; r < RUNS ; ++r ) {
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
