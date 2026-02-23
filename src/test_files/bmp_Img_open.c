#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define INPUT_SIZE 28
#define POOL_SIZE 2
#define OUTPUT_SIZE (CONVOLT_OUTPUT_SIZE / POOL_SIZE)
#define CONVOLT_OUTPUT_SIZE (INPUT_SIZE - POOL_SIZE)

#pragma pack(push, 1)  // Ensure no padding

typedef struct {
    unsigned short bfType;      // BMP signature
    unsigned int bfSize;        // File size
    unsigned short bfReserved1; // Reserved
    unsigned short bfReserved2; // Reserved
    unsigned int bfOffBits;     // Offset to pixel data
} BITMAPFILEHEADER;

typedef struct {
    unsigned int biSize;          // Header size
    int biWidth;                  // Width
    int biHeight;                 // Height
    unsigned short biPlanes;      // Color planes
    unsigned short biBitCount;    // Bits per pixel
    unsigned int biCompression;   // Compression
    unsigned int biSizeImage;     // Image size
    int biXPelsPerMeter;          // X pixels per meter
    int biYPelsPerMeter;          // Y pixels per meter
    unsigned int biClrUsed;       // Colors used
    unsigned int biClrImportant;  // Important colors
} BITMAPINFOHEADER;

#pragma pack(pop)

uint32_t littleToBigEndian(uint32_t little) {
	return ((little >> 24) & 0x000000FF) |
		   ((little >> 8)  & 0x0000FF00) |
		   ((little << 8)  & 0x00FF0000) |
		   ((little << 24) & 0xFF000000);
}
void writeBMP(const char *filename, double image[OUTPUT_SIZE][OUTPUT_SIZE]) {
    FILE *f = fopen(filename, "wb");

    // Calculate row size with padding (each row size must be a multiple of 4)
    int rowSizeWithoutPadding = OUTPUT_SIZE * 3; // 3 bytes per pixel (24-bit BMP)
    int paddingSize = (4 - (rowSizeWithoutPadding % 4)) % 4; // Calculate padding needed to make row size a multiple of 4
    int rowSize = rowSizeWithoutPadding + paddingSize; // Row size including padding
    int imageSize = rowSize * OUTPUT_SIZE; // Total image size

    // Create BMP file and info headers
    BITMAPFILEHEADER fileHeader = {0x4D42, 54 + imageSize, 0, 0, 54};
    BITMAPINFOHEADER infoHeader = {40, OUTPUT_SIZE, OUTPUT_SIZE, 1, 24, 0, imageSize, 0, 0, 0, 0};

    fwrite(&fileHeader, sizeof(fileHeader), 1, f);
    fwrite(&infoHeader, sizeof(infoHeader), 1, f);

    // Write pixel data (BGR format)
    int i, j;
    unsigned char padding[3] = {0, 0, 0}; // Maximum padding of 3 bytes (since 4 - 1 = 3)
    for (i = OUTPUT_SIZE - 1; i >= 0; i--) { // BMP stores pixels bottom-to-top
        for (j = 0; j < OUTPUT_SIZE; j++) {
            unsigned char pixel = (unsigned char)image[i][j];  // Grayscale value
            fwrite(&pixel, 1, 1, f);  // Blue
            fwrite(&pixel, 1, 1, f);  // Green
            fwrite(&pixel, 1, 1, f);  // Red
        }
        // Add padding for each row
        fwrite(padding, 1, paddingSize, f);
    }

    fclose(f);
}

void maxPooling(double input[CONVOLT_OUTPUT_SIZE][CONVOLT_OUTPUT_SIZE], double output[OUTPUT_SIZE][OUTPUT_SIZE]) {
	for ( int y = 0 ; y < CONVOLT_OUTPUT_SIZE; y += POOL_SIZE ) {
		for ( int x = 0 ; x < CONVOLT_OUTPUT_SIZE; x += POOL_SIZE ) {
			double biggest = -99999;		
			for ( int y_pool = 0 ; y_pool < POOL_SIZE ; ++y_pool ) {
				for ( int x_pool = 0 ; x_pool < POOL_SIZE ; ++x_pool ) {
					if (input[y + y_pool][x + x_pool] > biggest) {
						biggest = input[y + y_pool][x + x_pool];
					}		
				}		
			}
			output[y/POOL_SIZE][x/POOL_SIZE] = biggest;
		}		
	}		
}		

void loadOneImg(const char* filePath, double inputs[INPUT_SIZE][INPUT_SIZE]) {

	FILE* fptr = fopen(filePath, "rb");
	if (!fptr) {
		printf("File does not exist\n");
		return;
	}		
	int magic, num_sample;
	int rows, cols;

	/*             image info            */
	fread(&magic, 4, 1, fptr);
	fread(&num_sample, 4, 1, fptr);
	fread(&rows, 4, 1, fptr);
	fread(&cols, 4, 1, fptr);

	magic = littleToBigEndian(magic);
	num_sample = littleToBigEndian(num_sample);
	rows = littleToBigEndian(rows);
	cols = littleToBigEndian(cols);

	unsigned char buffer[INPUT_SIZE * INPUT_SIZE]; 
	fread(buffer, INPUT_SIZE * INPUT_SIZE, 1, fptr);
	for ( int i = 0 ; i < INPUT_SIZE ; ++i ) {
		for ( int j = 0 ; j < INPUT_SIZE ; ++j ) {
			//inputs[i][j] = (double)((int)buffer[i * INPUT_SIZE + j]/(double)255);
			inputs[i][j] = (double)((int)buffer[i * INPUT_SIZE + j]);
			//inputs[i][j] = ((double)buffer[i][j]/255);
			printf("%.2lf ", inputs[i][j]);
		}		
		// printf("\n");
	}		

	fclose(fptr);
}		

void convolute( double picture[INPUT_SIZE][INPUT_SIZE], int rows, int cols,
 					double filter[][3], int f_rows, int f_cols, double output[CONVOLT_OUTPUT_SIZE][CONVOLT_OUTPUT_SIZE] ) {
	int fmapColSize = cols - f_cols + 1;	// assume kernel = 3x3, fmapColSize = 26
	int fmapRowSize = rows - f_rows + 1;
	for ( int y = 0 ; y < fmapRowSize ; ++y ) {
		for ( int x = 0 ; x < fmapColSize ; ++x ) {
			for ( int y_filter = 0 ; y_filter < f_rows ; ++y_filter) {
				for ( int x_filter = 0 ; x_filter < f_cols ; ++x_filter) {
					output[y][x] += picture[y + y_filter][x + x_filter] * filter[y_filter][x_filter];	
				}		
			}		
		}		
	}
}		

double reLU(double x) {
	return (x > 0.00) ? x : 0.00;
}		

void reLU_pic(double pooled_pic[OUTPUT_SIZE][OUTPUT_SIZE]) {
	for ( int i = 0 ; i < OUTPUT_SIZE ; ++i ) {
		for ( int j = 0 ; j < OUTPUT_SIZE ; ++j ) {
			pooled_pic[i][j] = reLU(pooled_pic[i][j]);
		}			
	}			
}		

int main() {

	double inputs[INPUT_SIZE][INPUT_SIZE];
	loadOneImg("C:\\Users\\Benson Ling\\Desktop\\CNN\\MNIST\\train-images-idx3-ubyte\\train-images-idx3-ubyte", inputs);
	double convolt_output[CONVOLT_OUTPUT_SIZE][CONVOLT_OUTPUT_SIZE] = {0.00};
    double output[OUTPUT_SIZE][OUTPUT_SIZE] = {0.00}; // Output for pooled image
    // maxPooling(inputs, output);  // Perform max pooling

	double kernels[3][3] = {
			// Diagonal edge detection (45)
				{-1, 0, 1},
				{-2, 0, 2},
				{-1, 0, 1}
			};

	convolute(inputs,INPUT_SIZE, INPUT_SIZE, 
			  kernels, 3, 3, convolt_output);
	maxPooling(convolt_output, output);
	reLU_pic(output);

	printf("==========output==========\n");
	for ( int i = 0 ; i < OUTPUT_SIZE ; ++i ) {
		for ( int j = 0 ; j < OUTPUT_SIZE ; ++j ) {
			printf("%lf ", output[i][j]);
		}		
		printf("\n");
	}

    // Write pooled image to BMP file
    writeBMP("C:\\Users\\Benson Ling\\Desktop\\CNN\\45_line_detection.bmp", output);
    printf("Pooled image saved to pooled_image.bmp\n");

    return 0;
}		
