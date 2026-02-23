#include <stdio.h>
#include <stdlib.h>

// Define the size of the input and output image
#define INPUT_SIZE 28
#define POOL_SIZE 2
#define OUTPUT_SIZE (INPUT_SIZE / POOL_SIZE)

// Max pooling function
void maxPooling(int input[INPUT_SIZE][INPUT_SIZE], int output[OUTPUT_SIZE][OUTPUT_SIZE]) {
	int i, j, m, n;
    for (i = 0; i < INPUT_SIZE; i += POOL_SIZE) {
        for (j = 0; j < INPUT_SIZE; j += POOL_SIZE) {
            int maxVal = input[i][j];
            for (m = 0; m < POOL_SIZE; m++) {
                for (n = 0; n < POOL_SIZE; n++) {
                    if (input[i + m][j + n] > maxVal) {
                        maxVal = input[i + m][j + n];
                    }
                }
            }
            output[i / POOL_SIZE][j / POOL_SIZE] = maxVal;
        }
    }
}

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

void writeBMP(const char *filename, int image[OUTPUT_SIZE][OUTPUT_SIZE]) {
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

int main() {
    // Example input image (8x8 matrix with values 0-255)
    int input[28][28] ={{ 48, 125,   1,  42, 167, 219, 142, 148,  92, 137, 224, 245,  78, 223,  92, 115, 179, 158, 127, 229,  63, 147, 131,  79,  20, 247, 113, 138, },
{235,  16, 218,  27, 141, 220,  69,  52, 183, 212, 200,  19,  93, 169,   8, 171, 136, 101,  30,  59,   3, 157,  32,  67,  49, 164, 146,  69, },
{155,   3, 207, 134,  19, 170, 162, 160, 134, 231, 212,  61, 187, 157,  80,  24,  70,  89, 195, 206, 190, 226,  10, 193, 127,  42,   4, 176, },
{206, 150, 245, 106, 153, 197, 240, 172, 111, 146,  76, 245, 122,  33,  50,  53, 190, 130,  78,   4, 219,  17, 210, 153, 243, 220,  91, 115, },
{  7,  95,  35, 213, 246,  25,  63, 143, 222,  48,  60,  77, 194, 136,  66,  60, 169, 116, 114, 103, 246, 192, 107, 210, 209,  62, 107, 197, },
{ 26, 198,  56,  33,  38,  91, 247,  28, 116,  54, 171,  82, 102, 231, 159,  41, 112, 225, 101,  25,  85, 215, 129,  76, 151, 236,  30, 105, },
{ 42, 137,  46,  69,  80, 102, 102, 118, 193,  93, 146,  54, 148,  61, 136, 250,  37,  40,  35, 149,   9, 137, 174,  95,  96,  47, 171, 248, },
{ 28, 201,  97,  70,  82, 143, 139, 162, 245, 242,  24, 182,  79, 170, 236, 227, 232, 117, 222,  13, 157,   1, 162, 166, 138,  80,   5, 235, },
{128, 176, 227, 156, 121,  68, 226, 204, 211, 110, 110, 200,  96, 135, 126, 175,  49, 107, 147,  25, 224, 113,  38, 125, 114, 200,  35, 253, },
{ 25,  41, 232, 153, 217, 203,  53,  83,  15,  23,  31, 226, 133, 141, 170, 229,  20,  40, 149,  70, 147,  40,  95, 115, 153, 134, 240,  11, },
{ 78,  20,   8, 103,  61, 240,   0,  22, 187,  53, 105, 202,  77, 136, 172, 210,  22,  86, 184,  42, 127,  77, 112,  18, 117, 208, 134,  14, },
{ 86, 118,  25, 164, 138,  34,  12, 199,  18,  12, 222, 206,  66,  71, 152, 143, 208,  69,  97, 230, 155,  25,  16,  26, 102, 129,  45, 219, },
{ 81, 179, 233, 167,  41,   3,  75, 180,  37,  87, 123,  55, 100,  89,   5, 166, 161, 158,  53, 113, 227, 150,  87, 126, 176, 103, 153,  22, },
{232, 198, 242,  57, 121, 219, 224, 162, 222,  44,  86,   3, 131, 210,  59, 231,  43,  64, 141, 204, 222, 194,  61, 193,  89, 148,  64,   9, },
{252, 217,  31, 228, 159,  17,  30,  24, 237, 254, 186, 203,  42,  17, 207, 174, 227,  10, 149,  14,  74,  35, 219,  41, 229,  24, 234,  62, },
{173,  42,  71, 169,   3, 103, 141, 162, 120, 171, 186, 101, 170, 117,  49, 212, 134,   0, 130, 105,  10,  24, 119,  84,  59,  82, 125,  32, },
{107, 104,  95,  24, 146, 166, 193, 150,  13,  78,  56, 134, 250, 243, 235, 164, 104,  28, 120, 238,  28, 251,  87,  38,  19, 206, 123,  78, },
{ 33, 248, 110, 140,  96, 205, 164, 243, 116, 101, 137, 129, 179, 193,   7, 173, 180, 243,  81,  28,  15, 202,  10,  44, 197,  97,  82, 216, },
{ 48, 205,  38,  81, 198, 148, 221,  38,  98, 129,  25, 214, 230, 162,  87, 153, 100,  95,  71,  24,  82, 152,  53,  97,  98,  63, 141,  39, },
{161, 224, 255, 209, 173,  37,  34, 115, 186, 255, 154,  28, 128, 179, 242, 102,  86,  73, 255, 186, 168,  70, 210, 250, 223,   7,  92,  65, },
{ 71, 233, 105, 232, 201, 104, 185, 119, 142, 219, 234,  72, 218, 132, 100,  90,  56,  86, 192, 142, 159, 191,  72,  72,   6,  26,  66, 229, },
{ 34, 158,  38, 105, 136, 143,  81,  81, 248,  10, 200, 134, 229, 179, 206, 191,  55,  50,  25, 111, 136, 217, 253,  39, 152,  69, 111, 158, },
{ 96, 178, 131, 130,  80, 170, 235, 216,  57,  60,  42,  49,  70, 242, 183,  43, 165, 133, 234, 221, 183,   3,  76,  63, 220,  74, 103, 116, },
{143, 214,  19, 239, 136, 150, 113, 217,  64,  92, 177, 122, 152, 219, 171, 222, 206,  99,   9, 115, 232, 243,  80, 160, 246, 157, 223, 210, },
{231,  70,  71, 118,  29,  90, 102, 165, 240, 215, 126,  49,  52,  48, 171, 204,  11,  86, 171, 217, 185, 180,  77, 162, 168, 157,  66, 158, },
{ 58,  33, 113,  33, 104, 184, 152, 133,  18, 254,  42,   2, 213, 169,  51,   9, 217, 222, 214, 228,  53, 129, 190, 238,  53,  11, 144, 221, },
{168, 210, 124, 227, 244, 237,   4,  92, 165, 156, 225, 183, 154,  11, 185, 112, 180, 237, 121, 141, 203,  79, 114,   0, 208,  48, 239,   6, },
{ 59, 127, 227, 227,  82,  95, 198,  70,  76, 203, 162, 241, 103, 131, 168,   2, 142,  98, 114,  67,  79, 235, 208,  26,  59,  66,  27,  11, }};

    int output[OUTPUT_SIZE][OUTPUT_SIZE] = {0}; // Output for pooled image
    maxPooling(input, output);  // Perform max pooling

    // Write pooled image to BMP file
    writeBMP("pooled_image.bmp", output);
    printf("Pooled image saved to pooled_image.bmp\n");

    return 0;
}
