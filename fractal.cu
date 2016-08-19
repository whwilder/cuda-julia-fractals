#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <cuda.h>
#include <cuComplex.h>
#include <png.h>
#include <time.h>


__global__ void gen_fractal(double centerX, double centerY, double scale, unsigned int *output, double const_real, double const_imag, unsigned short width, unsigned short height){

   int x = (blockIdx.x * blockDim.x + threadIdx.x) % width;
   int y = (blockIdx.x * blockDim.x + threadIdx.x) / width;
   //printf("%d %d %d\n", blockIdx.x, blockDim.x, threadIdx.x);

   double a = (x - centerX) / (width * scale);
   double b = (y - centerY) / (width * scale);
   cuDoubleComplex z = make_cuDoubleComplex(a, b);
   int r;
   int i;
   
   int count;
   for( count=0; count<256; count++ ){
       z = cuCadd( cuCmul(z,z) , make_cuDoubleComplex(const_real, const_imag) );
       r = abs(cuCreal(z)+1);
       i = abs(cuCimag(z)+1);
       if( r > 2 || i > 2 ){
          //printf("[%d, %d]: real: %d  imag: %d  count: %d\n", x, y, r, i, count);
           break;
       }
   }
   //printf("[%d, %d]: %d\n", x, y, width*y+x);
   output[width*y + x] = (unsigned int) floor(((double)count/256.0)*768);
   //printf("[%d, %d]: real: %d  imag: %d  count: %d  val: %d\n", x, y, r, i, count, output[width*y+x]);
}

void write_tga(unsigned int *bitmap, char *fname, unsigned short width, unsigned short height){
    int datalen = 19 + width*height*3;
    unsigned int *data = (unsigned int*) calloc(1, datalen);
    data[2] = 2;
    memcpy(data+12, &width, 2);
    memcpy(data+14, &height, 2);
    data[16] = 24;
    unsigned int *offset = data+19;
    int i=0;
    for(int y=0;y<height;y+=1){
        for(int x=0;x<width;x+=1){
            offset[i] = bitmap[y*width + width-x-1];
            i += 1;
        }
    }
    FILE *file = fopen(fname,"wb");
    fwrite(data, 1, datalen, file);
    fclose(file);
    free(data);
}

inline void setRGB( png_byte *ptr, int val){
   int offset = val % 256;
   if( val < 256 ){
      ptr[0] = 0; ptr[1] = 0; ptr[2] = offset;
   }
   else if( val < 512 ){
      ptr[0] = 0; ptr[1] = offset; ptr[2] = 255-offset;
   }
   else{
      ptr[0] = offset; ptr[1] = 255-offset; ptr[2] = 0;
   }
   //ptr[0] = val >> 16; ptr[1] = (val >> 8) % 256; ptr[2] = 0;
}

void write_png( unsigned int *bitmap, char *fname, unsigned short width, unsigned short height ){
   FILE *fp = NULL;
   png_structp png_ptr = NULL;
   png_infop info_ptr = NULL;
   png_bytep row = NULL;

   fp = fopen(fname, "wb");

   // Initialize write structure
   png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
   
   // Initialize info structure
   info_ptr = png_create_info_struct(png_ptr);

   // Setup Exception handling
   if (setjmp(png_jmpbuf(png_ptr))) {
      fprintf(stderr, "Error during png creation\n");
      goto finalize;
   }
   png_init_io(png_ptr, fp);
   // Write header (8-bit color depth)
   png_set_IHDR(png_ptr, info_ptr, width, height,
         8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
         PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

   png_write_info(png_ptr, info_ptr);


   // Allocate memory for one row (3 bytes per pixel - RGB)
   row = (png_bytep) malloc(3 * width * sizeof(png_byte));

   // Write image data
   int x, y;
   for (y=0 ; y<height ; y++) {
      for (x=0 ; x<width ; x++) {
         //printf("[%d, %d]:  %d\n", x, y, bitmap[y*width+x]);
         setRGB(&(row[x*3]), bitmap[width*y + x]);
      }
      png_write_row(png_ptr, row);
   }

   // End write
   png_write_end(png_ptr, NULL);

   finalize:
   if (fp != NULL) fclose(fp);
   if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
   if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
   if (row != NULL) free(row);
}


int main(int argc, char **argv){
    if( argc < 3 ){
        printf("usage: %s width height a b scale\n", argv[0]);
        printf("a b are constants in f(z) = z^2 + a + bi\n");
        printf("scale adjusts the zoom of the fractal. start with 0.25.\n");
        printf("outputs to fractal.tga\n");
        return 1;
    }
    unsigned short width;
    unsigned short height;

    // prepare TGA for writing
    unsigned int *output;
    width = atof(argv[1]);
    height = atof(argv[2]);
    int size = width*height;
    output = (unsigned int*) calloc(sizeof(unsigned int), width*height);

    // declare variables
    int centerX = width/2;
    int centerY = height/2;
    //int iterations = 128;

    double scale = 0.4;
    if( argc == 6 ) scale = atof(argv[5]);

    double const_real = -0.221;
    double const_imag = -0.713;
    if( argc >= 5 ){ const_real = atof(argv[3]); const_imag = atof(argv[4]);}

    printf("Using function: f(z) = z^2 + %f + %fi\n", const_real, const_imag);

    unsigned int *output_device;
    cudaError_t err;
    if( (err = cudaMalloc( (void **) &output_device, size * sizeof(unsigned int))) != cudaSuccess)
      printf("Error:  could not malloc to device, code %d\n", cudaGetErrorString(err) );
    if( (err = cudaMemcpy(output_device, output, size, cudaMemcpyHostToDevice)) != cudaSuccess )
      printf("Error:  could not copy to device, code %s\n", cudaGetErrorString(err));

    int block_size = 64;
    int num_blocks = height*width/block_size;

    clock_t start = clock(), diff;
    gen_fractal <<<num_blocks, block_size>>> ((double) centerX, (double) centerY, scale, output_device, const_real, const_imag, width, height);
    cudaMemcpy(output, output_device, size*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    diff = clock() - start;

    int msec = diff * 1000 / CLOCKS_PER_SEC;

    printf("Fractal generated in %d seconds %d milliseconds\n", msec/1000, msec%1000);

    if( (err = cudaFree(output_device)) != cudaSuccess )
      fprintf(stderr, "Could not free device \"output_device\" (err code: %s)\n", cudaGetErrorString(err));

    if ( (err = cudaDeviceReset()) != cudaSuccess ){
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // loop through pixels and generate intensity map

    start = clock();
    write_png(output, "fractal.png", width, height);
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Created PNG in %d seconds %d milliseconds\n", msec/1000, msec%1000);
    //write_tga(output, "fractal.tga", width, height);
    return 0;
}
