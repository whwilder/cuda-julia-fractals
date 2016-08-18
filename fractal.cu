#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <cuda.h>
#include <cuComplex.h>


__global__ void testFunc(double centerX, double centerY, double scale, unsigned char *output, double const_real, double const_imag, unsigned short width, unsigned short height){

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
          //printf("[%d, %d : %d, %d]: real: %d  imag: %d  count: %d\n", blockIdx.x, threadIdx.x, x, y, r, i, count);
           break;
       }
   }
   //printf("[%d, %d]: %d\n", x, y, width*y+x);
   output[width*y + x] = (unsigned char) floor(((double)count/256)*255);
}

void write_tga(unsigned char *bitmap, char *fname, unsigned short width, unsigned short height){
    int datalen = 19 + width*height*3;
    unsigned char *data = (unsigned char*) calloc(1, datalen);
    data[2] = 2;
    memcpy(data+12, &width, 2);
    memcpy(data+14, &height, 2);
    data[16] = 24;
    unsigned char *offset = data+19;
    int i=0;
    for(int y=0;y<height;y+=1){
        for(int x=0;x<width;x+=1){
            offset[i] = bitmap[y*width + width-x-1];
            offset[i+1] = bitmap[y*width + width-x-1];
            offset[i+2] = bitmap[y*width + width-x-1];
            i += 3;
        }
    }
    FILE *file = fopen(fname,"wb");
    fwrite(data, 1, datalen, file);
    fclose(file);
    free(data);
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
    unsigned char *output;
    width = atof(argv[1]);
    height = atof(argv[2]);
    int size = width*height;
    output = (unsigned char*) calloc(sizeof(unsigned char), width*height);

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

    unsigned char *output_device;
    cudaError_t err;
    if( (err = cudaMalloc( (void **) &output_device, size)) != cudaSuccess)
      printf("Error:  could not malloc to device, code %d\n", cudaGetErrorString(err) );
    if( (err = cudaMemcpy(output_device, output, size, cudaMemcpyHostToDevice)) != cudaSuccess )
      printf("Error:  could not copy to device, code %s\n", cudaGetErrorString(err));

    int block_size = 64;
    int num_blocks = height*width/block_size;
    testFunc <<<num_blocks, block_size>>> ((double) centerX, (double) centerY, scale, output_device, const_real, const_imag, width, height);
    cudaMemcpy(output, output_device, size, cudaMemcpyDeviceToHost);

    if( (err = cudaFree(output_device)) != cudaSuccess )
      fprintf(stderr, "Could not free device \"output_device\" (err code: %s)\n", cudaGetErrorString(err));

    if ( (err = cudaDeviceReset()) != cudaSuccess ){
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // loop through pixels and generate intensity map

    write_tga(output, "fractal.tga", width, height);
    return 0;
}
