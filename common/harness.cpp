#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <assert.h>


#define WRITE_OUTPUT
#define CHECK_OUTPUT

#include "support.h"
#include <errno.h>
#include <limits.h>
#include "input_data.h"
#include "check_data.h"

int main(int argc, char **argv)
{
  // Parse command line.
  char *in_file;
  #ifdef CHECK_OUTPUT
  char *check_file;
  #endif
  //assert( argc<4 && "Usage: ./benchmark <input_file> <check_file>" );
  in_file = "/home/hackathon05/team05/spmv/crs/input.data";
  #ifdef CHECK_OUTPUT
  check_file = "/home/hackathon05/team05/spmv/crs/check.data";
  #endif

  // Load input data
  int in_fd;
  char *data = (char *) malloc(INPUT_SIZE);
  input_to_data(input_file, data);
  assert( data!=NULL && "Out of memory" );
  
  // Unpack and call
  run_benchmark( data );

  // Load check data
  #ifdef CHECK_OUTPUT
  int check_fd;
  char *ref = (char *) malloc(INPUT_SIZE);
  input_to_data(check_data_file, ref);
  #endif

  // Validate benchmark results
  #ifdef CHECK_OUTPUT
  if( !check_data(data, ref) ) {
    fprintf(stderr, "Benchmark results are incorrect\n");
    return -1;
  }
  else {
	  printf("Finished checking\n");
  }
  #endif

  printf("Success.\n");
  return 0;
}
