#include <stdlib.h>

double drand48() {
  return rand() / (RAND_MAX + 1.0);
}

void srand48(long seed) {
  srand(seed);
}
