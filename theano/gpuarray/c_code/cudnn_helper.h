#ifndef CUDNN_HELPER_H
#define CUDNN_HELPER_H

#include <cudnn.h>

#ifndef CUDNN_VERSION

#define CUDNN_VERSION -1
static inline int cudnnGetVersion() {
  return -1;
}
#endif



#endif
