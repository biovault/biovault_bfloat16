# biovault_bfloat16

A `bfloat16` implementation for projects of [BioVault](https://github.com/biovault) (Biomedical Visual Analytics Unit LUMC - TU Delft)

Originally based upon `dnnl::impl::bfloat16_t` from the [Deep Neural Network Library (DNNL)](https://github.com/intel/mkl-dnn) of Intel Corporation:
* https://github.com/intel/mkl-dnn/blob/v1.2/src/cpu/bfloat16.cpp
* https://github.com/intel/mkl-dnn/blob/v1.2/src/common/bfloat16.hpp

## References:

* Intel, [BFLOAT16 – Hardware Numerics Definition", White Paper, November 2018, Revision 1.0 Document Number: 338302-001US](https://software.intel.com/sites/default/files/managed/40/8b/bf16-hardware-numerics-definition-white-paper.pdf)
  https://software.intel.com/en-us/download/bfloat16-hardware-numerics-definition

* Wikipedia [bfloat16 floating-point format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
  [current revision, as edited at 19:51, 30 December 2019 (Updating to Intel latest releases).](https://en.wikipedia.org/w/index.php?title=Bfloat16_floating-point_format&oldid=933243816)

* John D. Cook, 15 November 2018, [Comparing bfloat16 range and precision to other 16-bit numbers](https://www.johndcook.com/blog/2018/11/15/bfloat16)

* [`tensorflow::bfloat16` from TensorFlow](https://github.com/tensorflow/tensorflow/tree/v2.1.0/tensorflow/core/lib/bfloat16)

* Shibo Wang, Pankaj Kanwar, August 23, 2019 [BFloat16: The secret to high performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

## Related issues:

* DNNL: [Prevent constructing signaling NaNs and denormals (subnormal floats) by bfloat16_t](https://github.com/intel/mkl-dnn/pull/649)

* DNNL: [Avoid undefined behavior (UB) bfloat16_t by removing type-pun via union](https://github.com/intel/mkl-dnn/pull/646) (Closed)

* TensorFlow: [bfloat16 does not flush denormals (subnormal floats) to zero](https://github.com/tensorflow/tensorflow/issues/36514)
 
* Visual C++: [Signaling NaN (float, double) becomes quiet NaN when returned from function (both x86 and x64)](https://developercommunity.visualstudio.com/content/problem/903305/signaling-nan-float-double-becomes-quiet-nan-when.html)