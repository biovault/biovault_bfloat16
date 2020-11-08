# biovault_bfloat16

A `bfloat16` implementation for projects of [BioVault](https://github.com/biovault) (Biomedical Visual Analytics Unit LUMC - TU Delft)

Originally based upon `dnnl::impl::bfloat16_t` from the [Deep Neural Network Library (DNNL)](https://github.com/intel/mkl-dnn) of Intel Corporation.
Updated to version 1.7 of their library, which has been renamed to [oneAPI Deep Neural Network Library (oneDNN)](https://github.com/oneapi-src/oneDNN).
* https://github.com/oneapi-src/oneDNN/blob/v1.7/src/cpu/bfloat16.cpp
* https://github.com/oneapi-src/oneDNN/blob/v1.7/src/common/bfloat16.hpp

Other consulted implementations: [`tensorflow::bfloat16`](https://github.com/tensorflow/tensorflow/tree/v2.2.0/tensorflow/core/lib/bfloat16) and [`Eigen::bfloat16`](https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/arch/Default/BFloat16.h)

## References:

* Intel&reg;, [BFLOAT16 – Hardware Numerics Definition", White Paper, November 2018, Revision 1.0 Document Number: 338302-001US](https://software.intel.com/sites/default/files/managed/40/8b/bf16-hardware-numerics-definition-white-paper.pdf)
  https://software.intel.com/en-us/download/bfloat16-hardware-numerics-definition

* Wikipedia [bfloat16 floating-point format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
  (The consulted revision was last edited at [20:39, 18 May 2020](https://en.wikipedia.org/w/index.php?title=Bfloat16_floating-point_format&oldid=957432439))

* John D. Cook, 15 November 2018, [Comparing bfloat16 range and precision to other 16-bit numbers](https://www.johndcook.com/blog/2018/11/15/bfloat16)

* Shibo Wang, Pankaj Kanwar, August 23, 2019 [BFloat16: The secret to high performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

* Intel&reg;, March 2020, [Architecture Instruction Set Extensions and Future Features, Programming Reference](https://software.intel.com/sites/default/files/managed/c5/15/architecture-instruction-set-extensions-programming-reference.pdf)

## Reported related issues:

* oneDNN pull request #646, Feb 3, 2020: [Avoid undefined behavior (UB) bfloat16_t by removing type-pun via union](https://github.com/oneapi-src/oneDNN/pull/646) (Closed,
fixed by Eugene Chereshnev (@echeresh), commit ff67087, Feb 4, 2020: [all: introduce and use utils::bit_cast() for safe type punning](https://github.com/oneapi-src/oneDNN/commit/ff670873307ed66a25a663181d3bff45d3e6469f))

* oneDNN pull request #649, Feb 10, 2020: [Prevent constructing signaling NaNs and denormals (subnormal floats) by bfloat16_t](https://github.com/oneapi-src/oneDNN/pull/649) (Closed with unmerged commit)

* oneDNN pull request #780, Jul 17, 2020: [bfloat16: add operator+=(a) overload for float as argument type](https://github.com/oneapi-src/oneDNN/pull/780) (Merged:
[common: bfloat16: replace operator+=(bfloat16_t) by operator+=(float)](https://github.com/oneapi-src/oneDNN/commit/d68271b7474f3a7e427d4637cef835592688be52), and
[gtests: internals: add bfloat16_t tests](https://github.com/oneapi-src/oneDNN/commit/49dcdb1d76bc43c058380f5697b75e2f7f2552a6), committed by Evarist M Fomenko (@emfomenk))

* oneDNN pull request #790, Jul 25, 2020: [all: bfloat16: initialize variables by T{} instead of integer constant 0](https://github.com/oneapi-src/oneDNN/pull/790) (Closed with unmerged commits)

* oneDNN pull request #801, Aug 10, 2020: [bfloat16: faster conversion and assignment from integer types](https://github.com/oneapi-src/oneDNN/pull/801) (Merged by Evarist M Fomenko (@emfomenk): commit
[073e346](https://github.com/oneapi-src/oneDNN/commit/073e3466cf82ddfd39c4be7ce21c814697b16565)

* TensorFlow issue #36514, Feb 6, 2020: [bfloat16 does not flush denormals (subnormal floats) to zero](https://github.com/tensorflow/tensorflow/issues/36514) (Closed,
resolved by @tensorflower-gardener, commit b04c4e0, Mar 20, 2020: [Flush denormals to +/- 0 when converting float to bfloat16](https://github.com/tensorflow/tensorflow/commit/b04c4e0e4338924d5281626445594a900bd673a6)) 

* TensorFlow pull request #41070, Jul 4, 2020: [Avoid undefined behavior by union type punning in round_to_bfloat16](https://github.com/tensorflow/tensorflow/pull/41070) (Closed with unmerged commits)

* Eigen merge request !163, Jul 11, 2020: [Allow implicit conversion from bfloat16 to float and double](https://gitlab.com/libeigen/eigen/-/merge_requests/163) (Merged:
commit [4ab32e2d](https://gitlab.com/libeigen/eigen/-/commit/4ab32e2de2511746e2108563a43cbbeb1922fbf2)) 

* Eigen merge request !164, Jul 14, 2020: [Avoid undefined behavior by union type punning in float_to_bfloat16_rtne](https://gitlab.com/libeigen/eigen/-/merge_requests/164) (Merged:
commit [b11f817b](https://gitlab.com/libeigen/eigen/-/commit/b11f817bcff04276f3024d6780f56a137968b81a))

* Eigen merge request !166, Jul 14, 2020: [Faster conversion from integer types to bfloat16](https://gitlab.com/libeigen/eigen/-/merge_requests/166) (Merged:
commit [0e1a33a4](https://gitlab.com/libeigen/eigen/-/commit/0e1a33a4617e605fbe93caf1ee51d1ef0f7cc29d)) 

* Eigen issue #1954, Jul 24, 2020: [Disagreement Eigen::bfloat16 and Intel on conversion from NaN to bfloat16](https://gitlab.com/libeigen/eigen/-/issues/1954)

* Visual C++ problem 903305, Feb 2, 2020: [Signaling NaN (float, double) becomes quiet NaN when returned from function (both x86 and x64)](https://developercommunity.visualstudio.com/content/problem/903305/signaling-nan-float-double-becomes-quiet-nan-when.html)