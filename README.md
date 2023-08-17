
This repositository contains the code for the paper:

# HELiKs: HE Linear Algebra Kernels for Secure Inference

## SEAL version:
The SEAL version of the kernels are implemented in the EzPC framework. The EzPC framework is available at https://github.com/mpc-msri/EzPC. To build and run the experiments presented in the paper, please follow these instructions:
1. Build the docker image `heliks_seal` using `seal_Dockerfile`:
```
docker build -t heliks_seal . -f seal_Dockerfile
```
2. Run the tests, `seal_mm_tests.sh` and `seal_cv_tests.sh` for the matrix multiplication and convolution kernels respectively. The tests can be run using the following command:
```
./seal_mm_tests.sh
./seal_cv_tests.sh
```

## OpenFHE version:
The OpenFHE version of the kernels are still in development. Do not use them for any production purposes. The current version of the code provides an accurate estimate of runtime and communication costs but may not yield numerically correct results. The OpenFHE framework is available at https://github.com/openfheorg/openfhe-development. To build and run the experiments presented in the paper, please follow these instructions:
1. Build the docker image `heliks_ofhe` using `ofhe_Dockerfile`:
```
docker build -t heliks_ofhe . -f ofhe_Dockerfile
```
2. Run the tests, `ofhe_mm_tests.sh` and `ofhe_cv_tests.sh` for the matrix multiplication and convolution kernels respectively. The tests can be run using the following command:
```
./ofhe_mm_tests.sh
./ofhe_cv_tests.sh
```