
This repositository contains the code for the paper:

# HELiKs: HE Linear Algebra Kernels for Secure Inference

## SEAL version:
The SEAL version of the kernels are implemented in the EzPC framework. The EzPC framework is available at https://github.com/mpc-msri/EzPC. To build and run the experiments presented in the paper, please follow these instructions:
1. Build the docker image `heliks_seal` using `heliks_seal_Dockerfile`:
```
docker build -t heliks_seal . -f heliks_Dockerfile
```
2. Run the tests, `mm_tests_seal.sh` and `cv_tests_seal.sh` for the matrix multiplication and convolution kernels respectively. The tests can be run using the following command:
```
./mm_tests_seal.sh
./cv_tests_seal.sh
```

## OpenFHE version:
The OpenFHE version of the kernels are still in development. Do not use them for any production purposes. The current version of the code provides an accurate estimate of runtime and communication costs but may not yield numerically correct results. The OpenFHE framework is available at https://github.com/openfheorg/openfhe-development. To build and run the test presented in the paper, please follow these instructions:
1. Build the docker image `heliks_ofhe` using `heliks_ofhe_Dockerfile`:
```
docker build -t heliks_ofhe . -f heliks_ofhe_Dockerfile
```
2. Run the tests, `mm_tests_ofhe.sh` and `cv_tests_ofhe.sh` for the matrix multiplication and convolution kernels respectively. The tests can be run using the following command:
```
./mm_tests_ofhe.sh
./cv_tests_ofhe.sh
```