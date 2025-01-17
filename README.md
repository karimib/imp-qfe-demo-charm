# IMP-QFE

## Scheme

The following project implements the bounded Quadratic Functional Encryption Scheme as described on p.11 of the following paper

* [Practical Functional Encryption for Quadratic Functions with Applications to Predicate Encryption](https://eprint.iacr.org/2017/151.pdf)

## Structure

* [qfebounded.py](./qfebounded.py) contains the implementation of the scheme
* [qfehelpers.py](./qfehelpers.py) helper functions used by the scheme
* [benchmark.py](./benchmark.py) calls the scheme in different ways and provides benchmarks

## Prerequisites

* Docker or Podman (if using podman replace the "docker" with "podman" in the instructions below)

## Usage

Clone the repository

```shell
git clone https://github.com/karimib/imp-qfe-demo-charm.git
cd imp-qfe-demo-charm
```

Build the image (assuming your in the root directory)

```shell
docker build -t qfedemo:v1 .
```

Create a container from the image

```shell
docker run qfedemo:v1 
```

Mount a volume to save benchmark csv

````shell
docker run -v "${PWD}/results:/data" qfedemo:v1 
````
