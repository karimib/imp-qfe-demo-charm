# Implementation of a quadratic functional encryption scheme

## Scheme

The scheme described on P. 11 in the following scheme has been implemented

* [Practical Functional Encryption for Quadratic Functions with Applications to Predicate Encryption](https://eprint.iacr.org/2017/151.pdf)

## Prerequisites

* Docker or Podman

## Usage

Clone the repository

```shell
git clone https://github.com/karimib/imp-qfe-demo-charm.git
cd imp-qfe-demo-charm
```

Build the image (assuming your in the root directory)

```shell
docker build -t charmdemo:v1 .
```

Create a container from the image

```shell
docker run charmdemo:v1 
```
