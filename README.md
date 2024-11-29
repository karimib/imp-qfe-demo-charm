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



## Notes
Only quadratic functions that represent billinear forms are supported.
https://chatgpt.com/share/6749fede-fe00-8004-8f13-5fd2a755ea08


Benchmarking
-> First we tried by increasing the k value 
-> Then we had a look at the graphs and saw that in the beginning there were spikes in the setup and keygen method that we assume are because the group gets initialized new for every new k
-> We tougth about fixing the group and its elements to have a more clear benchmark 
