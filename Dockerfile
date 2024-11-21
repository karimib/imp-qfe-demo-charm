FROM ubuntu:18.04 as base
MAINTAINER support@charm-crypto.com

RUN apt update && apt install --yes build-essential flex bison wget subversion m4 python3 python3-dev python3-setuptools python3-numpy libgmp-dev libssl-dev
RUN wget https://crypto.stanford.edu/pbc/files/pbc-0.5.14.tar.gz && tar xvf pbc-0.5.14.tar.gz && cd /pbc-0.5.14 && ./configure LDFLAGS="-lgmp" && make && make install && ldconfig
COPY ./ext /charm
RUN cd /charm && ./configure.sh && make && make install && ldconfig


from base as testing 
COPY ./qfehelpers.py .
COPY ./test_qfehelpers.py .
RUN python3 test_qfehelpers.py

from testing as final 
COPY ./qfepuredemo.py .

#from base as final
#COPY ./qfedemo.py .
# Festlegen des Startbefehls
CMD ["python3", "qfepuredemo.py"]
#CMD ["python3", "qfedemo.py"]