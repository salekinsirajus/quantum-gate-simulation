FROM pli11/gpgpusim:cuda_10_1

ENV foo 1
ADD . /code/

RUN "/bin/bash"
