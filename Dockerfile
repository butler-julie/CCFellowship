FROM python:3.8.2-buster

RUN apt-get update && \
        apt-get install -y --no-install-recommends sudo apt-utils && \

        apt-get install -y --no-install-recommends openssh-server \

        python-dev python-numpy python-pip python-virtualenv python-scipy \

        gcc gfortran libopenmpi-dev openmpi-bin openmpi-common openmpi-doc binutils && \

        apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
#COPY . /usr/src/app
COPY hello.py /usr/src/app/hello.py

CMD ["mpiexec","--allow-run-as-root","-n","2","python", "/usr/src/app/hello.py"]
