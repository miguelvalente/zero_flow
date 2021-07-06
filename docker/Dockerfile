FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Zet multiverse aan bij weghalen #
RUN cat /etc/apt/sources.list | sed -e '/multiverse/ s/\#//' > /etc/apt/sources2.list
RUN mv /etc/apt/sources2.list /etc/apt/sources.list

# Install all apt-get packages. Merendeel nodig voor anaconda
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion libbz2-dev libz-dev libpng-dev
RUN apt-get install p7zip-full
# Anaconda installeren
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN conda install nodejs

RUN pip install --ignore-installed tensorflow-gpu keras
RUN pip install --force-reinstall --no-deps h5py
RUN pip install torch wandb transformers matplotlib numpy sentencepiece
#RUN conda install -c conda-forge jupyterlab
RUN conda install -c conda-forge jupytext
# ADD environment.yml /data/simoesvalentema/zero_flow/environment.yml
# RUN conda env create -f /data/simoesvalentema/zero_flow/environment.yml

#RUN jupyter labextension install jupyterlab-jupytext

#RUN jupyter labextension install @jupyterlab/git
#RUN pip install --upgrade jupyterlab-git
#RUN jupyter serverextension enable --py jupyterlab_git

# Install ssh server, used by PyCharm for remote execution
RUN apt-get update && \
    apt-get -y install openssh-server && \
    mkdir -p /run/sshd && \
    sed -i -e 's/^UsePAM/#UsePam/' /etc/ssh/sshd_config
#RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 22
# Depending on the user-namespace you are running, you need to add an internal user to docker image
ARG USERID
ARG GROUPID
ARG USERNAME
ARG GROUPNAME
RUN grep -q '^'$USERNAME':' /etc/passwd ||\
    echo "$USERNAME:x:$USERID:$GROUPID::/root" >> /etc/passwd; \
    grep -q '^'$GROUPNAME':' /etc/group ||\
    echo "$GROUPNAME:x:$GROUPID:" >> /etc/group

