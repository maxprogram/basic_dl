FROM continuumio/miniconda3

# Update conda
RUN conda update -n base conda

# Hack to avoid re-installing environment every time
WORKDIR /basic_dl
ADD ./environment.yml ./environment.yml

# Load conda environment
RUN conda env create
RUN echo "source activate basic_dl" > ~/.bashrc
ENV PATH /opt/conda/envs/basic_dl/bin:$PATH

# Copy source
ADD ./ /basic_dl

EXPOSE 8888

CMD [ "/bin/bash" ]
