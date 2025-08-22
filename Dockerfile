# --- Base: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04 (H100-friendly) ---
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# OS deps
RUN apt-get update && apt-get install -y \
    wget git curl build-essential cmake unzip \
    hmmer mafft ncbi-blast+ python3-pip \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    openmpi-bin libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

# --- Miniconda ---
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# --- RF2-Lite source ---
RUN git clone https://github.com/lauramarie99/RF2-Lite.git /opt/RF2Lite
WORKDIR /opt/RF2Lite

# Channels + mamba
RUN conda config --system --set channel_priority strict && \
    conda config --system --add channels pytorch && \
    conda config --system --add channels nvidia && \
    conda config --system --add channels conda-forge && \
    conda install -n base -y -c conda-forge mamba

# --- Conda env: Torch 2.4.0 + CUDA 12.1 (H100 sm_90 supported) ---
RUN conda create -y -n RF2Lite python=3.10 && \
    mamba install -y -n RF2Lite -c pytorch -c nvidia -c conda-forge \
      pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 \
      "numpy>=1.26,<2.0" "pydantic>=1.10,<2.0" \
      cmake ninja setuptools wheel scipy pandas h5py biopython psutil tqdm && \
    conda clean -afy

# --- TorchData (matches torch 2.4) ---
RUN /bin/bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate RF2Lite && \
  pip install --no-cache-dir torchdata==0.8.0 json5 pyyaml"

# --- DGL for torch 2.4 + cu121 ---
RUN /bin/bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate RF2Lite && \
  pip install --no-cache-dir 'dgl==2.4.0' \
    -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html"

# --- PyTorch Geometric wheels for torch 2.4.0 + cu121 ---
RUN /bin/bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate RF2Lite && \
  pip install --no-cache-dir pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html && \
  pip install --no-cache-dir torch-geometric==2.6.1"

# --- Build SE3Transformer (compile for sm_90; do NOT let it reinstall torch) ---
RUN /bin/bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate RF2Lite && \
  cd /opt/RF2Lite/SE3Transformer && \
  grep -v -E '^(torch|torchvision|torchaudio)([=<>].*)?$' requirements.txt > req_no_torch.txt && \
  pip install --no-cache-dir -r req_no_torch.txt && \
  export TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' && \
  python setup.py install"

# --- RF2-Lite weights ---
RUN /bin/bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate RF2Lite && \
  cd /opt/RF2Lite/networks && \
  wget http://files.ipd.uw.edu/pub/pathogens/weights.tar.gz && \
  tar xfz weights.tar.gz && rm weights.tar.gz"

# --- DGL runtime env (GraphBolt not needed for RF2-Lite) ---
ENV DGLBACKEND=pytorch
ENV DGL_DISABLE_GRAPHBOLT=1

WORKDIR /opt/RF2Lite
CMD ["/bin/bash"]