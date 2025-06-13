FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime AS vcs_base_image_classification

ENV DEBIAN_FRONTEND=noninteractive 
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y \
        bash \
        build-essential \
        git \
        curl \
        ca-certificates \
        python3 \
        python3-pip \
        python3-opencv \
    && rm -rf /var/lib/apt/lists

ENV WORKDIR="/app"
WORKDIR ${WORKDIR}

ADD https://astral.sh/uv/0.7.12/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ADD ./pyproject.toml ${WORKDIR}/pyproject.toml
ADD ./uv.lock ${WORKDIR}/uv.lock
RUN uv sync --locked

ENTRYPOINT ["uv", "run", "-m", "src.main"]
