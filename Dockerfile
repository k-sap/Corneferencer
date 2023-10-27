FROM tensorflow/tensorflow:2.9.3-gpu

# Configure Poetry
ENV POETRY_VERSION=1.5.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    git

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    python3-pip \
    python3-venv \ 
    git

ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH

RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT

RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
    echo 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init --path)"\nfi' >> ~/.bashrc

RUN eval "$(pyenv init --path)" && \
    pyenv install 3.7.12

RUN eval "$(pyenv init --path)" && \
    pyenv global 3.7.12

WORKDIR /app

COPY . /app



ENV PATH="$PYENV_ROOT/shims:$PATH"

RUN python3 -m venv $POETRY_VENV \
   && $POETRY_VENV/bin/pip install --no-cache-dir -U pip setuptools \
   && $POETRY_VENV/bin/pip install --no-cache-dir poetry==${POETRY_VERSION}

ENV PATH="${PATH}:${POETRY_VENV}/bin"

RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "/app/corneferencer/main.py"]
