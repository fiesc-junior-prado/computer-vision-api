# 1. Usa uma imagem Python oficial leve
FROM python:3.12-slim

# 2. Copia o binário do uv diretamente da imagem oficial da Astral
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Instala dependências do sistema necessárias para o OpenCV e ONNXRuntime
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Cria um usuário não-root
RUN useradd -m -u 1000 user

# 5. Muda para o usuário criado e ajusta variáveis de ambiente
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=10000

# 6. Define o diretório de trabalho dentro da pasta do usuário
WORKDIR $HOME/app

# 7. Copia os arquivos de configuração do uv dando a posse para o usuário "user"
COPY --chown=user pyproject.toml uv.lock ./

# 8. Instala apenas as dependências do projeto para aproveitar o cache
RUN uv sync --frozen --no-install-project --no-dev

# 9. Copia o restante do código do projeto
COPY --chown=user . .

# 10. Instala o projeto
RUN uv sync --frozen --no-dev

# 11. Expõe a porta dinâmica
EXPOSE ${PORT}

# 12. Executa o uvicorn (no Render ele usará a variável $PORT injetada; localmente usará 7860)
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]