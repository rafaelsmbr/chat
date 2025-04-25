# Usa uma imagem base do Python
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos da aplicação para dentro do container
COPY . .

# Instala as dependências da aplicação
RUN pip install --upgrade pip && \
    pip install flask langchain_community

# Expõe a porta que o Flask usará
EXPOSE 5000

# Define as variáveis de ambiente (pode ser sobrescrito via docker run ou docker-compose)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Comando para iniciar a aplicação
CMD ["flask", "run"]
