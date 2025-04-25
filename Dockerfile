# Usa uma imagem base do Python
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos da aplicação e o requirements.txt para o container
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia o restante dos arquivos da aplicação
COPY . .

# Expõe a porta que o Flask usará
EXPOSE 5000

# Define as variáveis de ambiente (pode ser sobrescrito via docker run ou docker-compose)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Comando para iniciar a aplicação
CMD ["flask", "run"]
