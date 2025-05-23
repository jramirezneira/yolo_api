FROM python:3.10.12-buster

USER root

# 1. Establece el directorio de trabajo
WORKDIR /usr/app

# 2. Copia solo los archivos de dependencias primero
COPY requirements3.txt requirements-nodeps.txt ./

# 3. Instala dependencias del sistema (en una sola capa para eficiencia)
RUN apt-get update && \
    apt-get install -y python3-pandas python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. Instala dependencias de Python
RUN pip install --trusted-host pypi.python.org -r requirements2.txt && \
    pip install --trusted-host pypi.python.org -r requirements-nodeps.txt --no-deps

# 5. Ahora copia el resto del código
COPY . .

# 6. Expone los puertos
EXPOSE 5000 5001

# 7. Define el punto de entrada
ENTRYPOINT ["python", "-m", "server.server2"]
