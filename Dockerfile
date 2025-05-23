git # Base image
FROM python:3.10.12-buster

# Running every next command wih this user
USER root

# Creating work directory in docker
WORKDIR /usr/app

# Copying files to docker
ADD . '/usr/app'

# Installing Flask App
#RUN pip install flask


RUN apt-get update && \
    apt-get -y install python3-pandas

RUN apt-get -y install python3-opencv

RUN pip install --trusted-host pypi.python.org -r requirements2.txt
RUN pip install --trusted-host pypi.python.org -r requirements-nodeps.txt --no-deps
# Exposing the flask app port from container to host
EXPOSE 5001
EXPOSE 5000
# Starting application
ENTRYPOINT ["python", "-m", "server.server2"]
