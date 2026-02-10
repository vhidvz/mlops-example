# ####################################################################
# docker build \
#   --build-arg MLFLOW_TRACKING_USERNAME=admin \
#   --build-arg MLFLOW_TRACKING_PASSWORD=password1234 \
#   --build-arg MLFLOW_SERVER_ENDPOINT_URL=http://127.0.0.1:5000 \
#   -t mlops-example/python:3.12 .
# make build
# 
# docker run --gpus all -p 3000:3000 mlops-example/python:3.12
# docker run --runtime=nvidia -p 3000:3000 mlops-example/python:3.12
# 
# sudo apt-get install -y nvidia-container-toolkit
# sudo nvidia-ctk runtime configure --runtime=docker
# ####################################################################
FROM mlops-example/python:3.12-base

ARG MLFLOW_TRACKING_USERNAME=admin
ARG MLFLOW_TRACKING_PASSWORD=password1234
ARG MLFLOW_SERVER_ENDPOINT_URL=http://127.0.0.1:5000

WORKDIR /app

COPY . .

RUN python upgrade.py

ENV PORT=3000
ENV WORKERS=1

EXPOSE $PORT

CMD ["fastapi", "run", "--workers", "${WORKERS}", "app.py", "--port", "${PORT}"]
