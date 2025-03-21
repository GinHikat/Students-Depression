FROM python:3.9-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y & apt install awscli -y

RUN apt-get update && pip install -r requirement.txt
CMD ["python3", 'app.py']