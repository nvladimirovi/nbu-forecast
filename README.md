# Forecast - Exam 2
## Prerequisites
You need:
- Git
- Python 3.9.13
- Docker

## Setup
- Git Clone the project
- Create content folder
- Download the dataset from https://www.kaggle.com/code/sshikamaru/fruit-classification-starter-cnn and unzip the content in `./content/dataset`

Create python virtual env
```bash
# In the root of the project run
python -m venv venv
```

Install packages
```bash
# For Windows use  ./requirements-win.txt
pip install -r ./requirements.txt
```

## How to run the server locally
```bash
# From the root of the project
python ./server/manage.py runserver
```

## How to call the prediction API
Example curl
```bash
curl --location --request POST 'http://127.0.0.1:8000/predict/' \
--header 'Content-Disposition: attachment; filename=0000.jpg' \
--header 'Content-Type: image/jpeg' \
--data-binary '@/path/to/your/image.jpg'
```

# Software Architectures for ML - Exam 2
## Build Docker Image
```docker build -t sem2-forecast-exam2 .```

## Start Docker Container
```docker run --name fruit-classifier -it -p 8020:8020 sem2-forecast-exam2```
If container with same name already exists
```docker run fruit-classifier```

Exampe Predict API:
```bash
curl --location --request POST 'http://127.0.0.1:8020/predict/' \
--header 'Content-Disposition: attachment; filename=0000.jpg' \
--header 'Content-Type: image/jpeg' \
--data-binary '@/path/to/your/image.jpg'
```

## Kill container
```docker kill fruit-classifier```
https://docs.docker.com/engine/reference/commandline/kill/

## Push container to Docker Hub
```docker commit fruit-classifier nvladimirovi/nbu```
