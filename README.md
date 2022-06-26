## Build Docker Image
```docker build -t sem2-forecast-exam2 .```

## Start Docker Container
```docker run --name fruit-classifier -it -p 8020:8020 sem2-forecast-exam2```

## Kill container
```docker kill fruit-classifier```
https://docs.docker.com/engine/reference/commandline/kill/
