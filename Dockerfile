FROM python:3.9-buster

RUN apt-get update && apt-get install nginx vim -y --no-install-recommends
COPY nginx.default /etc/nginx/sites-available/default
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log

# Prepare model stuff
RUN mkdir -p /opt/app/content/vgg16_bn.pth
COPY content/vgg16_bn.pth /opt/app/content/vgg16_bn.pth
COPY VGG16_v2-Fruit-Classifier.pt /opt/app/

RUN mkdir -p /opt/app
RUN mkdir -p /opt/app/pip_cache
RUN mkdir -p /opt/app/server
COPY requirements.txt start-server.sh /opt/app/
# COPY .pip_cache /opt/app/pip_cache/
COPY server /opt/app/server/
WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN chown -R www-data:www-data /opt/app

EXPOSE 8020
STOPSIGNAL SIGTERM
CMD ["/opt/app/start-server.sh"]


