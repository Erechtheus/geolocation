FROM python:3.6

COPY *.py /app/
COPY requirements.txt app/
COPY data/binaries/ /app/data/binaries/
COPY data/models/textBranchNorm.h5 /app/data/models/

WORKDIR /app
RUN pip install -r requirements.txt


EXPOSE 5000
CMD ["python", "./webservice.py"]


###Some comands I used for building this docker container

##Build docker container from 'Dockerfile'
#docker build -t geoloc .

##Execute docker container
#docker run -d -p   5000:5000 --network host  geoloc

##Contact docker webservice
#http://127.0.0.1:5000/predictText?text=Montmartre%20is%20truly%20beautiful

##Export docker container
#docker save geoloc > geolocV2.tar

########################### Other commands ###########################
##list images
#docker images -a

##ls for docker
#docker container ls

#Interactive execution of docker container
#docker exec -i -t 3411bb89b103 /bin/bash






