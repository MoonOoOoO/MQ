FROM ubuntu:18.04
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    python3-pip \
    && pip3 install pip --upgrade \
    && pip3 install pyzmq tensorflow opencv-python flask gevent
COPY ./client.py /foo/client.py
COPY ./arraytool.py /foo/arraytool.py
WORKDIR /foo
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["client.py"]
