FROM python:2

RUN mkdir -p /usr/src/api
WORKDIR /usr/src/api/

COPY . /usr/src/api

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["main.py"]