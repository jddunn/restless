FROM python:3.7

RUN pip install fastapi uvicorn

EXPOSE 4712

COPY ./restless /restless

WORKDIR ./restless/

RUN ["chmod", "550", "./server.py"]

CMD ["python", "./server.py"]
