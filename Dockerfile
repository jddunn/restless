FROM python:3.7

RUN pip install fastapi uvicorn

EXPOSE 5712

COPY ./app /app

WORKDIR ./app/

RUN ["chmod", "550", "./server.py"]

CMD ["python", "./server.py"]
