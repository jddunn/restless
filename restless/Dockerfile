FROM python:3.7

RUN pip install fastapi uvicorn
RUN pip install pdoc3

EXPOSE 4712

COPY ./restless /restless

WORKDIR ./restless/

RUN ["chmod", "550", "./server.py"]

# RUN "pdoc --html restless --force"

# rm -rf docs; mv html docs; cd docs; cd restless; mv * .[^.]* ..; cd ..; rm -rf restless"

# RUN ["python -m http.server 4713"]

# RUN ["cd .."]

CMD ["python", "./server.py"]
