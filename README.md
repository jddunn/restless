<!-- PROJECT LOGO -->
<p align="center">
  <!-- <a href="https://github.com/github_username/repo">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->
  <h3 align="center">restless</h3>
  <p align="center">
    Always-on anti-malware software using malicious language processing (nlp-inspired) techniques.
        <br>
    <a href="https://github.com/github_username/repo">Explore the docs »</a>
    <a href="https://github.com/github_username/repo">View Demo</a>
        </p>
</p>


<!-- TABLE OF CONTENTS -->
## Table of contents

* [About](#about)
  * [Concepts overview](#concepts-overview)
  * [Architecture overview](#architecture-overview)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation with pip](#installation-with-pip)
  * [Using Docker](#using-docker)
* [Usage](#usage)
  * [CLI](#cli-usage)
  * [REST API](#rest-api-usage)
  * [Web UI](#web-ui-usage)
* [Docs](#docs)
  * [Auto-generating docs](#auto-generating-docs)
  * [API docs (interactive)](#api-docs)
  * [App docs](#app-docs)
* [Roadmap](#roadmap)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About


**Restless** is (or will be) a suite of platform-agnostic security tools with three key features:

* Analysis of files for malware probabilty based on comparing the extracted metadata and file contents with trained Portable Executable data *(completed)*
* Analysis of system logs for abnormal activities / deviations *(not started)*
* Analysis of web pages during browsing sessions to determine maliciousness *(not started)*

All analyses will be constantly ongoing, occur in real-time, and performed through machine learning models (using both supervised and unsupervised training techniques).

[Ray](https://ray.io/) is used for parallel / distributed computing, allowing bypassing of Python's bottleneck with the GIL and for asynchronous functionality.

By constantly watching and only scanning new files as verified by timestamps, **restless** can offer ongoing protection with minimal effort resources. You can also configure **restless** to run full or partial system scans on a schedule.

Restless aims to be fast, lightweight, and fully functional offline. The current configuration is for Ubuntu-based machines, but can be modified for other platforms by editing the `Dockerfile` and `docker-compose.yml` (eventually, Docker images will be built for Mac and Windows and linked here for download).

###  Concepts overview

Signature detection, the traditional method of antiviruses which creates the need to connect to online databases for incesstant updating, cannot keep up with the emergence of new malware, or even of known malware that's able to [change itself](https://nakedsecurity.sophos.com/2012/07/31/server-side-polymorphism-malware/), and while heuirstics-based approaches can combat polymorphic viruses while offering further degrees of granularity, they tend to give off so many false positives that they do more harm than good by wasting computing resources and increasing cognitive load.

The incorporation of machine learning (usually, natural langauge processing techniques although computer vision algorithms can also be applied) in antivirus software empowers them with a level of pattern recognition that previous approaches do not have. Rather than relying on known vulnerabilities / exploits or commonly re-used patterns of malicious code, **restless** and other ML-powered antiviruses can work well on classifying malware never seen before. This integration of NLP methods with information security has been dubbed [malicious language processing](https://www.elastic.co/blog/nlp-security-malicious-language-processing) by Elastic.

Unless drastic changes in programming paradigms occur for writing malware (which is possible), **restless** should be able to continue classifying new malware written years and years after the original models were trained with at least some amount of effectiveness (given an adequate training corpus).


### Architecture overview


<!-- GETTING STARTED -->
## Getting started

### Prerequisites

* [Python 3.7+](Python 3.7+)
* [Spark](Spark)
* [TensorFlow](TensorFlow) / [Keras](Keras)
* [FastAPI](FastAPI)
* [Uvicorn](Uvicorn)
* [spacey](spacey)
* [pdoc3](pdoc3] (fork of pdoc)
* [recentmost](recentmost) and C compiler

or

* [Docker](Docker) / [Docker-Compose](Docker-Compose)

### Installing with Pip

(Skip if using Docker)

```sh
pip install -r requirements.txt
```

### Building with Docker

(Skip to Using Docker-Compose if using Docker-Compose)

Build the image
```sh
docker build -t restless .
```

<!-- USAGE EXAMPLES -->
## Usage

It is recommended to run **restless** via Docker, unless say, you don't have enough storage to mount your drive in the container.

Restless's services will be running at port 4712 by default.

### Running restless locally

##### Starts ASGI server wth reload (recommended to use a virtualenv with all the dependencies installed)

```sh
cd restless/restless/app
python server.py
```

### Running with Docker

##### Running a container from a built image

The example command will mount all the files in a Ubuntu machine into the container, so a full system scan of the host machine can be performed through the dockerized service.

(Although the training data consists of malware that targets Windows OSes, research has shown that antivirus programs designed for Windows works on average 95% of the time in detecting Linux-based malware, see references at the bottom).

The container also adds an extra barrier of attack for potential bad agents, but keep in mind, is still not completely secure or sandboxed away from the host environment.


```sh
docker run -p 4712:4712 -e APP_ENV=docker --mount source=home,target=/home/ubuntu/restless restless
```
^ When running the Docker container, an env var `APP_ENV` should be set to `docker` (this var only needs to be set if using Docker). The param `source` would ideally be set to the home / root dir of the host drive (for full protection) or is whatever dir you want to be scanning,  and`target` must always point to the dir containing the Dockerfile and app files.

$ useful for eventual dynamic analysis of files, when they will need to be tested in an isolated place).


##### Using Docker-Compose 

(See the explanation above ^ to see how Docker will be mounting and communicating with your machine drive. If you're not using Ubuntu or a Debian-based Linux distro, then you'll need to edit `docker-compose.yml` to change the `source` and `target` paths under the `volume` key to reflect your OS's filepaths).

When using Docker-Compose, you won't need to build the image, or fiddle with parameters (unless you need to edit the Dockerfile or docker-compose.yml file).

Just run
```sh
docker-compose up
```

### CLI usage

##### Training your own dataset

##### Testing a dataset

##### Prediction

### API usage

### Web UI


##### Running the server

<!-- DOCS -->
## Docs

### Auto-generating docs

### API docs (interactive)
FastAPI will automatically generate interactive API docs according to [OpenAPI Spec](https://swagger.io/docs/specification/about/).

```sh
http://localhost:4712/docs
```

### App / library docs 

Lib docs (uses `pdoc` for autogeneration); the below command generates docs and creates a reasonable folder structure
```sh
pdoc --html restless --force; rm -rf docs; mv html docs; cd docs; cd restless; mv * .[^.]* ..; cd ..; rm -rf restless; cd ..
```

You can then browse the documentation web files or run a web server.

If you're using Docker, the app docs will be accessible (served statically) here:
```
http://localhost:4712/app_docs/index.html
```


<!-- CODE -->
## Code

### Linting

```sh
black restless
```

<!-- TESTS -->
## Tests

```sh
python -m unittest discover
```

<!-- ROADMAP -->
## Roadmap

In order of desirability

* Add analyzing system logs in real-time (will use k-means clustering to find logs with lots of deviation, or abnormal logs)
* Add analyzing browser URLs (detect clicks and listen for address changes in browser API) for malicious websites
* Train new model on Mac malware dataset, and load model dynamically based on OS / filetype found (better cross-platform compatability)
* Add dynamic analysis of executables by executing / testing files inside an isolated sandbox environmnet.
* Add generating MD5 hashes for new files and checking against known malware databases (like VirusTotal) - this might be the most important actually

<!-- PROJECT AUTHORS -->
## Project Authors

* [Johnny Dunn](https://github.com/jddunn) - johnnyddunn@gmail.com


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements / Other Contributions

* []()
* []()
* [Detecting Malware Across Operating Systems](https://www.opswat.com/blog/detecting-malware-across-operating-systems)
