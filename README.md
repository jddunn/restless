<!-- PROJECT LOGO -->
<p align="center">
  <a href="https://github.com/github_username/repo">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
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
* [Roadmap](#roadmap)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About


Restless is (or will be) a suite of platform-agnostic security tools with three key features: analysis of files (static) for malware probabilty based on extracted PE (Portable Executable) data, analysis of system logs for abnormal activities / deviations (not started), and analysis of web pages during browsing sessions to determine maliciousness (not started). All analyses will be constantly ongoing, occur in real-time, and will performed through machine learning models (using both supervised and unsupervised training techniques). 

[Ray](https://ray.io/) is used for parallel / distributed computing, allowing bypassing of Python's bottleneck with the GIL and for asynchronous functionality. Restless also incorporates [recentmost](https://github.com/shadkam/recentmost), a program written in C which can find the most recently updated files on a machine with blazingly fast performance. 

By constantly watching and scanning new files verified by timestamps, **restless** can offer ongoing and complete protection (after an initial system scan) with minimal effort and processing power.

Restless aims to be fast, lightweight, and fully offline. 

###  Concepts overview

Signature detection, the traditional method of antiviruses which creates the need to connect to online databases and update incesstantly, cannot keep up with the emergence of new malware or of known malware that's able to [change itself](https://nakedsecurity.sophos.com/2012/07/31/server-side-polymorphism-malware/), and while heuirstics-based approaches can combat polymorphic viruses and offer further degrees of granularity, they pretty much always manage to give off so many false positives that they do more harm than good, by wasting computing resources and increasing cognitive load for the end user.

The incorporation of machine learning (usually, natural langauge processing techniques although computer vision algorithms can also be applied) in antivirus software empowers them with a level of pattern recognition that no other approach before can come close to. Rather than rely on known security patches, vulnerabilities / exploits, or commonly used patterns of code, the models restless and other ML-powered antimalware software use can work well on classifying data that anti-malware software may never even have seen before. 

Unless drastic changes in programming paradigms occur for malware (which is possible), **restless** should be able to continue classifying new malware written years and years after the original models were trained with some effectiveness, even without ever updating or training the models with new data.


### Architecture overview





<!-- GETTING STARTED -->
## Getting started

### Prerequisites

* Python 3.7
* Spark
* TensorFlow / Keras
* FastAPI

or

* Docker

### Installing with Pip

```sh
pip install -r requirements.txt
```

### Using Docker (no installing dependencies required)

Build the image:
```sh
docker build -t restless .
```

or download from here:



<!-- USAGE EXAMPLES -->
## Usage

It is recommended to run **restless** as an indepedent service via Docker.

### Running restless locally

##### Starts ASGI server wth reload (recommended to use a virtualenv with all the dependencies)

```sh
cd restless/restless/app
uvicorn app.main:app --host 0.0.0.0 --port 80 --reload
```

### Running restless as a service with Docker

##### Running a container
The example command will mount the entirety of a Ubuntu drive into the container, so a full system scan of the host machine can be performed through the dockerized service.

The container also adds an extra barrier of attack for potential bad agents, but keep in mind, is still not completely secure or sandboxed away from the host environment.

```sh
docker run --name restless-container -p 80:80 -e APP_ENV=docker --mount source=home,target=/home/ubuntu/restless restless
```
^ When running the Docker container, env var `APP_ENV` should be set to `docker`. The param `source` would ideally be set to the home / root dir of the host drive (for full protection), `target` must always point to the dir containing the Docker files.

As new files are saved / ingested in the host machine, the volume mounted in the Docker container should update accordingly. (All this will be useful for eventual dynamic analysis of files, when they need to be tested / executed in an isolated sandbox env).

### CLI usage

##### Training your own dataset

##### Testing a model

### API usage

##### Docs

### Web UI

##### Running the server

```sh
python http-server
```

<!-- ROADMAP -->
## Roadmap

In order of desirability

* Add analyzing system logs in real-time (will use k-means clustering to find logs with lots of deviation, or abnormal logs)
* Add analyzing browser URLs (detect clicks and listen for address changes in browser API) for malicious websites
* Train new model on Mac malware dataset, and load model dynamically based on OS / filetype found (better cross-platform compatability)
* Add dynamic analysis of executables by executing / testing files inside an isolated sandbox environmnet.

<!-- PROJECT AUTHORS -->
## Project Authors

* [Johnny Dunn](https://github.com/jddunn) - johnnyddunn@gmail.com


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements / Other Contributions

* []()
* []()
* []()



