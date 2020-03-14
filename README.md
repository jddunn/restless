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
* [Roadmap](#roadmap)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About


**Restless** is (or will be) a suite of platform-agnostic security tools with three key features: 

* Analysis of files for malware probabilty based on comparing the extracted metadata and file contents with trained Portable Executable data *(completed)*
* Analysis of system logs for abnormal activities / deviations *(not started)*
* Analysis of web pages during browsing sessions to determine maliciousness *(not started)*

All analyses will be constantly ongoing, occur in real-time, and performed through machine learning models (using both supervised and unsupervised training techniques).

[Ray](https://ray.io/) is used for parallel / distributed computing, allowing bypassing of Python's bottleneck with the GIL and for asynchronous functionality. Restless also incorporates [recentmost](https://github.com/shadkam/recentmost), a program written in C which can find the most recently updated files on a machine with blazingly fast performance.

By constantly watching and only scanning new files as verified by timestamps, **restless** can offer ongoing protection with minimal effort resources. You can also configure **restless** to run full or partial system scans on a schedule.

Restless aims to be fast, lightweight, and fully functional offline.

###  Concepts overview

Signature detection, the traditional method of antiviruses which creates the need to connect to online databases for incesstant updating, cannot keep up with the emergence of new malware, or even of known malware that's able to [change itself](https://nakedsecurity.sophos.com/2012/07/31/server-side-polymorphism-malware/), and while heuirstics-based approaches can combat polymorphic viruses while offering further degrees of granularity, they tend to give off so many false positives that they do more harm than good by wasting computing resources and increasing cognitive load.

The incorporation of machine learning (usually, natural langauge processing techniques although computer vision algorithms can also be applied) in antivirus software empowers them with a level of pattern recognition that previous approaches do not have. Rather than relying on known vulnerabilities / exploits or commonly re-used patterns of malicious code, **restless** and other ML-powered antiviruses can work well on classifying malware never seen before. This integration of NLP methods with information security has been dubbed [malicious language processing](https://www.elastic.co/blog/nlp-security-malicious-language-processing) by Elastic.

Unless drastic changes in programming paradigms occur for writing malware (which is possible), **restless** should be able to continue classifying new malware written years and years after the original models were trained with at least some amount of effectiveness (given an adequate training corpus).


### Architecture overview





<!-- GETTING STARTED -->
## Getting started

### Prerequisites

* Python 3.7
* Spark
* TensorFlow / Keras
* FastAPI
* spacey

or

* Docker

### Installing with Pip

```sh
pip install -r requirements.txt
```

### Using Docker (no installing dependencies required)

Build the image
```sh
docker build -t restless .
```

or download from here:



<!-- USAGE EXAMPLES -->
## Usage

It is recommended to run **restless** via Docker, unless say, you don't have enough storage to mount your drive in the container.

### Running restless locally

##### Starts ASGI server wth reload (recommended to use a virtualenv with all the dependencies installed)

```sh
cd restless/restless/app
uvicorn app.main:app --host 0.0.0.0 --port 80 --reload
```

### Running restless as a service with Docker

##### Running a container from a built image
The example command will mount all the files in a Ubuntu machine into the container, so a full system scan of the host machine can be performed through the dockerized service.

(Although the training data consists of malware that targets Windows OSes, research has shown that antivirus programs designed for Windows works on average 95% of the time in detecting Linux-based malware, see references at the bottom).

The container also adds an extra barrier of attack for potential bad agents, but keep in mind, is still not completely secure or sandboxed away from the host environment.

```sh
docker run -p 80:80 -e APP_ENV=docker --mount source=home,target=/home/ubuntu/restless restless
```
^ When running the Docker container, an env var `APP_ENV` should be set to `docker` (this var only needs to be set if using Docker). The param `source` would ideally be set to the home / root dir of the host drive (for full protection) or is whatever dir you want to be scanning,  and`target` must always point to the dir containing the Dockerfile and app files.

As new files are saved / ingested in the host machine, the volume mounted in the Docker container should update accordingly (All this will be useful for eventual dynamic analysis of files, when they will need to be tested in an isolated place).

### CLI usage

##### Training your own dataset

##### Testing a dataset

##### Prediction

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
* Add generating MD5 hashes for new files and checking against known malware databases (like VirusTotal) - this might be the most important actually

<!-- PROJECT AUTHORS -->
## Project Authors

* [Johnny Dunn](https://github.com/jddunn) - johnnyddunn@gmail.com


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements / Other Contributions

* []()
* []()
* [Detecting Malware Across Operating Systems](https://www.opswat.com/blog/detecting-malware-across-operating-systems)
