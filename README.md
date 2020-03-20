<!-- PROJECT LOGO -->
<p align="center">
  <!-- <a href="https://github.com/github_username/repo">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->
  <h3 align="center">restless</h3>
  <p align="center">
    Always-on anti-malware software using malicious language processing (nlp-inspired) techniques.
        <br>
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
  * [Using Docker-Compose](#using-docker-compose)
* [Usage](#usage)
  * [CLI](#cli-usage)
  * [REST API](#rest-api-usage)
  * [Web UI](#web-ui-usage)
* [Docs](#docs)
  * [API docs (interactive)](#api-docs)
  * [App docs](#app-docs)
* [Roadmap](#roadmap)
* [Acknowledgements](#acknowledgements)


<!-- ABOUT THE PROJECT -->
## About

**Restless** is (or will be) a suite of security tools with these key features:

* Analysis of files for malware probabilty based on comparing the extracted metadata and file contents with trained Portable Executable data *(completed model with CheckSum data; eventually model needs to have more features)*
* Analysis of system logs for abnormal activities / deviations *(not started)*
* Analysis of web pages during browsing sessions to determine maliciousness *(not started)*
* Constant, efficient and ongoing analysis of system files / logs *(in-progress)*

All analyses will be constantly ongoing, occur in real-time, and performed through machine learning models (using both supervised and unsupervised training techniques). No database is currently integrated with the code, but Spark clusters will be used along with Ray for distributed processing (eventually).

By constantly watching and only scanning new files as verified by timestamps, **restless** can offer ongoing protection with minimal effort resources. You can also configure **restless** to run full or partial system scans on a schedule.

Restless aims to be fast and fully functional offline. The current configuration is for Ubuntu-based machines, but can be modified for other platforms by editing the `Dockerfile` and `docker-compose.yml` (eventually, Docker images will be built for Mac and Windows and linked here for download).

Currently, there is no REST API functionality besides serving the documentation; only the CLI and library is functional.

----------------------------------------------------
Preliminary results from checksum (default) model (I was only able to train two epochs because my current machine's not powerful enough):
```
Train on 4168 samples, validate on 1042 samples
Epoch 1/2
4168/4168 [==============================] - 217s 52ms/step - loss: 0.5504 - acc: 0.7452 - val_loss: 0.5544 - val_acc: 0.7428
Epoch 2/2
4168/4168 [==============================] - 216s 52ms/step - loss: 0.5426 - acc: 0.7490 - val_loss: 0.5515 - val_acc: 0.7457
```
---------------------------------------------------


---------------------------------------------------
Example program usage (CLI):

-i = folder to scan (containing a single known malware executable at the time of scanning)
```
cd restless
python cli.py -i /home/ubuntu/restless/restless/data
..
2020-03-20 19:16:00 INFO Restless initializing. Running system-wide scan: False
2020-03-20 19:16:00 INFO Initializing Restless.Scanner with PE Analyzer: <pe_analyzer.pe_analyzer.PE_Analyzer object at 0x7efc0a7775d0>
2020-03-20 19:16:00 INFO Restless.Watcher is now watching over system and scanning new incoming files.
File:  /home/ubuntu/restless/restless/data/malicious/0.exe.zip  cannot be analyzed -  'DOS Header magic not found.'
File features found for:  /home/ubuntu/restless/restless/data/malicious/0.exe [23117, 144, 3, 0, 4, 0, 65535, 0, 184, 0, 0, 0, 64, 0, b'\x00\x00\x00\x00\x00\x00\x00\x00', 0, 0, b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', 216, 332, 2, 2011.2483564814816, 0, 0, 224, 271, 267, 6, 0, 0, 71680, 0, 5538, 4096, 4096, 4194304, 4096, 512, 4, 0, 0, 0, 4, 0, 77824, 1024, 100669, 2, 0, 1048576, 4096, 1048576, 4096, 0, 16]
..
2020-03-20 19:16:01 INFO Scanned /home/ubuntu/restless/restless/data/malicious/0.exe - predicted: 0.11168818 benign and 0.8883118 malicious
```
---------------------------------------------------

###  Concepts overview

Signature detection, the traditional method of antiviruses which creates the need to connect to online databases for incesstant updating, cannot keep up with the emergence of new malware, or even of known malware that's able to [change itself](https://nakedsecurity.sophos.com/2012/07/31/server-side-polymorphism-malware/), and while heuirstics-based approaches can combat polym$

The incorporation of machine learning (usually, natural langauge processing techniques although computer vision algorithms can also be applied) in antivirus software empowers them with a level of pattern recognition that previous approaches do not have. Rather than relying on known vulnerabilities / exploits or commonly re-used patterns of malicious code, **restless** and other ML-powered antimalware programs will be able to see and understand new patterns / exploits never seen before.

Unless drastic changes in programming paradigms occur for writing malware (which is possible), **restless** should be able to continue classifying new malware written years and years after the original models were trained with at least some amount of effectiveness (given an adequate training corpus).

### Architecture overview

* Hierarchical attention network (LSTM) for binary classification (benign vs malicious) of EXE files via extracting PE data / other metadata (like CheckSum, which is currently completed). The HANN model is perfect as it retains some element of document structure, which is important when analyzing file contents and looking for potentially destructive patterns in code.
* HANN trained for system API calls for benign vs malicious binary classification of logs (planned)
* K-means clustering for learning unsupervised patterns of abnormal / deviant logs (planned)
..

<!-- GETTING STARTED -->
## Getting started

### Prerequisites

* Python 3.75+
* [Spark](Spark)
* [TensorFlow](TensorFlow) / [Keras](Keras)
* [FastAPI](FastAPI)
* [Uvicorn](Uvicorn)
* [pdoc3](pdoc3) (fork of pdoc)

or

* [Docker](Docker) / [Docker-Compose](Docker-Compose)

### Installation with Pip

(Skip if using Docker)

```sh
pip install -r requirements.txt
```

then:

Starts ASGI server wth reload (recommended to use a virtualenv with all the dependencies installed):

```sh
cd restless
python server.py
```

Restless's services will be running at port 4712 by default.

### Using Docker

(Skip to Using Docker-Compose if using Docker-Compose)

Build the image
```sh
docker build -t restless .
```

The example command will mount all the files in a Ubuntu machine into the container, so a full system scan of the host machine can be performed through the dockerized service.

(Although the training data consists of malware that targets Windows OSes, research has shown that antivirus programs designed for Windows works on average 95% of the time in detecting Linux-based malware, see references at the bottom).

The container also adds an extra barrier of attack for potential bad agents, but keep in mind, is still not completely secure or sandboxed away from the host environment.

```sh
docker run -p 4712:4712 -e APP_ENV=docker --mount source=home,target=/home/ubuntu/restless restless
```
^ When running the Docker container, an env var `APP_ENV` should be set to `docker` (this var only needs to be set if using Docker). The param `source` would ideally be set to the home / root dir of the host drive (for full protection) or is whatever dir you want to be scanning,  and`target` must always point to the dir containing the Dockerfile and app files.

$ useful for eventual dynamic analysis of files, when they will need to be tested in an isolated place).


### Using Docker-Compose

(See the explanation above ^ to see how Docker will be mounting and communicating with your machine drive. If you're not using Ubuntu or a Debian-based Linux distro, then you'll need to edit `docker-compose.yml` to change the `source` and `target` paths under the `volume` key to reflect your OS's filepaths).

When using Docker-Compose, you won't need to build the image, or fiddle with parameters (unless you need to edit the Dockerfile or docker-compose.yml file).

Just run
```sh
docker-compose up
```

and the app will be live at:
'''sh
http://localhost:4712
'''

## Usage

### CLI usage

You can use the CLI like this to scan folders / files for malware probability:
```sh
python cli.py -i /home/ubuntu/restless/restless/data
```

`restless/data/malicious` contains a zipped up known malicious executable you can use for testing (unzip it first, as the scanner only works for EXE files at the moment).

### API usage

The API will be up at:
```
http://localhost:4712
````

At the moment, only the docs are accessible through the API. Eventually, we'll be able to send data to be analyzed via REST and get scan results.

### Web UI

In-progress


<!-- DOCS -->
## Docs

### API docs
FastAPI will automatically generate interactive API docs according to [OpenAPI Spec](https://swagger.io/docs/specification/about/).

```sh
http://localhost:4712/api_docs
```

### App docs 
Lib / app docs (uses `pdoc` for autogeneration); the below command generates docs and creates a reasonable folder structure
```sh
pdoc --html restless --force; rm -rf docs; mv html docs; cd docs; cd restless; mv * .[^.]* ..; cd ..; rm -rf restless; cd ..
```

You can then browse the documentation web files or run a web server.

If you're using Docker, the app docs will be accessible (served statically) here:
```
http://localhost:4712/app_docs/index.html
```

They'll look like this:

![Restless app docs screenshot](/screenshots/restless-app-docs-screenshot.png?raw=true)


<!-- CODE -->
## Code

### Linting
```
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

* [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
* [ClaMP (Classification of Malware with PE headers)](https://github.com/urwithajit9/ClaMP)
* [Detecting Malware Across Operating Systems](https://www.opswat.com/blog/detecting-malware-across-operating-systems)


