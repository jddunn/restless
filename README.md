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

* Analysis of files for malware probabilty based on comparing the extracted metadata and file contents with trained Portable Executable data *(completed model with almost a dozen extracted PE header features)*
* Analysis of system logs for abnormal activities / deviations *(not started)*
* Analysis of web pages during browsing sessions to determine maliciousness *(not started)*
* Constant, efficient and ongoing analysis of system files / logs *(in-progress)*

All analyses will be constantly ongoing, occur in real-time, and performed through machine learning models (using both supervised and unsupervised training techniques). No database is currently integrated with the code, but Spark clusters will be used along with Ray for distributed processing (eventually).

By constantly watching and only scanning new files as verified by timestamps, **restless** can offer ongoing protection with minimal effort resources. You can also configure **restless** to run full or partial system scans on a schedule.

Restless aims to be fast and fully functional offline. The current configuration is for Ubuntu-based machines, but can be modified for other platforms by editing the `Dockerfile` and `docker-compose.yml` (eventually, Docker images will be built for Mac and Windows and linked here for download).

Currently, there is no REST API functionality besides serving the documentation; only the CLI and library is functional.

----------------------------------------------------
### Preliminary results of training the HAN (Hierarchical Attention Network)  model with extracted PE features (CheckSum, AddressOfEntryPoint, e_minalloc, e_maxalloc, etc.):

```
cd restless/components/nlp/hann
python hann.py
Training HANN model now..
Train on 4148 samples, validate on 1036 samples
Epoch 1/5
150/4148 [>.............................] - ETA: 4:23 - loss: 0.8208 - acc: 0.5200
700/4148 [====>.........................] - ETA: 3:12 - loss: 0.6838 - acc: 0.6071
1525/4148 [==========>...................] - ETA: 2:20 - loss: 0.6341 - acc: 0.6689
2650/4148 [==================>...........] - ETA: 1:19 - loss: 0.6043 - acc: 0.6985
3725/4148 [=========================>....] - ETA: 22s - loss: 0.5980 - acc: 0.7034
```
---------------------------------------------------
After just one epoch, we have an accuracy of 70%, maxing out at 75% around 3-4 epochs. Given that this is just a preliminary model with just a basic amount of feature engineering done, these results are really promising. The default dataset the HANN script looks for is `./malware-dataset.csv`.

---------------------------------------------------
### Example program usage (CLI):

-i = folder to scan (containing a single known malware executable at the time of scanning)
```
cd restless
python cli.py -i data/
Using TensorFlow backend.
Succesfully loaded HANN model:  /home/ubuntu/restless/restless/components/nlp/hann/default.h5
Total 6766 unique tokens.
..
2020-03-23 21:15:06 INFO Restless initializing. Running system-wide scan: False
2020-03-23 21:15:06 INFO Initializing Restless.Scanner with PE Analyzer: <pe_analyzer.pe_analyzer.PE_Analyzer object at 0x7f7b7392e610>
2020-03-23 21:15:06 INFO Restless.Watcher is now watching over system and scanning new incoming files.
PEAnalyzer scanning:  data
..
File:  data/benign/audacity-win-2.3.3.exe  cannot be analyzed -  'DOS Header magic not found.'
File:  data/malicious/eh.exe  cannot be analyzed -  'The file is empty'
2020-03-23 21:15:08 INFO Scanned data/benign/7z1900-x64.exe - predicted: 0.79037845 benign and 0.20962158 malicious
2020-03-23 21:15:09 INFO Scanned data/malicious/bx89.exe - predicted: 0.8926386 benign and 0.10736139 malicious
2020-03-23 21:15:09 INFO Scanned data/malicious/microsoft office 2007 service pack 2.exe - predicted: 0.7755149 benign and 0.22448513 malicious
2020-03-23 21:15:10 INFO Scanned data/malicious/0.exe - predicted:  benign and 0.6572292 malicious
2020-03-23 21:15:11 INFO Scanned data/malicious/tekdefense.dll - predicted: 0.293381 benign and 0.706619 malicious
..
```
---------------------------------------------------

Malicious executables obtained from [http://www.tekdefense.com/downloads/malware-samples/](http://www.tekdefense.com/downloads/malware-samples/). Training dataset taken from [https://github.com/urwithajit9/ClaMP](https://github.com/urwithajit9/ClaMP).

###  Concepts overview

Signature detection, the traditional method of antiviruses which creates the need to connect to online databases for incesstant updating, cannot keep up with the emergence of new malware, or even of known malware that's able to change itself, and while heuirstics-based approaches can combat polymorphic viruses while offering further degrees of granularity, they tend to give off so many false positives that they do more harm than good by wasting computing resources and increasing cognitive load.

The incorporation of machine learning (usually, natural langauge processing techniques although computer vision algorithms can also be applied) in antivirus software empowers them with a level of pattern recognition that previous approaches do not have. Rather than relying on known vulnerabilities / exploits or commonly re-used patterns of malicious code, restless and other ML-powered antiviruses can work well on classifying malware never seen before. This integration of NLP methods with information security has been dubbed malicious language processing by Elastic.

### Architecture overview

* Hierarchical attention network (LSTM) for binary classification (benign vs malicious) of EXE files via extracting PE data / other metadata (like CheckSum, which is currently completed). The HANN model is perfect as it retains some element of document structure, which is important when analyzing file contents and looking for potentially destructive patterns in code.
* HANN trained for system API calls for benign vs malicious binary classification of logs (planned)
* K-means clustering for learning unsupervised patterns of abnormal / deviant logs (planned)
..

<!-- GETTING STARTED -->
## Getting started

### Prerequisites

* Python 3.75+
* [Spark](Spark) (currently configured by Docker but unused by the code; training models will eventually be processed on Spark clusters)
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

This will also be useful for eventual dynamic analysis of files, when they will need to be tested in an isolated place).


### Using Docker-Compose

(See the explanation above ^ to see how Docker will be mounting and communicating with your machine drive. If you're not using Ubuntu or a Debian-based Linux distro, then you'll need to edit `docker-compose.yml` to change the `source` and `target` paths under the `volume` key to reflect your OS's filepaths).

When using Docker-Compose, you won't need to build the image, or fiddle with parameters (unless you need to edit the Dockerfile or docker-compose.yml file).

Just run
```sh
docker-compose up
```

and the app will be live at:
```sh
http://localhost:4712
```

## Usage

### CLI usage

You can use the CLI like this to scan folders / files for malware probability:
```sh
python cli.py -i /home/ubuntu/restless/restless/data
```

`restless/data/malicious` contains a zipped up known malicious executable you can use for testing (unzip it first, as the scanner only works for EXE files at the moment). The password to unzip infected zips is `infected`.

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


