<!-- PROJECT LOGO -->
<p align="center">
  <a href="https://jddunn.github.io/restless/">
    <img src="./screenshots/Transparent.png" alt="Logo" width="205" height="48">
  </a>
  <p align="center">
    Always-on anti-malware software using malicious language processing (nlp-inspired) techniques.
        <br>
        </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of contents
* [About](#about)
  * [Screenshots and Results](#screenshots-and-results)
  * [Training Hierarchical Attention Network Model](#training-hierarchical-attention-network-model)
  * [Example CLI Usage](#example-cli-usage)
* [Background](#background)
  * [Concepts](#concepts)
  * [Architecture](#architecture)
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

* Analysis of files for malware probabilty based on comparing the extracted metadata and file contents with trained Portable Executable data *(completed model with almost a dozen extracted PE header features with **~83% accuracy**)*
* Analysis of system logs for abnormal activities / deviations *(not started)*
* Constant, efficient and ongoing analysis of system files / logs *(in-progress)*

All analyses will be constantly ongoing, occur in real-time, and performed through machine learning models (using both supervised and unsupervised training techniques). No database is currently integrated with the code, but Spark clusters will be used along with Ray for distributed processing (eventually).

Restless aims to be fast and fully functional offline. The current configuration is for Ubuntu-based machines, but can be modified for other platforms by editing the `Dockerfile` and `docker-compose.yml` (eventually, Docker images will be built for Mac and Windows and linked here for download).

Currently, there is no REST API functionality besides serving the documentation; only the CLI and library is functional.

## Screenshots and Results
<div>
  <h4>Feature Selection</h4>
  <p>We use Pearson coefficient to see how our features correlate with each other, and since we're doing binary classification, we can get the point-biserial correlation for each feature compared to our target feature (classification of "benign" or "malicious").</p>
  <p>Unlike conventional feature selection for regression, our model (HANN) takes a representation of documents as features (features ~= document sentences in this context). For HANN, we care about <b>any</b> feature that has some linear correlation, positive or negative.</p>
  <a href="./screenshots/model_results/Features Correlation Matrix for PE Header Data 2020-04-07 03:57:48.png" style="left">
    <img src="./screenshots/model_results/Features Correlation Matrix for PE Header Data 2020-04-07 03:57:48.png" alt="Features Correlaton Matrix for PE Header Data" width="450">
  </a>
  <a href="./screenshots/model_results/Top Features Correlation Matrix for PE Header Data (Minimum threshold of 0.1) 2020-04-07 03:57:50.png" style="right">
    <img src="./screenshots/model_results/Top Features Correlation Matrix for PE Header Data (Minimum threshold of 0.1) 2020-04-07 03:57:50.png" alt="Top Features Correlaton Matrix for PE Header Data" width="450">
  </a>
</div>

---------------------------------------------------
### Training Hierarchical Attention Network Model with extracted PE features (CheckSum, AddressOfEntryPoint, e_minalloc, e_maxalloc, etc.):

```
cd restless/components/nlp/hann
python train_hann.py
Training HANN model now..
Creating HANN model now, with K-Fold cross-validation. K= 1 and length:  4147 1037 for training / validation.
Train on 4147 samples, validate on 1037 samples
Epoch 1/3
 - 123s - loss: 0.3684 - accuracy: 0.8375 - val_loss: 0.2548 - val_accuracy: 0.9190
Epoch 2/3
 - 120s - loss: 0.2091 - accuracy: 0.9280 - val_loss: 0.2200 - val_accuracy: 0.9219
Epoch 3/3
 - 119s - loss: 0.1777 - accuracy: 0.9378 - val_loss: 0.2060 - val_accuracy: 0.9272
..
Metrics summed and averaged:  {'accuracy': 0.9486916521149886, 'loss': 1.7721489931601824, 'precision': 0.9523880705481174, 'recall': 0.9491701254870464, 'f1': 0.9499961372374752, 'kappa': 0.8971714579760806, 'auc': 0.9501432171274349}
The best performing model (based on F1 score) was number 4. That is the model that will be returned.
Training successful.
```
---------------------------------------------------
Stats of our best performing model (highest F1 score):
```
Model evaluation metrics: 
        Confusion matrix:
                 benign malicious 
       benign     481.0      31.0 
    malicious       4.0     521.0 
        Accuracy: 0.9662487945998072    Loss: 1.165749239481399
        Precision: 0.9923809523809524
        Recall: 0.9438405797101449
        F1 score: 0.967502321262767
        Cohens kappa score: 0.9324428709965026
        ROC AUC score: 0.9677965785148663
```
This is the current default model for HANN in this repo.
---------------------------------------------------
### Example CLI Usage
-i = file or folder to scan recursively
```
cd restless
python cli.py -i test_exes/
Using TensorFlow backend.
Succesfully loaded HANN model:  /home/ubuntu/restless/restless/components/nlp/hann/default.h5
Total 6766 unique tokens.
..
2020-04-07 06:10:23 INFO Initializing HANN module.
2020-04-07 06:10:23 INFO Restless initializing. Running system-wide scan: False
2020-04-07 06:10:23 INFO Restless.Watcher is now watching over system and scanning new incoming files.
PEAnalyzer scanning:  ../test_exes/
Error while saving features:  'Structure' object has no attribute 'BaseOfData'
2020-04-07 06:11:39 INFO Scanned ../test_exes/benign/CuteWriter.exe - predicted: 26.8862247467041% benign and 72.2758412361145% malicious
2020-04-07 06:11:41 INFO Scanned ../test_exes/benign/7z1900-x64.exe - predicted: 32.600608468055725% benign and 55.8910608291626% malicious
2020-04-07 06:11:42 INFO Scanned ../test_exes/benign/Explorer++_1.exe - predicted: 8.707889914512634% benign and 86.38086318969727% malicious
2020-04-07 06:11:44 INFO Scanned ../test_exes/benign/setup-lightshot.exe - predicted: 26.8862247467041% benign and 72.2758412361145% malicious
2020-04-07 06:11:46 INFO Scanned ../test_exes/benign/WinCDEmu-4.1.exe - predicted: 98.88728857040405% benign and 1.0872036218643188% malicious
2020-04-07 06:11:47 INFO Scanned ../test_exes/benign/putty.exe - predicted: 99.82845187187195% benign and 0.12594759464263916% malicious
2020-04-07 06:11:48 INFO Scanned ../test_exes/benign/peazip-7.1.1.WIN64.exe - predicted: 26.8862247467041% benign and 72.2758412361145% malicious
2020-04-07 06:11:50 INFO Scanned ../test_exes/malicious/bx89.exe - predicted: 13.227593898773193% benign and 81.54139518737793% malicious
2020-04-07 06:11:51 INFO Scanned ../test_exes/malicious/Bombermania.exe - predicted: 62.76884078979492% benign and 31.845125555992126% malicious
2020-04-07 06:11:56 INFO Scanned ../test_exes/malicious/3.exe - predicted: 8.707889914512634% benign and 86.38086318969727% malicious
2020-04-07 06:11:57 INFO Scanned ../test_exes/malicious/711.exe - predicted: 8.707889914512634% benign and 86.38086318969727% malicious
2020-04-07 06:12:01 INFO Scanned ../test_exes/malicious/microsoft office 2007 service pack 2.exe - predicted: 1.6987472772598267% benign and 98.32490682601929% malicious
2020-04-07 06:12:05 INFO Scanned ../test_exes/malicious/se.exe - predicted: 18.145954608917236% benign and 78.49445343017578% malicious
2020-04-07 06:12:07 INFO Scanned ../test_exes/malicious/mcpatcher.exe - predicted: 1.6987472772598267% benign and 98.32490682601929% malicious
2020-04-07 06:12:08 INFO Scanned ../test_exes/malicious/tekdefense.dll - predicted: 56.837183237075806% benign and 38.82768452167511% malicious
2020-04-07 06:12:11 INFO Scanned ../test_exes/malicious/25000.exe - predicted: 1.8150955438613892% benign and 97.06166982650757% malicious
```
---------------------------------------------------
Malicious executables obtained from [http://www.tekdefense.com/downloads/malware-samples/](http://www.tekdefense.com/downloads/malware-samples/). Training dataset taken from [https://github.com/urwithajit9/ClaMP](https://github.com/urwithajit9/ClaMP). 

## Background

### Concepts

Signature detection, the traditional method of antiviruses which creates the need to connect to online databases for incesstant updating, cannot keep up with the emergence of new malware, or even of known malware that's able to change itself, and while heuirstics-based approaches can combat polymorphic viruses while offering further degrees of granularity, they tend to give off so many false positives that they do more harm than good by wasting computing resources and increasing cognitive load.

The incorporation of machine learning (usually, natural langauge processing techniques although computer vision algorithms can also be applied) in antivirus software empowers them with a level of pattern recognition that previous approaches do not have. Rather than relying on known vulnerabilities / exploits or commonly re-used patterns of malicious code, restless and other ML-powered antiviruses can work well on classifying malware never seen before. This integration of NLP methods with information security has been dubbed malicious language processing by Elastic.

### Architecture

* Hierarchical attention network (LSTM) for binary classification (benign vs malicious) of EXE files via extracting PE metadata and strings from file contents (including obfuscated strings). The HANN model is perfect as it retains some element of document structure, which is important when analyzing file contents and looking for potentially destructive patterns in code.
* HANN trained for system API calls for benign vs malicious binary classification of logs (planned)
* K-means clustering for learning unsupervised patterns of abnormal / deviant logs (planned)
..

Restless's current classifications are implemented through a [hierarchical attention network](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-atten$), a type of recurrent neural network with an attention layer that can find the most important words and sentences to represent a document, as it is able to read the data in a bidirectional way to learn context.

The architecture from the paper has been modified in Restless's implenetation of HAN, in that a vector (array of features) is now acceptable (as well as scalars) for both "sentences" and "words" that build up the vocabulary.  

This allows us to represent any arbitrary set of features as a document that the HAN model can learn from, as GloVe (the pre-trained word embeddings) has weights for words as well as numbers. For example, in a word embeddings model like GloVe that considers numbers, a number like 8069 will have contextual meaning (as 8069 would be referenced a lot in computing-related documents).

Typically a document would be tokenized into sentences and then into words to fit the format the HAN model needs, but we can construct a representation of a document that corresponds to the metadata of our file. By extracting PE features like CheckSum, AddressOfEntryPoint, e_minalloc, and more, and considering each feature as a sentence, we can create a HAN classifier that reads executable files and their metadata like documents, and make use of the attention layer so it understands which features have more contextual importance than others.

Originally, Restless's classifier was going to extract strings (including obfuscated strings) from file contents of known malicious and benign files, and then build document representations from that dataset using the hierarchical attention neural network. However, collecting a dataset of executables (from trustworthy sources) is proving to be very time-consuming, so the current focus will focus on the file metadata representation construction.

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
python cli.py -i /home/ubuntu/
```

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


