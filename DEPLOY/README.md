# DbyDeep
DbyDeep: Exploration of MS detectable peptides via deep learning [AnalyChem 2023]  

## Hardware
DbyDeep requires  
* a GPU with CUDA support

## Installation
Prosit requires



Prosit was tested on Ubuntu 16.04, CUDA 8.0, CUDNN 6 with Nvidia Tesla K40c and Titan Xp graphic cards with the dependencies above.

The time installation takes is dependent on your download speed (Prosit downloads a 3GB docker container). In our tests installation time is ~5 minutes.

## Model
Prosit assumes your models are in directories that look like this:

model.yml - a saved keras model
config.yml - a model specifying names of inputs and outputs of the model
weights file(s) - that follow the template weights_{epoch}_{loss}.hdf5


## Usage
The following command will load your model from /path/to/model/. In the example GPU device 0 is used for computation. The default PORT is 5000.

make server MODEL_SPECTRA=/path/to/fragmentation_model/ MODEL_IRT=/path/to/irt_model/
Currently two output formats are supported: a MaxQuant style msms.txt not including the iRT value and a generic text file (that works with Spectronaut)

## Example
Please find an example input file at example/peptidelist.csv. After starting the server you can run the following commands, depending on what output format you prefer:

curl -F "peptides=@examples/peptidelist.csv" http://127.0.0.1:5000/predict/generic

curl -F "peptides=@examples/peptidelist.csv" http://127.0.0.1:5000/predict/msp

curl -F "peptides=@examples/peptidelist.csv" http://127.0.0.1:5000/predict/msms
The examples take about 4s to run. Expected output files (.generic, .msp and .msms) can be found in examples/.

## Using DbyDeep on your data
You can adjust the example above to your own needs. Send any list of (Peptide, Precursor charge, Collision energy) in the format of /example/peptidelist.csv to a running instance of the Prosit server.

Please note: Sequences with amino acid U, O, or X are not supported. Modifications except "M(ox)" are not supported. Each C is treated as Cysteine with carbamidomethylation (fixed modification in MaxQuant).