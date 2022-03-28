# 2022-Interspeech
Data used in experiments described in the paper "Sequence-to-Sequence CNN-BiLSTM Based Glottal Closure Instant Detection from Raw Speech".

An example of a Python code to train and test a CNN-BiLSTM model, a joint convolutional (CNN) and recurrent (RNN) neural network model, for detecting glottal closure instants (GCIs) in the speech signal, including a brief data and detection/classification procedure description, is shown in interactive Jupyter notebook `GCI_detection.ipynb`. Please see the [official page](http://jupyter.org/) for an information on how to launch the jupyter notebook.

We recommend to use [direnv](https://direnv.net/) together with [pyenv](https://github.com/pyenv/pyenv) (with python 3.8.7 installed in our case) to prepare your working virtual environment (as described, for instance, [here](https://stackabuse.com/managing-python-environments-with-direnv-and-pyenv/)):

```console
git clone https://github.com/ARTIC-TTS-experiments/2022-Interspeech.git
mkdir 2022-Interspeech
cd 2022-Interspeech
echo -e layout pyenv 3.8.7"\n"LIB=\"'$(pwd)/lib'\""\n"export PYTHONPATH=\"'$LIB'\" > .envrc
direnv allow
pip install -r requirements.txt
```
