## Transferable Representation Learning

Hi there, welcome to this page!

This page contains the code and data used in the paper [Cross-Project Transfer Representation Learning for Vulnerable Function Discovery] (https://ieeexplore.ieee.org/abstract/document/8329207/) by Guanjun Lin; Jun Zhang; Wei Luo; Lei Pan; Yang Xiang; Olivier De Vel and Paul Montague.

### Instructions:

The Vulnerabilities_info.xlsx file contains information of the collected function-level vulnerabilities. These vulnerabilities are from 6 open source projects: [FFmpeg](https://github.com/FFmpeg/FFmpeg), [LibTIFF](https://github.com/vadz/libtiff), [LibPNG](https://github.com/glennrp/libpng), [Pidgin](https://pidgin.im/), [Asterisk](https://www.asterisk.org/get-started) and [VLC Media Player](https://www.videolan.org/vlc/index.html). And vulnerability information was collected from [National Vulnerability Database(NVD)](https://nvd.nist.gov/) until the end of July 2017.

### Requirements for code:

 * [Tensorflow](https://www.tensorflow.org/)
 * [Keras](https://github.com/fchollet/keras/tree/master/keras)
 * [Scikit-learn](http://scikit-learn.org/stable/)
 * [Gensim](https://radimrehurek.com/gensim/)
 * Python >= 2.7

The dependencies can be installed using [Anaconda](https://www.anaconda.com/download/). For example:

```bash
$ bash Anaconda3-5.0.1-Linux-x86_64.sh
```

The "Data" folder contains the following sub folders:
1) VulnerabilityData -- It contains the vulnerable function data from 6 open source projects. Each file was named with the name of the project. For example, the FFmpeg.zip file contains the 4 .pkl files. The PKL file stores the Python object in binary form. The ffmpeg_list.pkl file stores all the FFmpeg functions (including vulnerable and non-vulnerable functions) in serialized AST forms. The ffmpeg_list_id.pkl file stores all the name of the functions as their IDs. The except_ffmpeg_list.pkl file stores the functions (in serialized AST forms) of the other 5 opens source projects. 

In the pre-training phase, the except_ffmpeg_list.pkl file is used as the historical data for training a LSTM network (the labels can be generated based on the functions IDs, please see the code for more details). Then, the ffmpeg_list.pkl file is used as the input to the pre-trained network for generating representations. Finally, the generated representations can be used as features for training a classifier. 

2) CodeMetrics -- It stores the code metrics of the open source projects. The code metrics are used as features to train a random forest classifier as the baseline to compare with the method which uses transfer-learned representations as features. We used [Understand](https://scitools.com/) which is a commercial code enhancement tool for extracting function-level code metrics. We included 23 code metrics extracted from the vulnerable functions of 6 projects.
 
3) TrainedTokenizer -- It contains the trained tokenizer file which is used for converting the serialized AST lists to numeric tokens.

4) TrainedWord2vecModel -- It includes the trained Word2vec model. The model was trained on the code base of 6 open source projects. The Word2vec model is used in the embedding layer of the LSTM network for converting input sequence to meaningful embeddings.

The "Code" folder contains the Python code samples. 
1) TransferableRepresentationLearning_LSTM_DNN.py file is for LSTM network training. It defines the structure of the Bi-LSTM network used in the paper. The input of the file is the historical vulnerable functions that have labels. The output of the file is a trained LSTM network capable of obtaining vulnerable function representations. 

2) ExtractLearnedFeaturesAndClassification.py file is for obtaining the function representations from the pre-trained LSTM network. It also includes the code for training a random forest classifier based on the obtained function representations as features.

3) CodeMetrics.py file is to train a random forest classifier based on the selected 23 code metrics.

The data in this repository is processed data (*.pkl files store the processed ASTs in serialized format). If you are interested in our project and would like to have the vulnerable functions we labeled, please contact junzhang@swin.edu.au. If you use our code and data in your work, please kindly cite our paper in your work. 

The latex format:

@article{lin2018cross,
  title={Cross-Project Transfer Representation Learning for Vulnerable Function Discovery},
  author={Lin, Guanjun and Zhang, Jun and Luo, Wei and Pan, Lei and Xiang, Yang and De Vel, Olivier and Montague, Paul},
  journal={IEEE Transactions on Industrial Informatics},
  year={2018},
  publisher={IEEE}
}

Thank you!
