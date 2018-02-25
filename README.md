## Transferable Representation Learning

Hi there, welcome to this pape!

### Instructions:

The Vulnerabilities_info.xlsx file contains information of the collected function-level vulnerabilities. These vulnerabilities are from 6 open source projects: [FFmpeg](https://github.com/FFmpeg/FFmpeg), [LibTIFF](https://github.com/vadz/libtiff), [LibPNG](https://github.com/glennrp/libpng), [Pidgin](https://pidgin.im/), [Asterisk](https://www.asterisk.org/get-started) and [VLC Media Player](https://www.videolan.org/vlc/index.html) . And vulnerability information was collected from [National Vulnerability Database(NVD)](https://nvd.nist.gov/) until the end of July 2017.

The "Data" folder contains the following subfolders:
1) VulnerabilityData -- It contains the vulnerable function data from 6 open source projects. Each file was named with the name of the project. For example, the FFmpeg.zip file contains the 4 .pkl files. The PKL file stores the Python object in binery form. The ffmpeg_list.pkl file stores all the FFmpeg functions (including vulnerable and neutral functions) in serialized AST forms. The ffmpeg_list_id.pkl file stores all the name of the functions as this IDs. The except_ffmpeg_list.pkl file stores the functions (in serialized AST forms) of the other 5 opens source projects.
2) CodeMetrics -- It stores the code metrics of the open source projects. The code metrics are used as features to train a random forest classifier as the baseline to compare with the method which uses transfer-learned representations as features. We used [Understand](https://scitools.com/) which is a commercial code enhancement tool for extracting function-level code metrics. We included 23 code metrics extracted from the vulnerable functions of 6 projects. 
3) TrainedTokenizer -- It contains the trained tokenizer file which is used for converting the serialized AST lists to numeric tokens.
4) TrainedWord2vecModel -- It includes the trained Word2vec model on the code base of 6 open source projects. The Word2vec model is used in the embedding layer of the LSTM network for converting input sequence to meaningful embeddings.

The "Code" folder contains the Python code samples. 
1) TransferableRepresentationLearning_LSTM_DNN.py file is for LSTM network training. The input of the file is the historical vulnerable functions that have labels. The output of the file is a trained LSTM network capable of obtaining vulnerable function representations. 
2) ExtractLearnedFeaturesAndClassification.py file is to obtain the function representations from the pre-trained LSTM network. It also includes the code for training a random forest classifier based on the obtained function representations as features.
3) CodeMetrics.py file is to train a random forest classifer based on the selected 23 code metrics.

Thanks!