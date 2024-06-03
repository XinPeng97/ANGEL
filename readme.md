# ANGEL

## Dataset
To investigate the effectiveness of AMPLE, we adopt three vulnerability datasets from these paper: 
* Big-Vul: <https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing>
* Reveal: <https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy>
* Devign: <https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF>


## Requirement
Our code is based on Python3. There are a few dependencies to run the code. The major libraries are listed as follows:
* torch
* torch_geometric
* dgl
* numpy
* sklearn

## 📥 Guide

#### 1、Preprocessing

- (1) **Joern**:
  
  We download Joern [here](https://github.com/joernio/joern).

- (2) **Parse**:
  
  Follow the Joern documentation to generate a code property graph.

#### 2、Word2Vec
-  (3) **Word2Vec**:
For code property graph, we use the word2vec to initialize the node representation.
```bash
python word2vec.py
```

#### 3、Training
-  (4) **Model Training**:
```bash
python main.py --dataset 'Big_Vul'
```


## Acknowledgement
--[https://github.com/AMPLE001/AMPLE/tree/main](https://github.com/AMPLE001/AMPLE/tree/main)
