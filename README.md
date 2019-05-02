# Math Expression Extraction
CRF and RNN models to extract inline math expressions from unstructured plain text.

## Folder Structure
* CRF - Contains code of CRF model
* RNN - Contains the code of RNN models
* Data - Training data

## Training Models
* Run models in CRF and RNN using "python <filename>.py"
* All models are evaluated using 10-fold cross validation
* CRF model is in CRF/Train_CRF.py
* There are 3 RNN models
    1. LSTM with word embeddings - Train_LSTM.py
    2. Bi-LSM with word embeddings - Train_Bi-LSTM.py
    3. Bi-LSTM with word and character embeddings - Train_W-CH-Bi-LSTM.py

  
