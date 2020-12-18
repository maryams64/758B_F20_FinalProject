# Review-Level Recommendation Implementation Model
This is a basic implementation of a multi-input review-level neural network. The structure of this model was inspired by the following paper:

*Chen, Chong, et al. "Neural attentional rating regression with review-level explanations." Proceedings of the 2018 World Wide Web Conference. 2018.*

Contributors: Munish Khurana, Maryam Soomro, Ellen Zhang, Makaela Jackson

## Data Source

Digital Music dataset from Amazon Reviews Datasets provided by UCSD.

## Network Structure

Network aims to quantify the impact of userID and asin on the rating given to a product. 

Inputs: 
`reviewText_user`
`reviewText_item`

LSTM Target:
`overall_user`
`overall_item`

FCFN Target:
`overall`

- Inputs are fed into individual LSTM cells, one for reviewuser items and one for reviewitem items. 
- The two outputs from the LSTM cells are flattened and concatenated.
- The flattened output is fed into a FCFN to predict overall rating. 

## Environment
```
python 3.8.5
numpy 1.18.5
pandas 1.1.5
re 2.2.1
spacy 2.2.4
torch 1.7.0
sklearn 0.22.1
nltk 3.2.5
surprise 1.1.1
```
## Before Running the Code

Your local machine may need to have the following packages installed:
- spacy
- torch
- sklearn
- nltk
- surprise
- Microsoft Visual C++

Follow these steps to download the code onto your local machine:
`git clone "https://github.com/maryams64/758B_F20_FinalProject/"`

Run the following line to ensure latest version of repository is downlaoded:
`git pull "https://github.com/maryams64/758B_F20_FinalProject/"`

Now you are ready to run the code!

## Running the Code

Run the following line in your terminal (make sure you are within your local repository) to run the model:
`python main.py`

The code will take a few minutes to run but it will generate a similar output as this:
```
Epoch # 1
Training Set:
Average Loss: 0.0990  |  Average Accuracy: 0.2957

Validation Set:
Average Loss: 0.1061  |  Average Accuracy: 0.7586


Epoch # 2
Training Set:
Average Loss: 0.0954  |  Average Accuracy: 0.4029

Validation Set:
Average Loss: 0.1033  |  Average Accuracy: 0.7586


Epoch # 3
Training Set:
Average Loss: 0.0959  |  Average Accuracy: 0.3971

Validation Set:
Average Loss: 0.1014  |  Average Accuracy: 0.7586


Epoch # 4
Training Set:
Average Loss: 0.0940  |  Average Accuracy: 0.4319

Validation Set:
Average Loss: 0.0991  |  Average Accuracy: 0.7586


Epoch # 5
Training Set:
Average Loss: 0.0928  |  Average Accuracy: 0.4696

Validation Set:
Average Loss: 0.0974  |  Average Accuracy: 0.7586
```
## Comparison Code
In order to run the comparison code, run the following line in your terminal in the local repository: `python comparison.py`

The code will print the following:
- Random Forest Average Accuracy
- SVM Accuracy
- NMF Average Accuracy
- KNN Accuracy

Our graphs from our report are generated within Excel.


