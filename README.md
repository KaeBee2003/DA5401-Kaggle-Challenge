
# Problem Statement

The challenge is about evaluation of conversational AI, the ability to automatically determine whether a test case (prompt-response pairs) is aligning with a specific evaluation metric.

Given:
- **Input 1**: Metric definition (as text embedding) - npy file
- **Input 2**: Prompt and response text pair - JSON file

Predict:
- **Output**: Fitness score on a scale of 1-10

This is a regression problem within a metric learning context, where the model learns semantic distance between evaluation intent and test cases.

---
# Data Summary

| File                         | Purpose                 | Content                                   |
| ---------------------------- | ----------------------- | ----------------------------------------- |
| `train_data.json`            | Training data           | Labeled prompt-response pairs with scores |
| `test_data.json`             | Test data               | Unlabeled prompt-response pairs           |
| `metric_names.json`          | Metric catalogue        | List of all evaluation metric names       |
| `metric_name_embeddings.npy` | Pre-computed embeddings | 256-dimensional vectors for each metric   |
| `sample_submission.csv`      | Submission template     | Expected output format                    |

>Train size: 5000
>Test size: 3638

### `metric_name_embeddings.npy` and `metric_names.json`

- `metric_names.json` contains the list of metric names
- Because embeddings are expensive to compute, and generating them for every metric names is time consuming, the assignment has provided them **already computed**.

### `train_data.json` and `test_data.json`

#### The JSON file contains these features.

`metric_name`  - This is the label for whatever metric is being evaluated.
`score`  - The evaluated score of how well the response is... with respect to the metric.
`user_prompt` - message the user sent to the model/chatbot
`response` - the assistant’s reply to the user prompt
`system_prompt`  - additional instructions that guided the model's response behavior

**The train data has all of these features, but the test data have all columns except `score` which is the target Variable**  

## Score distribution
##### Train Score Distribution

```
Score │ Count
──────────────────────────────────────────────
  0   │ 
  1   │ 
  2   │ 
  3   │ 
  4   │ 
  5   │ 
  6   │ █
  7   │ ██
  8   │ ████
  9   │ ██████████████████████████████████████████████████████████████
 10   │ █████████████████████████████████████████████████████████████████████████████
```

##### Sample Score Distribution

```
Score │ Count
──────────────────────────────────────────────
  1   │ ███████████
  2   │ █████████
  3   │ ███████████
  4   │ █████████
  5   │ ███████████
  6   │ █████████
  7   │ ███████████
  8   │ █████████
  9   │ ███████████
 10   │ ████████████████████
```

---
‎ 
‎ 
‎ 
 
# Code Summary

> The code contains a lot of functions defined, mainly because of re-usability.


## Importing the files

(I made a compatible code for reusing in the Kaggle, I ran this code locally in the beginning)
```python
DATA_DIR = Path(".")
TRAIN_JSON = DATA_DIR / "train_data.json"
TEST_JSON = DATA_DIR / "test_data.json"
METRIC_NAMES_JSON = DATA_DIR / "metric_names.json"
METRIC_EMB_NPY = DATA_DIR / "metric_name_embeddings.npy"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

TRAIN_EMB = Path("Train_pr_emb.npy")
TEST_EMB = Path("Test_pr_emb.npy")
SUB_OUT = "submission.csv"
```

## Functions

```python
def create_dataframe():
	# This function will convert JSON into a pandas DataFrame with columns:
	# `ID` (integer)
	# `metric_name` (string)
	# `combined_text` (string combining user_prompt, response, system_prompt) 
	# `metric_name_embedding` (as numpy array)
	# `score` if is_train = True 
	
def ensure_array():
	# take the pandas Series whose values are numpy arrays (or lists) and stack them into a 2-D numpy array.
	
def compute_features():
	# Computes cosine similarity, and concatenates them
	
def preprocess_embeddings():
	# ensures metric_name_embedding and context_embedding using ensure_array
	# applies cosine similarity to metric_name_embedding and context_embedding using compute_features
	# if is_train = True , convert "score" to float
	
def build_and_save_submission():
	# prediction may be higher than 10 or lower than 0, it is cliped to (1,10), combines "ID" and "scores", as for the submission format

def get_embeddings():
	# sees whether embeddings are already calculated, if Yes, it loads directly
	# if embeddings are not present, it is generated using the model
	# batch size is reduced to run the model locally in my laptop
	# if there are fewer embeddings than required rows it adds rows of zeros.
	# if there are too many embeddings, it chops off the extra rows.
	# save the file if freshly created

def main():
	# The final main function, which does the following
	# imports all the necessary data
	# gets them into Dataframe format
	# creates embedding if required for both train and test
	# preapre the data for train and test
	# implements MLP regressor with train data
	# final predictions are created using test data
	# K-Fold validation is done for 5 fold during training
	# log error is used
	
```

---
# Implementation Strategy

#### Embedding
`google/embeddinggemma-300m` model was used because it was specified in the challenge. I wasn't able to run this model in my laptop initially, so I had to use light weight model `intfloat/multilingual-e5-base` which was compatible with my GPU's VRAM availability. Later I was able to use `google/embeddinggemma-300m` itself by reducing the batch size.

#### Preprocessing
- There were several missing values in `system_prompt` , they were filled with ""
- user_prompt, response, system_prompt were combined, before generating embeddings
- embeddings were generated using the specified model, and reduced batch size
- Cosine similarity as a feature was  added, to reduce model's burden, ε was added to avoid zero division error
```math
cosine = \frac{\mathbf{m} \cdot \mathbf{c}}{\lVert \mathbf{m} \rVert , \lVert \mathbf{c} \rVert + \varepsilon}  
```

- PCA was done on the model to reduce the features from around 513 to 478, since 478 features were required to explain 95% of variance 
- The target Values were transformed, to over come the issue of right skew, before training.

```math
y_{\text{transformed}} = \log\left(1 + y_{\text{raw}}\right)
```

 - And inverse transform was also applied to the predicted values 

```math
y_{\text{pred}} = e^{y_{\text{pred\_log}}} - 1
```


- The new reduced features was sent to MLP for training


#### MLP Architechture 
Architecture: Two hidden layers (256 → 128 neurons)
Activation Function: Hyperbolic tangent (tanh)
Max Iterations: 1000 epochs
Random State: 42

#### Training

Logarithmic transformation is used
- The training score distribution is Right-skewed (from the plots)
- Prevents the model from being too sensitive to extreme high value scores


#### Cross Validation 

 K-Fold Cross-Validation have been used because.
 - Makes the model more robust, prevent over-fit
 - Better than single train-test split
 

```python
n_splits=5 # 5 Fold Validation

shuffle=True # Randomisation for more robustness
```

- K value is set to 5
- Train on 80% of data
- Validate on remaining 20%
- RMSE has been used to evaluate each fold, both for log transformed values as well as re transformed (original) values

| Fold     | Log RMSE            | Original RMSE       |
| -------- | ------------------- | ------------------- |
| 1        | 0.2711              | 2.2543              |
| 2        | 0.2590              | 2.2744              |
| 3        | 0.2868              | 2.3463              |
| 4        | 0.2728              | 2.2269              |
| 5        | 0.2913              | 2.3459              |
| **Mean** | **0.2762 ± 0.0116** | **2.2896 ± 0.0486** |
> Low Variance Suggests that there was stable performance and no significant overfitting or fold-specific anomalies (x̄ ± σ format has been used)

#### Prediction

The final trained model was used to predict the scores in test data, and were clipped to (0,10)
##### Test Score Distribution (After Running the Model)

```
Score │ Count
──────────────────────────────────────────────
  1   │ ██
  2   │ ████
  3   │ ██████
  4   │ █████████
  5   │ ████████████
  6   │ ████████████████
  7   │ ████████████████████
  8   │ ████████████████████████
  9   │ ████████████████████████████
 10   │ ████████████████████████████████
```

**Test Set Predictions:**
Distribution still slightly aligns with training data (right-skewed) but there are no extreme outliers after clipping. Smooth distribution suggests that generalization is good

#### Overall process Summary

```
Raw JSON Data
    ↓
Text Combination & DataFrame Creation
    ↓
Embedding Generation (Cached)
    ↓
Feature Engineering (Cosine Similarity)
    ↓
Preprocessing (Scaling + PCA)
    ↓
Log Transformation of Target
    ↓
MLP Training with 5-Fold CV
    ↓
Ensemble Predictions
    ↓
Inverse Transform & Clipping
    ↓
Submission File
```

