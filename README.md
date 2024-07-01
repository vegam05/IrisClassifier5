## Setup:
```
git clone https://github.com/vegam05/IrisClassifier5
cd IrisClassifier5
pip install -r requirements.txt
```
Then run all codeblocks of ```driver.ipynb```
This will generate/overwrite 2 files: iris_classification_model.h5( the trained model) and classes.npy(numpy file having various flower classes to be used by streamlit)
Then run:
```
streamlit run irisClassifier.py
```
The webapp should be launched automatically or just open your localhost url to access it.
## Usage:
Enter :
* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)
  
of the Iris flower you want to know about and based on that the model will classify it as a specific species.
Dataset acquired from:[Kaggle](https://www.kaggle.com/datasets/uciml/iris)


