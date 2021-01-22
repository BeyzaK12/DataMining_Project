# 160709017 Beyza Kurt |Data Mining Final Project

Description:
  The data set is taken from https://www.kaggle.com/berkaykocaoglu/tr-sign-language:
  * There are no images of dotted or capped letters (i, ç, ş, ü, ö, ğ).
  * It consists of 2 main folders as "train" and "test".
  * There are at least 4000 photos for each letter in the "train" folder, and at least 1500 photos for those in the "test" folder.
  * Apart from letters, there are 3 more folders named "del", "nothing" and "space" in "train" and "test" folders.

  In the data set made ready for use in the project:
  * "del", "nothing" and "space" folders have been removed.
  * When it was understood that the program could not use all images due to lack of hardware,
    the folders of each letter in the "train" were moved to the main location with the first 2000 images.
    Thus, "train" and "test" files were removed and the data set in the code was divided into "train" and "test".

  Each image was read using skimage.io.imread, made 2-dimensional using skimage.color.rgb2gray(),
  resized as 80x80 using skimage.transform.resize() and finally unidimensional using np.reshape().  

  The results were observed by creating the following models and applying them to the mentioned data set:
  * Sklearn's linear models: LogisticRegression and SGDClassifier
  * Sklearn's neural network-based models: MLPClassifier
  * Sklearn's ensemble learning-based: VotingClassifier and BaggingClassifier
  
  In addition, the above mentioned models were tested with
  Sklearn's Digits data set and the results are included in the project report.
  
  Apart from these, Keras's Conv2D model was tested by using the entire data set,
  without preprocessing the images, and the results are included in the project report.


Team & Roles: Beyza Kurt | all-in-one member


Structure:
  - Report_LaTeX:             Latex files of the project report
   |-> images
   |-> main.tex
   
  - images:                   Sample plots
   |-> A_original.png
   |-> A_transformed_2.png
   |-> thresholding.png
   |-> A_original_2.png
   |-> morphological_snakes.png
   |-> A_transformed.png
   |-> structural_similarity_index.png
   
  - results:                  Results of the models
   |-> Bagging-MLP_Scores_80_500.txt
   |-> Logistic_Score_digit_digit.txt
   |-> Bagging-MLP_Scores_digit_digit.txt
   |-> MLP_Scores_80_500.txt
   |-> Bagging_Score_80_500.txt
   |-> MLP_Scores_digit_digit.txt
   |-> Bagging_Score_digit_digit.txt
   |-> SGD_Score_digit_digit.txt
   |-> Keras_Score.txt
   |-> Voting_Score_80_500.txt
   |-> Keras_results.txt
   |-> Voting_Score_digit_digit.txt
   |-> Logistic_Score_80_500.txt
   
  - tr_signLanguage_dataset:  Files of the data set
   |-> A
   |-> B
   |...
   |-> Z
   
  - Report.pdf:               Project report in pdf format
  - keras_main.py:            Python code for Conv2D with Keras
  - main.py:                  Python code for subjects
  - plot_data.py:             Python code for plot sample's original
                              and after preprocessing forms 


Language, version, and main file: 
  * Python 3.8
  - main.py
