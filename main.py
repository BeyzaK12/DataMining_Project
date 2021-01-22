from datetime import datetime
import os
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from PIL import Image

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Ignore warnings
import warnings

warnings.filterwarnings('ignore')


# for writing results to file
def write_file(title_, array_):
    file_name_ = title_ + ".txt"

    with open(file_name_, "w") as txt_file_:
        txt_file_.write(title_ + "\n\n")
        for line_ in array_:
            txt_file_.write(str(line_) + "\n")


# to split D: 70% for training, 30% for testing
def split_train_test(X_, y_):
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_,
                                                            train_size=0.7,
                                                            test_size=0.3)
    return X_train_, X_test_, y_train_, y_test_


# read image and apply preprocessing
def load_image(img_path, size_):
    # read image with image path 
    img = io.imread(img_path)
    
    # img.shape -> mostly (224, 224, 3)

    # convert an image with RGB channels into an image with a single grayscale channel
    img = rgb2gray(img)

    # resize image to match a certain size
    # anti_aliasing: apply a Gaussian filter to smooth the image prior to down-scaling
    img = resize(img, (size_, size_), anti_aliasing=False)

    # 2d array to 1d array
    img = img.reshape(size_ * size_)

    return img


def create_data(folder_path_, size_, limit_):
    X_, y_ = [], []

    folders = os.listdir(folder_path_)
    for folder in folders:
        current_time = datetime.now().strftime("%H:%M:%S")
        print(current_time, "|", folder, "| adding files...")

        path_ = "{}/{}".format(folder_path_, folder)
        files_ = os.listdir(path_)

        for no_, file_ in enumerate(files_):
            if no_ == limit_:
                break

            img_path = "{}/{}".format(path_, file_)
            img = load_image(img_path, size_)

            X_.append(img)
            y_.append(folder)

    X_ = np.array(X_)
    y_ = np.array(y_)

    return X_, y_


# Logistic Regression

def LogisticRegression_(X_train_, y_train_, X_test_, y_test_):
    scores_ = []

    for _ in range(5):
        clf = LogisticRegression(n_jobs=-1, C=100, max_iter=10000)
        clf = make_pipeline(StandardScaler(), clf)

        clf.fit(X_train_, y_train_)

        score_ = clf.score(X_test_, y_test_)

        scores_.append(score_)

    score_ = np.mean(np.array(scores_))

    return score_


def SGDClassifier_(X_train_, y_train_, X_test_, y_test_):
    scores_ = []

    for _ in range(5):
        clf = SGDClassifier(loss="log", max_iter=10000, n_jobs=-1)

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train_, y_train_)

        score_ = clf.score(X_test_, y_test_)

        scores_.append(score_)

    score_ = np.mean(np.array(scores_))

    return score_


# Neural Network-based Classification
sizes = []  # (512, 256, 128, 64, 32, 16, 8, 4, 2)

for i in range(9, 0, -1):
    sizes.append(2 ** i)

sizes = tuple(sizes)


def MLPClassifier_(X_train_, y_train_, X_test_, y_test_):
    # to keep classifiers in format that suitable
    # for using as parameter in VotingClassifier
    # ex: [('clf1', classifier), ...]
    classifiers_ = []

    scores_ = []

    # creation 10 classifiers
    for ii in range(4, -1, -1):
        classifier = MLPClassifier(hidden_layer_sizes=sizes[ii:], max_iter=1000)
        clf = make_pipeline(StandardScaler(), classifier)

        clf.fit(X_train_, y_train_)

        score_ = clf.score(X_test_, y_test_)
        scores_.append(score_)

        classifiers_.append(('clf{}'.format(ii + 1), clf))

    # 5  (32, 16, 8, 4, 2)
    # 6  (64, 32, 16, 8, 4, 2)
    # 7  (128, 64, 32, 16, 8, 4, 2)
    # 8  (256, 128, 64, 32, 16, 8, 4, 2)
    # 9  (512, 256, 128, 64, 32, 16, 8, 4, 2)

    return scores_, classifiers_


# Ensemble Learning-based Classification
def VotingClassifier_(X_train_, y_train_, X_test_, y_test_, classifiers_):
    # with using MLP Classifiers
    Voting_clf = VotingClassifier(estimators=classifiers_)
    Voting_clf = make_pipeline(StandardScaler(), Voting_clf)

    Voting_clf.fit(X_train_, y_train_)

    Voting_accuracy_ = Voting_clf.score(X_test_, y_test_)

    return Voting_accuracy_


def BaggingClassifier_(X_train_, y_train_, X_test_, y_test_):
    # for split and using in BaggingClassifier as base_estimator
    mlp_clsf = MLPClassifier(hidden_layer_sizes=(32, 16, 8, 4, 2), max_iter=1000)
    mlp_clsf = make_pipeline(StandardScaler(), mlp_clsf)

    subsets = []

    # split trainset into 5 equal pieces
    i1, i2, i3, i4, i5 = None, None, None, None, None

    # last indexes of train subsets
    for ii in range(5):
        ii_name = "i{}".format(ii + 1)

        # i1 = int(1 * (len(X_train_) / 5))
        # i2 = int(2 * (len(X_train_) / 5)) ...
        vars()[ii_name] = int((ii + 1) * (len(X_train_) / 5))

    subsets.append([X_train_[ 0:i1], y_train_[ 0:i1]])  # 1. subset
    subsets.append([X_train_[i1:i2], y_train_[i1:i2]])  # 2. subset
    subsets.append([X_train_[i2:i3], y_train_[i2:i3]])  # 3. subset
    subsets.append([X_train_[i3:i4], y_train_[i3:i4]])  # 4. subset
    subsets.append([X_train_[i4:i5], y_train_[i4:i5]])  # 5. subset

    # for comparing with Bagging scores
    mlp_scores_ = []
    for ii in range(5):
        mlp_X_train = subsets[ii][0]
        mlp_y_train = subsets[ii][1]

        mlp_clsf.fit(mlp_X_train, mlp_y_train)

        mlp_score_ = mlp_clsf.score(X_test_, y_test_)
        mlp_scores_.append(mlp_score_)


    bag_clf = BaggingClassifier(base_estimator=mlp_clsf, n_estimators=5)
    bag_clf = make_pipeline(StandardScaler(), bag_clf)

    bag_clf.fit(X_train_, y_train_)
    bag_score_ = bag_clf.score(X_test_, y_test_)

    return bag_score_, mlp_scores_


def run_all(log=True, sgd=True, mlp=True, voting=True, bagging=True, opt_write_file=True):
    print("")
    if log:
        current_time = datetime.now().strftime("%H:%M:%S")
        print(current_time, "| LogisticRegression_ START")

        Log_score = LogisticRegression_(X_train, y_train, X_test, y_test)

        current_time = datetime.now().strftime("%H:%M:%S")
        print(current_time, "| LogisticRegression_ DONE\n")
        
	# write results to file
        if opt_write_file:
            with open("Logistic_Score_{}_{}.txt".format(resize_, limit), "w") as txt_file_:
                txt_file_.write("LogisticRegression_ Score" + "\n\n" + str(Log_score))
    
            current_time = datetime.now().strftime("%H:%M:%S")
            print(current_time, "| Logistic_Score.txt DONE\n\n")

    if sgd:
        current_time = datetime.now().strftime("%H:%M:%S")
        print(current_time, "| SGDClassifier_ START")

        SGD_score = SGDClassifier_(X_train, y_train, X_test, y_test)
        
        current_time = datetime.now().strftime("%H:%M:%S")
        print(current_time, "| SGDClassifier_ DONE\n")
        
	# write results to file
        if opt_write_file:
            with open("SGD_Score_{}_{}.txt".format(resize_, limit), "w") as txt_file_:
                txt_file_.write("SGDClassifier_ Score" + "\n\n" + str(SGD_score))
    
            current_time = datetime.now().strftime("%H:%M:%S")
            print(current_time, "| SGD_Score.txt DONE\n\n")

    if mlp:
        current_time = datetime.now().strftime("%H:%M:%S")
        print(current_time, "| MLPClassifier_ START")

        MLP_scores, MLP_classifiers = MLPClassifier_(X_train, y_train, X_test, y_test)

        current_time = datetime.now().strftime("%H:%M:%S")
        print(current_time, "| MLPClassifier_ DONE\n")
        
	# write results to file
        if opt_write_file:
            write_file("MLP_Scores_{}_{}".format(resize_, limit), MLP_scores)
    
            current_time = datetime.now().strftime("%H:%M:%S")
            print(current_time, "| MLP_Scores.txt DONE\n\n")

        if voting:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(current_time, "| VotingClassifier_ START")
    
            Voting_score = VotingClassifier_(X_train, y_train, X_test, y_test, MLP_classifiers)

            current_time = datetime.now().strftime("%H:%M:%S")
            print(current_time, "| VotingClassifier_ DONE\n")
            
	    # write results to file
            if opt_write_file:
                with open("Voting_Score_{}_{}.txt".format(resize_, limit), "w") as txt_file_:
                    txt_file_.write("VotingClassifier_ Score" + "\n\n" + str(Voting_score))
        
                current_time = datetime.now().strftime("%H:%M:%S")
                print(current_time, "| Voting_Score.txt DONE\n\n")

    if bagging:
        current_time = datetime.now().strftime("%H:%M:%S")
        print(current_time, "| BaggingClassifier_ START")

        Bag_score, Bag_MLP_scores = BaggingClassifier_(X_train, y_train, X_test, y_test)
        
        current_time = datetime.now().strftime("%H:%M:%S")
        print(current_time, "| BaggingClassifier_ DONE\n")
        
	# write results to file
        if opt_write_file:
            with open("Bagging_Score_{}_{}.txt".format(resize_, limit), "w") as txt_file_:
                txt_file_.write("BaggingClassifier_ Score" + "\n\n" + str(Bag_score))
    
            current_time = datetime.now().strftime("%H:%M:%S")
            print(current_time, "| Bagging_Score.txt DONE")
            
            write_file("Bagging-MLP_Scores_{}_{}".format(resize_, limit), Bag_MLP_scores)
            
            current_time = datetime.now().strftime("%H:%M:%S")
            print(current_time, "| Bagging-MLP_Scores.txt DONE")


folder_path = "tr_signLanguage_dataset"
resize_ = 80    # img.shape(80,80)
limit = 500     # -1 for no limit

X, y = create_data(folder_path, resize_, limit)     # (11500, 6400)
# X, y = load_digits(return_X_y=True)     # (1257, 64)

X_train, X_test, y_train, y_test = split_train_test(X, y)
run_all(opt_write_file=True)
