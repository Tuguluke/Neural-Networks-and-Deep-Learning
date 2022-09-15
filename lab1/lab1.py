import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def create_categories(x):
    '''
    change tech column to 1,
    non-tech to 0
    '''
    if x == "Tech":
        return 1
    else:
        return 0


def add_embedding(data):
    '''
    adding doc2vec column based on text
    '''
    # tokenize and tag the card text
    card_docs = [TaggedDocument(doc.split(' '), [i])
                 for i, doc in enumerate(data.text)]

    # instantiate model
    model = Doc2Vec(vector_size=64, window=2,
                    min_count=1, workers=8, epochs=40)
    # build vocab
    model.build_vocab(card_docs)
    # train model
    model.train(card_docs, total_examples=model.corpus_count,
                epochs=model.epochs)
    card2vec = [model.infer_vector((data['text'][i].split(' ')))
                for i in range(0, len(data['text']))]
    # Create a list of lists
    dtv = card2vec
    # set list to dataframe column
    data['doc2vec'] = dtv
    return data


def split_data(test_ratio=None):
    '''
    split the data with
    test_radio input: .3
    '''
    return train_test_split(X, y, test_size=test_ratio, random_state=1)


def train_with_activation_funciton_nnodes(activation_function=None, n_hidden_nodes=None):
    # Standardize data
    stdsc = StandardScaler()
    stdsc.fit(X_train)
    X_train_std = stdsc.transform(X_train)
    X_test_std = stdsc.transform(X_test)
    # activation{'identity', 'logistic', 'tanh', 'relu'}, default='relu'
    mlp = MLPClassifier(activation=activation_function, hidden_layer_sizes=[
                        n_hidden_nodes, n_hidden_nodes])

    mlp.fit(X_train_std, y_train)
    y_predicted = mlp.predict(X_test_std)
    # other metric
    report = metrics.classification_report(
        y_test, y_predicted, output_dict=True)
    # plot
    mat = metrics.confusion_matrix(y_test, y_predicted)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.title("Activaton Function: {}, Hidden layer: {}\n Acuracy: {}".format(
        activation_function, n_hidden_nodes, mlp.score(X_test_std, y_test)))
    plt.show()

    return report


def training_by_percentage(percentage=None):
    start_num_epochs = 10
    finish_num_epochs = 200
    inc_amt = 10

    pred_scores = []
    num_epochs = []
    # standard then slice
    X_train_slice = X_train[:round(percentage*len(X_train))]
    X_train_std_slice = stdsc.transform(X_train_slice)
    X_test_std = stdsc.transform(X_test)
    y_train_slice = y_train[:round(percentage*len(y_train))]

    for epoch_count in range(start_num_epochs, finish_num_epochs, inc_amt):
        my_classifier = MLPClassifier(
            activation='relu', random_state=20, max_iter=epoch_count)
        my_classifier.fit(X_train_std_slice, y_train_slice)
        score = my_classifier.score(X_test_std, y_test)
        pred_scores.append(score)
        num_epochs.append(epoch_count)

#     plt.figure()
    plt.plot(num_epochs, pred_scores, 'r-+', linewidth=2)
    plt.xlabel("Num of epochs")
    plt.ylabel("Accuracy")
    plt.title(
        "Impact of number of training epochs, percentage: {}".format(percentage))
    plt.show()


if __name__ == "__main__":

    tech_data = pd.read_csv('tech_nontech_classification_annotations.csv')
    tech_data = add_embedding(tech_data)
    tech_data["binary"] = tech_data['accept'].apply(create_categories)
    X = (np.vstack(np.array(tech_data['doc2vec'])))
    y = np.array(tech_data['binary'])
    X_train, X_test, y_train, y_test = split_data(test_ratio=.3)
    # part1:
    reports = {}
    for hiden in [10, 20, 30]:
        for act_func in ['relu', 'logistic', 'tanh']:
            report = train_with_activation_funciton_nnodes(
                activation_function=act_func, n_hidden_nodes=hiden)
            reports["Activaton Function: {}, Hidden layer: {}"
                    .format(act_func, hiden)] = report
    # print(pd.DataFrame(reports).T[['0','1','accuracy']].to_latex())
    # part2:
    percent_slice = [.2, .4, .6, .8, 1.0]
    for percentage in percent_slice:
        training_by_percentage(percentage=percentage)
