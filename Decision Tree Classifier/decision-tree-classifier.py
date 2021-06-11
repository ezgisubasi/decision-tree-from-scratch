import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import random
from pprint import pprint


def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):  # excluding the last column which is the label
        values = data[:, column_index]
        unique_values = np.unique(values)

        potential_splits[column_index] = unique_values

    return potential_splits


def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_left = data[split_column_values <= split_value]
        data_right = data[split_column_values > split_value]

    # feature is categorical
    else:
        data_left = data[split_column_values == split_value]
        data_right = data[split_column_values != split_value]

    return data_left, data_right


def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy


def calculate_gini(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    gini = 1 - sum(probabilities ** 2)

    return gini


def information_gain(data_left, data_right, method):
    n_data_points = len(data_left) + len(data_right)

    p_data_left = len(data_left) / n_data_points
    p_data_right = len(data_right) / n_data_points

    if method == "entropy":
        left_entr = calculate_entropy(data_left)
        right_entr = calculate_entropy(data_right)

        overall_entropy = (p_data_left * left_entr + p_data_right * right_entr)

        information_gain = (left_entr + right_entr) - overall_entropy

    if method == "gini":
        left_entr = calculate_gini(data_left)
        right_entr = calculate_gini(data_right)

        overall_gini = (p_data_left * left_entr +
                        p_data_right * right_entr)

        information_gain = (left_entr + right_entr) - overall_gini

    return information_gain


def determine_best_split(data, potential_splits, method):
    if method == "entropy":
        i_g = 0
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_left, data_right = split_data(data, split_column=column_index, split_value=value)
                current_ig = information_gain(data_left, data_right, method)

                if current_ig >= i_g:
                    i_g = current_ig
                    best_split_column = column_index
                    best_split_value = value

    if method == "gini":
        i_g = 9999
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_left, data_right = split_data(data, split_column=column_index, split_value=value)
                current_ig = information_gain(data_left, data_right, method)

                if current_ig <= i_g:
                    i_g = current_ig
                    best_split_column = column_index
                    best_split_value = value

    return best_split_column, best_split_value


def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")

    return feature_types


def decision_tree_algorithm(df, counter=0, method="gini", min_samples=2, max_depth=5):
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df

    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)

        return classification

    # recursive part
    else:
        counter += 1

        # helper functions
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits, method)
        data_left, data_right = split_data(data, split_column, split_value)

        # check for empty data
        if len(data_left) == 0 or len(data_right) == 0:
            classification = classify_data(data)
            return classification

        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)

        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)

        # instantiate sub-tree
        sub_tree = {question: []}

        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_left, counter, method, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_right, counter, method, min_samples, max_depth)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":  # feature is continuous
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer

    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)


def calculate_accuracy(df, tree, label):
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["classification_correct"] = df["classification"] == df[label]

    accuracy = df["classification_correct"].mean()

    return accuracy


if __name__ == "__main__":
    df = pd.read_csv("data/Titanic.csv")
    # handling missing values
    median_age = df.Age.median()
    mode_embarked = df.Embarked.mode()[0]

    df = df.fillna({"Age": median_age, "Embarked": mode_embarked})

    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    cols = df.columns.tolist()
    cols = cols[2:] + cols[:1]
    df = df[cols]

    df.Survived[df.Survived == 0] = 'Died'
    df.Survived[df.Survived == 1] = 'Survived'


    # Gini #

    train_df, test_df = train_test_split(df, 0.2)
    tree = decision_tree_algorithm(train_df, counter=0, method="gini", min_samples=2, max_depth=20)
    accuracy = calculate_accuracy(test_df, tree, "Survived")

    #pprint(tree, width=50)
    print("With Gini: ",accuracy)

    # Entropy #

    train_df, test_df = train_test_split(df, 0.2)
    tree = decision_tree_algorithm(train_df, counter=0, method="entropy", min_samples=2, max_depth=20)
    accuracy = calculate_accuracy(test_df, tree, "Survived")

    #pprint(tree, width=50)
    print("With Entropy: ",accuracy)

    # Hyperparameter Tuning - Max Depth #

    max_depth_gini = []
    max_depth_entropy = []
    x = [10,15,20,25,30,35,40,45,50]
    for i in x:
        train_df, test_df = train_test_split(df, 0.2)
        tree = decision_tree_algorithm(train_df, counter=0, method="gini", min_samples=2, max_depth=i)
        accuracy = calculate_accuracy(test_df, tree, "Survived")
        max_depth_gini.append(accuracy)

        tree2 = decision_tree_algorithm(train_df, counter=0, method="entropy", min_samples=2, max_depth=i)
        accuracy2 = calculate_accuracy(test_df, tree2, "Survived")
        max_depth_entropy.append(accuracy2)

    print(max_depth_gini)
    print(max_depth_entropy)

    # plotting the line 1 points
    plt.plot(x, max_depth_gini, label="max_depth_gini")
    # plotting the line 2 points
    plt.plot(x, max_depth_entropy, label="max_depth_entropy")
    plt.xlabel('max depth')
    # Set the y axis label of the current axis.
    plt.ylabel('gini/entropy')
    # Set a title of the current axes.
    plt.title('Max Depth Hyperparameter Tuning')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()

    # Hyperparameter Tuning - Min Samples #

    min_samples_gini = []
    min_samples_entropy = []
    x = [2,3,4,5,9,10,15]
    for i in x:
        train_df, test_df = train_test_split(df, 0.2)
        tree = decision_tree_algorithm(train_df, counter=0, method="gini", min_samples=i, max_depth=20)
        accuracy = calculate_accuracy(test_df, tree, "Survived")
        min_samples_gini.append(accuracy)

        tree2 = decision_tree_algorithm(train_df, counter=0, method="entropy", min_samples=i, max_depth=20)
        accuracy2 = calculate_accuracy(test_df, tree2, "Survived")
        min_samples_entropy.append(accuracy2)

    print(min_samples_gini)
    print(min_samples_entropy)

    # plotting the line 1 points
    plt.plot(x, min_samples_gini, label="min_samples_gini")
    # plotting the line 2 points
    plt.plot(x, min_samples_entropy, label="min_samples_entropy")
    plt.xlabel('min samples')
    # Set the y axis label of the current axis.
    plt.ylabel('gini/entropy')
    # Set a title of the current axes.
    plt.title('Min Samples Hyperparameter Tuning')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()

    # One Hot Encoding #

    obj_df = df[["Embarked", "Sex"]]
    data_encoded = pd.get_dummies(data=obj_df)
    df.drop(["Embarked", "Sex"], axis=1, inplace=True)
    df = df.join(data_encoded)

    # Scikit Learn #

    from sklearn.model_selection import train_test_split
    X = df.loc[:, df.columns != 'Survived']
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # import dtree
    from sklearn.tree import DecisionTreeClassifier

    # Gini #

    clf = DecisionTreeClassifier(random_state=20, criterion="gini", min_samples_leaf=2, max_depth=20)
    clf.fit(X_train, y_train)

    print("Accuracy of scikit-learn -gini-:", clf.score(X_test, y_test))

    X = df.loc[:, df.columns != 'Survived']
    y = df["Survived"]

    from sklearn import tree

    fig = plt.figure(figsize=(50, 50))
    _ = tree.plot_tree(clf,
                       feature_names=X.columns.values,
                       class_names=np.unique(y),
                       filled=True)

    fig.savefig("decistion_tree-gini.png")

    # Entropy #

    clf = DecisionTreeClassifier(random_state=20, criterion="entropy", min_samples_leaf=2, max_depth=20)
    clf.fit(X_train, y_train)

    print("Accuracy of scikit-learn -entropy-:", clf.score(X_test, y_test))

    X = df.loc[:, df.columns != 'Survived']
    y = df["Survived"]

    from sklearn import tree

    fig = plt.figure(figsize=(50, 50))
    _ = tree.plot_tree(clf,
                       feature_names=X.columns.values,
                       class_names=np.unique(y),
                       filled=True)

    fig.savefig("decistion_tree-entropy.png")


















