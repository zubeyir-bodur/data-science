from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_data():
    """
    Read the data from the .csv file.
    :return:
    """
    DATA = pd.read_csv("falldetection_dataset.csv", encoding="ISO-8859-2", header=None)
    X_TRAIN = np.array(DATA[[i + 2 for i in range(DATA.shape[1] - 2)]])
    Y_TRAIN = np.array(DATA[1])
    fall_detector = lambda x: (x != "N") & (x == "F")
    Y_TRAIN = fall_detector(Y_TRAIN).astype(int)
    return X_TRAIN, Y_TRAIN


def min_max_scale(ndarray):
    """
    Min-max normalization meets the needs.
    However, better approaches can be used
    such as L1, L2, max-abs, gaussian normalization ...
    :param ndarray:
    :return:
    """
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(ndarray)
    return transformed, scaler


def inverse_transform(transformed, scaler):
    return scaler.inverse_transform(transformed)


def plot_pca(x_norm_reduced, y, show_labels=True):
    x_norm_reduced_f = []
    x_norm_reduced_nf = []
    for i in range(x_norm_reduced.shape[0]):
        if y[i]:
            x_norm_reduced_f.append(x_norm_reduced[i])
        else:
            x_norm_reduced_nf.append(x_norm_reduced[i])
    x_norm_reduced_f = np.array(x_norm_reduced_f)
    x_norm_reduced_nf = np.array(x_norm_reduced_nf)
    plt.figure(figsize=(8, 6), dpi=90)
    if show_labels:
        plt.scatter(x=x_norm_reduced_f[:, 0], y=x_norm_reduced_f[:, 1], c="#FF0000", s=1.5)
        plt.scatter(x=x_norm_reduced_nf[:, 0], y=x_norm_reduced_nf[:, 1], c="#00FF00", s=1.5)
    else:
        plt.scatter(x=x_norm_reduced[:, 0], y=x_norm_reduced[:, 1], c="#0000FF", s=1.5)
    plt.xlabel("First PCA component")
    plt.ylabel("Second PCA component")
    plt.legend(labels=["Fall", "Non-Fall"])
    plt.title("PCA Visualization of Telehealth Data")


def plot_clustered_data(x_norm_reduced, y_predict, n_clusters):
    # Initialize k empty arrays
    x_norm_reduced_j = [[] for _ in range(n_clusters)]

    # Colors
    c = ['#00FF00', '#FF0000', '#0000FF', '#F0F000',
         '#00F0F0', '#F000F0', '#FFA500', '#FFC0CB',
         '#000000', '#851E1E']
    for i in range(x_norm_reduced.shape[0]):
        for j in range(n_clusters):
            if j == y_predict[i]:
                x_norm_reduced_j[j].append(x_norm_reduced[i])

    plt.figure(figsize=(8, 6), dpi=90)
    for j in range(n_clusters):
        x_norm_reduced_j_first_pca = []
        x_norm_reduced_j_second_pca = []
        for m in range(len(x_norm_reduced_j[j])):
            x_norm_reduced_j_first_pca.append(x_norm_reduced_j[j][m][0])
            x_norm_reduced_j_second_pca.append(x_norm_reduced_j[j][m][1])
        plt.scatter(x=x_norm_reduced_j_first_pca, y=x_norm_reduced_j_second_pca, c=c[j], s=1.5)
    plt.xlabel("First PCA component")
    plt.ylabel("Second PCA component")
    plt.legend(labels=[f"Cluster {i}" for i in range(n_clusters)])
    plt.title(f"K-Means Visualization of Telehealth Data w/ K={n_clusters}")


def export_excel(results_sorted, is_svm):
    """
    Writes a dictionary into an Excel file
    :param results_sorted:
    :param is_svm:
    :return:
    """
    data = pd.DataFrame(results_sorted)
    if is_svm:
        writer = pd.ExcelWriter('plots/SVM_validation_results.xlsx')
    else:
        writer = pd.ExcelWriter('plots/MLP_validation_results.xlsx')
    data.to_excel(writer, 'Sheet 1', index=False)
    writer.save()


def find_best_svm(x_train, y_train, x_val, y_val, c_values, kernel_types, degrees, gamma_values):
    """
    Compute the best SVM by training
    Record the training histories to a table
    :param x_train:
    :param y_train:
    :param x_val:
    :param y_val:
    :param c_values:
    :param kernel_types:
    :param degrees:
    :param gamma_values:
    :return:
    """
    results = []
    for c_ in c_values:
        for kernel_type in kernel_types:
            for gamma in gamma_values:
                if kernel_type == "poly":
                    for degree in degrees:
                        print(f"\n\nOn the setting c={c_}, kernel={kernel_type}, gamma={gamma}, degree={degree}")
                        model = SVC(C=c_, kernel=kernel_type, degree=degree,
                                    gamma=gamma, max_iter=100000, random_state=42)
                        model.fit(x_train, y_train)
                        predictions = model.predict(x_val)
                        val_accuracy = metrics.accuracy_score(y_val, predictions) * 100
                        row = {'Regularization Parameter': c_,
                               'Kernel Type': kernel_type,
                               'Degree': degree,
                               'Kernel Coefficient': gamma,
                               'Validation Accuracy (%)': val_accuracy}
                        results.append(row)
                        print(row)
                else:
                    print(f"\n\nOn the setting c={c_}, kernel={kernel_type}, gamma={gamma}, degree=NULL")
                    model = SVC(C=c_, kernel=kernel_type, gamma=gamma,
                                max_iter=100000, random_state=42)
                    model.fit(x_train, y_train)
                    predictions = model.predict(x_val)
                    val_accuracy = metrics.accuracy_score(y_val, predictions) * 100
                    row = {'Regularization Parameter': c_,
                           'Kernel Type': kernel_type,
                           'Degree': "NULL",
                           'Kernel Coefficient': gamma,
                           'Validation Accuracy (%)': val_accuracy}
                    results.append(row)
                    print(row)
    results_sorted = sorted(
        results, key=lambda x: x['Validation Accuracy (%)'], reverse=True)
    export_excel(results_sorted, is_svm=True)
    print('\n\nSVM data is written to an Excel file successfully.')

    best_SVM_params = results_sorted[0]
    c_best = best_SVM_params["Regularization Parameter"]
    kernel_type_best = best_SVM_params['Kernel Type']
    degree_best = best_SVM_params['Degree']
    gamma_value_best = best_SVM_params['Kernel Coefficient']
    best_val_acc = best_SVM_params['Validation Accuracy (%)']
    return c_best, kernel_type_best, degree_best, gamma_value_best, best_val_acc


def find_best_mlp(x_train, y_train, x_val, y_val, hidden_layer_sizes, learning_rates, alphas, solvers,
                  activation_functions=["relu"]):
    """
    Compute the best MLP by training
    Record the training histories to a table
    :param x_train:
    :param y_train:
    :param x_val:
    :param y_val:
    :param hidden_layer_sizes:
    :param learning_rates:
    :param alphas:
    :param solvers:
    :param activation_functions:
    :return:
    """
    results = []
    for size in hidden_layer_sizes:
        for lr in learning_rates:
            for alpha in alphas:
                for solver in solvers:
                    for activation_function in activation_functions:
                        print(f"\n\nOn the setting size={size}, lr={lr}, alpha={alpha}, solver={solver}, act_func={activation_function}")
                        model = MLPClassifier(hidden_layer_sizes=size, activation=activation_function,
                                              solver=solver, alpha=alpha, learning_rate_init=lr, max_iter=100000,
                                              random_state=42)
                        model.fit(x_train, y_train)
                        predictions = model.predict(x_val)
                        val_accuracy = metrics.accuracy_score(y_val, predictions) * 100
                        row = {"Hidden Layer Size": size,
                               "Activation Function": activation_function,
                               "Solver": solver,
                               "Alpha": alpha,
                               "Learning Rate": lr,
                               "Validation Accuracy (%)": val_accuracy}
                        results.append(row)
                        print(row)

    results_sorted = sorted(
        results, key=lambda x: x['Validation Accuracy (%)'], reverse=True)
    export_excel(results_sorted, is_svm=False)
    print('MLP data is written to an Excel file successfully.')
    best_MLP_params = results_sorted[0]
    size_best = best_MLP_params['Hidden Layer Size']
    lr_best = best_MLP_params['Learning Rate']
    alpha_best = best_MLP_params['Alpha']
    solver_best = best_MLP_params['Solver']
    activation_function_best = best_MLP_params['Activation Function']
    best_val_acc = best_MLP_params['Validation Accuracy (%)']
    return size_best, lr_best, alpha_best, solver_best, activation_function_best, best_val_acc
