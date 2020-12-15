import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


def generatePlots(scorevalues, evaluationMetric, runtime):
    algo_names = ['Decision Tree', 'Random Forest', 'KNN', 'SVM', 'Naive Byes', 'XGBoost']
    plt.xlabel('Algorithms')
    plt.ylabel(evaluationMetric + ' Score')
    plt.title("Overall " +runtime+ " " + evaluationMetric + " Score")
    plt.bar(algo_names, scorevalues, color=['green', 'blue', 'red', 'grey', 'maroon','purple'])
    plt.show()


def generateConfusionMatrix(cm,accuracy,evaluationMetric):
    column_name = ["Likely Democratic", "Likely Republican", "Likely Independent"]
    plot_confusion_matrix(cm, cmap=plt.cm.BuPu, class_names=column_name)
    plt.title(label='Confusion Matrix for '+evaluationMetric, loc='center', pad=None)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, 1 - accuracy))
    plt.show()