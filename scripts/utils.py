from matplotlib import pyplot as plt
from scikitplot.metrics import plot_confusion_matrix


def confusion_matrix(y_true, y_pred):
    plot_confusion_matrix(y_true, 
                        y_pred,
                        figsize=(12,12),
                        normalize=True,
                        text_fontsize=14)
    plt.savefig('confusion_matrix.png')
    plt.close()
    

def visualize_history(history):
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'train loss'], loc='upper left')
    plt.savefig('accuracy_history.png')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_history.png')
    plt.close()