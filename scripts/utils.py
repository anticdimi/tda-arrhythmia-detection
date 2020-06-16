from matplotlib import pyplot as plt
from scikitplot.metrics import plot_confusion_matrix


def confusion_matrix(y_true, y_pred, accuracy):
    plot_confusion_matrix(y_true, 
                        y_pred,
                        figsize=(12,12),
                        normalize=True,
                        text_fontsize=20,
                        title=f'Accuracy: {accuracy:.2f}',
                        title_fontsize=20)
    plt.savefig('./history/confusion_matrix.png')
    plt.close()
    

def visualize_history(history):
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'train loss'], loc='upper left')
    plt.savefig('./history/accuracy_history.png')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./history/loss_history.png')
    plt.close()