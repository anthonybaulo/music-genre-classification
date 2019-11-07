import matplotlib.pyplot as plt



# TODO: Create function to save model plots
# Consider directory and informative filename
def show_summary_stats(history, savepath=None, dpi=200):
    # List all data in history
    print(history.history.keys())

    # Summarize history for accuracy
    plt.figure(figsize=(5,3))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    if savepath:
        plt.savefig(os.path.join(savepath, f''), 
                    dpi=dpi)
    plt.show()

    # Summarize history for loss
    plt.figure(figsize=(5,3))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    if savepath:
        plt.savefig(savepath, dpi=dpi)
    plt.show()