import matplotlib.pyplot as plt
from src.utils import load_pickle

def plot_loss(losses, docs):
    final_losses= []
    final_docs = []
    print(len(losses), len(docs))

    for index in range(len(losses)-1):
        if docs[index] != 0:
            loss = losses[index]/docs[index]
            final_losses.append(loss)
            final_docs.append(docs[index])
        else:
            break
    print("Minimum loss: ", sorted(final_losses[:10]))
    plt.plot(final_docs, final_losses)
    plt.show()

if __name__ == '__main__':
    path_classifier = '../models/Partial/Classifier/'
    path_gnn = '../models/Partial/Gnn/'

    # loss_class = load_pickle(path_classifier+'losses_classifier')
    # docs_class = load_pickle(path_classifier+'docs_classifier')
    # # print(sorted(loss_class))
    # plot_loss(loss_class, docs_class)

    loss_class = load_pickle(path_gnn+'losses_classifier')
    docs_class = load_pickle(path_gnn+'docs_classifier')
    plot_loss(loss_class, docs_class)
