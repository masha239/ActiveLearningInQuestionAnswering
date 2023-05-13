from matplotlib import pyplot as plt


def visualize_dataset(dataset):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    axs[0, 0].hist([len(s['question']) for s in dataset], bins=20)
    axs[0, 0].set_xlabel('Length of question')
    axs[0, 0].set_ylabel('Number of samples')

    axs[0, 1].hist([len(s['context']) for s in dataset], bins=20)
    axs[0, 1].set_xlabel('Length of context')
    axs[0, 1].set_ylabel('Number of samples')

    axs[1, 0].hist([s['category'] for s in dataset])
    axs[1, 0].set_xlabel('Category')
    axs[1, 0].set_ylabel('Number of samples')

    axs[1, 1].hist([len(s['long']) for s in dataset])
    axs[1, 1].set_xlabel('Number of long candidates')
    axs[1, 1].set_ylabel('Number of samples')

    fig.suptitle('Dataset visualization')
    plt.show()
