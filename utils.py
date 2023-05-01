from matplotlib import pyplot as plt


def format_dataset(sample):
    question = sample['question']['text']
    context = sample['document']['tokens']['token']
    is_html = sample['document']['tokens']['is_html']
    long_answers = sample['annotations']['long_answer']
    short_answers = sample['annotations']['short_answers']

    context_string = " ".join([context[i] for i in range(len(context)) if not is_html[i]])

    for answer in sample['annotations']['yes_no_answer']:
        if answer == 0 or answer == 1:
            return {"question": question, "context": context_string, "short": [], "long": [],
                    "category": "no" if answer == 0 else "yes"}

    short_targets = []
    for s in short_answers:
        short_targets.extend(s['text'])
    short_targets = list(set(short_targets))

    long_targets = []
    for s in long_answers:
        if s['start_token'] == -1:
            continue
        answer = context[s['start_token']: s['end_token']]
        html = is_html[s['start_token']: s['end_token']]
        new_answer = " ".join([answer[i] for i in range(len(answer)) if not html[i]])
        if new_answer not in long_targets:
            long_targets.append(new_answer)

    category = "long_short" if len(short_targets + long_targets) > 0 else "null"

    return {"question": question, "context": context_string, "short": short_targets, "long": long_targets,
            "category": category}


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


