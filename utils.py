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
