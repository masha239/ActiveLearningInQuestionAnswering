import random


def sample_jsonl_dataset(dataset_path, sampled_dataset_path, n_sample):
    with open(dataset_path, 'r') as json_file:
        json_list = list(json_file)
    sample = random.sample(json_list, n_sample)
    with open(sampled_dataset_path, 'w') as f:
        for item in sample:
            f.write(item)


if __name__ == '__main__':
    n_sample = 50000
    dataset_path = '/users/masha239/Downloads/v1.0-simplified_simplified-nq-train.jsonl'
    sampled_dataset_path = '/users/masha239/Downloads/data_sampled.jsonl'
    sample_jsonl_dataset(dataset_path, sampled_dataset_path, n_sample)
