"""Knowledge Graph dataset pre-processing functions."""

import os
import pickle
import json

import numpy as np

os.environ["DATA_PATH"] = "/mnt/c/Users/96551/Documents/Master Arbeit/KGEmb-master/data"

def get_idx(path):
    """Map entities and relations to unique ids.

    Args:
      path: path to directory with raw dataset files (tab-separated train/valid/test triples)

    Returns:
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids
    """
    entities, relations = set(), set()
    for split in ["train", "valid", "test"]:
        with open(os.path.join(path, split), "r") as lines:
            for line in lines:
                lhs, rel, rhs = line.strip().split("\t")
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)
                  
    ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}
    rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}

    return ent2idx, rel2idx

def to_np_array(dataset_file, ent2idx, rel2idx):
    """Map raw dataset file to numpy array with unique ids.

    Args:
      dataset_file: Path to file containing raw triples in a split
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids

    Returns:
      Numpy array of size n_examples x 3 mapping the raw dataset file to ids
    """
    examples = []
    with open(dataset_file, "r") as lines:
        for line in lines:
            lhs, rel, rhs = line.strip().split("\t")
            try:
                examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs]])
                
            except ValueError:
                continue

    return np.array(examples).astype("int64")

# Added by Xu: Divide test triples into 3 groups
def test_examples_sampling_score(triples, entity_counts, relation_counts):
    test_ranked_examples = []
    for triple in triples:
        lhs, rel, rhs = map(str, triple)
        lhs_count = entity_counts.get(lhs, 1)
        rel_count = relation_counts.get(rel, 1)
        rhs_count = entity_counts.get(rhs, 1)
        test_sampling_score = np.log(lhs_count * rel_count * rhs_count)
        test_ranked_examples.append([int(lhs), int(rel), int(rhs), test_sampling_score])

    test_ranked_examples = sorted(test_ranked_examples, key=lambda x: x[3], reverse=True)
    test_ranked_examples_score = [example[-1] for example in test_ranked_examples]
    test_ranked_examples = [example[:-1] for example in test_ranked_examples]

    max_score = max(test_ranked_examples_score)
    min_score = min(test_ranked_examples_score)
    
    range_ = max_score - min_score
    threshold1 = min_score + range_ / 3
    threshold2 = min_score + 2 * range_ / 3
    
    hard_examples = []
    medium_examples = []
    easy_examples = []
    
    for example, score in zip(test_ranked_examples, test_ranked_examples_score):
        if score <= threshold1:
            hard_examples.append(example)
        elif score <= threshold2:
             medium_examples.append(example)
        else:
            easy_examples.append(example)

    return np.array(easy_examples).astype("int64"), np.array(medium_examples).astype("int64"), np.array(hard_examples).astype("int64"), test_ranked_examples_score

def process_dataset(path):
    ent2idx, rel2idx = get_idx(dataset_path)
    examples = {}
    splits = ["train", "valid", "test"]
    for split in splits:
        dataset_file = os.path.join(path, split)

        if split == "test":
            examples[split] = to_np_array(dataset_file, ent2idx, rel2idx)
            test_examples = examples[split]
            with open(os.path.join(path, 'train_ent2idx_counts.json'), 'r') as f:
                entity_counts = json.load(f)
            with open(os.path.join(path, 'train_rel2idx_counts.json'), 'r') as f:
                relation_counts = json.load(f)
            easy_examples, medium_examples, hard_examples, test_ranked_examples_score = test_examples_sampling_score(test_examples, entity_counts, relation_counts)

    return easy_examples, medium_examples, hard_examples, test_ranked_examples_score


if __name__ == "__main__":
    data_path = os.environ["DATA_PATH"]
    for dataset_name in os.listdir(data_path):
        if dataset_name == "FB237_64":
            dataset_path = os.path.join(data_path, dataset_name)
            easy_examples, medium_examples, hard_examples, test_ranked_examples_score = process_dataset(dataset_path)
                
            with open(os.path.join(dataset_path, "test_easy.pickle"), "wb") as save_file:
                pickle.dump(easy_examples, save_file)
            
            with open(os.path.join(dataset_path, "test_medium.pickle"), "wb") as save_file:
                pickle.dump(medium_examples, save_file)

            with open(os.path.join(dataset_path, "test_hard.pickle"), "wb") as save_file:
                pickle.dump(hard_examples, save_file)

            with open(os.path.join(dataset_path, "test_ranked_examples_score"), "w") as save_file:
                for score in test_ranked_examples_score:
                    save_file.write(str(score) + '\n')