"""Knowledge Graph dataset pre-processing functions."""

import collections
import os
import pickle

from collections import defaultdict
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
    entity_counts = defaultdict(int)
    relation_counts = defaultdict(int)
    for split in ["train", "valid", "test"]:
        with open(os.path.join(path, split), "r") as lines:
            for line in lines:
                lhs, rel, rhs = line.strip().split("\t")
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)

                if split == "train":
                    entity_counts[lhs] += 1
                    entity_counts[rhs] += 1
                    relation_counts[rel] += 1
                  
    ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}
    rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}

    for split in ["train"]:
        if split == "train":
            ent2idx_counts = {ent2idx[ent]: count for ent, count in entity_counts.items()}
            rel2idx_counts = {rel2idx[rel]: count for rel, count in relation_counts.items()}

            with open(os.path.join(path, split + '_ent2idx_counts.json'), 'w') as f:
                json.dump(ent2idx_counts, f)

            with open(os.path.join(path, split + '_rel2idx_counts.json'), 'w') as f:
                json.dump(rel2idx_counts, f)             

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

# Added by Xu: Categorize the triples in the training set into 3 groups.
def examples_sampling_score(triples, ent2idx_counts, rel2idx_counts):
    ranked_examples_all = []
    for triple in triples:
        lhs, rel, rhs = map(str, triple)
        sampling_score = np.log(ent2idx_counts[lhs] * rel2idx_counts[rel] * ent2idx_counts[rhs])
        ranked_examples_all.append([int(lhs), int(rel), int(rhs), sampling_score])
    
    ranked_examples_all = sorted(ranked_examples_all, key=lambda x: x[3], reverse=True)
    examples_score = [example[-1] for example in ranked_examples_all]
    ranked_examples = [example[:-1] for example in ranked_examples_all]
    
    # Divide equally based on quantity
    size = len(ranked_examples) // 3
    train_1 = ranked_examples[:size]
    train_2 = ranked_examples[size:size*2]
    train_3 = ranked_examples[size*2:]

    return np.array(train_1).astype("int64"), np.array(train_2).astype("int64"), np.array(train_3).astype("int64"), examples_score, ranked_examples_all
        

def get_filters(examples, n_relations):
    """Create filtering lists for evaluation.

    Args:
      examples: Numpy array of size n_examples x 3 containing KG triples
      n_relations: Int indicating the total number of relations in the KG

    Returns:
      lhs_final: Dictionary mapping queries (entity, relation) to filtered entities for left-hand-side prediction
      rhs_final: Dictionary mapping queries (entity, relation) to filtered entities for right-hand-side prediction
    """
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)
    for lhs, rel, rhs in examples:
        rhs_filters[(lhs, rel)].add(rhs)
        lhs_filters[(rhs, rel + n_relations)].add(lhs)
    lhs_final = {}
    rhs_final = {}
    for k, v in lhs_filters.items():
        lhs_final[k] = sorted(list(v))
    for k, v in rhs_filters.items():
        rhs_final[k] = sorted(list(v))
    return lhs_final, rhs_final


def process_dataset(path):
    """Map entities and relations to ids and saves corresponding pickle arrays.

    Args:
      path: Path to dataset directory

    Returns:
      examples: Dictionary mapping splits to with Numpy array containing corresponding KG triples.
      filters: Dictionary containing filters for lhs and rhs predictions.
    """
    ent2idx, rel2idx = get_idx(dataset_path)
    examples = {}
    splits = ["train", "valid", "test"]
    for split in splits:
        dataset_file = os.path.join(path, split)
        examples[split] = to_np_array(dataset_file, ent2idx, rel2idx)

        if split == "train":
            train_examples = examples[split]

            with open(os.path.join(path, split + '_ent2idx_counts.json'), 'r') as f:
                ent2idx_counts = json.load(f)
            with open(os.path.join(path, split + '_rel2idx_counts.json'), 'r') as f:
                rel2idx_counts = json.load(f)

            train_1, train_2, train_3, examples_score, ranked_examples_all = examples_sampling_score(train_examples, ent2idx_counts, rel2idx_counts)

    all_examples = np.concatenate([examples[split] for split in splits], axis=0)
    lhs_skip, rhs_skip = get_filters(all_examples, len(rel2idx))
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}

    return examples, filters, train_1, train_2, train_3, examples_score, ranked_examples_all


if __name__ == "__main__":
    data_path = os.environ["DATA_PATH"]
    for dataset_name in os.listdir(data_path):
        if dataset_name == 'FB237_curr':
            dataset_path = os.path.join(data_path, dataset_name)
            dataset_examples, dataset_filters, train_1, train_2, train_3, examples_score, ranked_examples_all= process_dataset(dataset_path)
            for dataset_split in ["train", "valid", "test"]:
                save_path = os.path.join(dataset_path, dataset_split + ".pickle")
                with open(save_path, "wb") as save_file:
                    pickle.dump(dataset_examples[dataset_split], save_file)
            with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
                pickle.dump(dataset_filters, save_file)
                    
            with open(os.path.join(dataset_path, "train_1.pickle"), "wb") as save_file:
                pickle.dump(train_1, save_file)
            
            with open(os.path.join(dataset_path, "train_2.pickle"), "wb") as save_file:
                pickle.dump(train_2, save_file)
            
            with open(os.path.join(dataset_path, "train_3.pickle"), "wb") as save_file:
                pickle.dump(train_3, save_file)
                
            with open(os.path.join(dataset_path, "ranked_examples_score"), "w") as save_file:
                for score in examples_score:
                    save_file.write(str(score) + '\n')
            
            with open(os.path.join(dataset_path, "ranked_examples_all"), "w") as save_file:
                for line in ranked_examples_all:
                    save_file.write(str(line) + '\n')
            
