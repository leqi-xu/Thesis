"""Dataset class for loading and processing KG datasets."""

import os
import pickle as pkl

import numpy as np
import torch

from collections import defaultdict


class KGDataset(object):
    """Knowledge Graph dataset class."""

    def __init__(self, data_path, debug):
        """Creates KG dataset object for data loading.

        Args:
             data_path: Path to directory containing train/valid/test pickle files produced by process.py
             debug: boolean indicating whether to use debug mode or not
             if true, the dataset will only contain 1000 examples for debugging.
        """
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        for split in ["train", "test", "valid", "test_easy", "test_medium", "test_hard"]: # Changed by Xu
            file_path = os.path.join(self.data_path, split + ".pickle")
            with open(file_path, "rb") as in_file:
                self.data[split] = pkl.load(in_file)
        
        filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")
        self.to_skip = pkl.load(filters_file)
        filters_file.close()

        # Added by Xu: Open ranked_examples.pickle 
        ranked_examples_file = os.path.join(self.data_path, "ranked_examples.pickle")
        with open(ranked_examples_file, "rb") as ranked_file:
            self.ranked_examples_array = pkl.load(ranked_file)
        
        # Added by Xu: Open probabilities.pt
        probabilities_file = os.path.join(self.data_path, "probabilities_tensor.pt")
        self.probabilities = torch.load(probabilities_file)
        

        max_axis = np.max(self.data["train"], axis=0)
        self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
        self.n_predicates = int(max_axis[1] + 1) * 2

    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data[split]
        
        if split == "train":
            copy = np.copy(examples)
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2
            examples = np.vstack((examples, copy))
        
        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]
        return torch.from_numpy(examples.astype("int64"))
    
    def get_filters(self, ):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip
    
    def get_ranked_examples(self, ):
        return self.ranked_examples_array

    def get_probabilities(self, ):
        return self.probabilities

    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities

    # Added by Xu: Generate the sampling probability based on the reciprocal_ranking of training triples.
    def generate_probability_files(self, train_reciprocal_ranks, al_train_examples):
        data1 = train_reciprocal_ranks.to("cpu")
        data1 = np.array(data1)
        al_train_examples = al_train_examples.numpy()

        entity_sum = defaultdict(int)
        entity_count = defaultdict(int)
        relation_sum = defaultdict(int)
        relation_count = defaultdict(int)

        i = 0
        for triple in al_train_examples:
            lhs, rel ,rhs = triple
            entity_sum[lhs] += data1[i]
            entity_count[lhs] += 1
            entity_sum[rhs] += data1[i]
            entity_count[rhs] += 1
            relation_sum[rel] += data1[i]
            relation_count[rel] += 1
            i += 1

        entity = {k: entity_sum[k] / entity_count[k] for k in entity_sum}
        relation = {k: relation_sum[k] / relation_count[k] for k in relation_sum}

        examples = []
        for triple in al_train_examples:
            lhs, rel, rhs = triple
            sampling_score = entity[lhs] * relation[rel] * entity[rhs]
            examples.append([lhs, rel, rhs, sampling_score])

        # Adding inverse relationships
        symmetrical_examples = []
        for example in examples:
            lhs, rel, rhs, score = example
            symmetrical_score = score
            symmetrical_rel = rel + len(relation) 
            symmetrical_examples.append([rhs, symmetrical_rel, lhs, symmetrical_score])
            
        examples.extend(symmetrical_examples)
        examples_score = [example[-1] for example in examples]
        min_score = min(examples_score)
        max_score = max(examples_score)
        examples_score_normalized = (examples_score - min_score) / (max_score - min_score)
        examples_score_normalized = 1 - examples_score_normalized

        with open(os.path.join(self.data_path, "al_train_examples_score"), "w") as save_file:
            for score in examples_score_normalized:
                save_file.write(str(score) + '\n')

        train_examples = [example[:-1] for example in examples]
        train_examples = torch.tensor(train_examples)
        probabilities = torch.tensor(examples_score_normalized)

        return train_examples, probabilities