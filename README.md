This code is an extension of the original work https://github.com/HazyResearch/KGEmb,  
my job focuses on the data processing and the adjustments to the training process.  
Using wandb to record the training curve.

run.py contains two functiong: train() and train1().   
train() denotes ramdom sampling, serving as the baseline, while train1() denotes biased sampling.  
When train1 with "ranked_examples.pickle" and "probabilities_tensor.pt", refers to descending sampling;  
with "ascending_ranked_examples.pickle" and "ascending_probabilities_tensor.pt", refers to ascending sampling.

run1.py is an implementation of the curriculum learning method.  
run2.py is an implementation of the adaptive learning-based training method.  

process.py generate descending sampling probability.  
process_ascending.py generate ascending sampling probability.  
process_curr.py categorizes the training triples into 3 groups.  

