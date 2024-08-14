"""Train Knowledge Graph embeddings for link prediction."""

import argparse
import json
import logging
import os

import torch
import torch.optim

import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params

import wandb

torch.cuda.empty_cache()

"""import matplotlib.pyplot as plt"""

os.environ["LOG_DIR"] = "/workspace/KGEmb-master/logs"

# 创建 ArgumentParser 对象，用于解析命令行参数
parser = argparse.ArgumentParser(
    description="Knowledge Graph Embedding"
)

parser.add_argument(
    "--dataset", default="FB237_base", choices=["WN18RR_al", "FB237_al", "YAGO3-10_al"],
    help="Knowledge Graph dataset")

parser.add_argument(
    "--model", default="ComplEx", choices=["TransE","RotatE","ComplEx"], help="Knowledge Graph embedding model"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adagrad",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=300, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=5, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--rank", default=500, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--batch_size", default=1000, type=int, help="Batch size"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--learning_rate", default=0.1, type=float, help="Learning rate"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)

def train(args):
    save_dir = get_savedir(args.model, args.dataset)

    wandb.init(project='11', entity='xuleqi')

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log")
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))

    # create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    # load data
    logging.info("\t " + str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    al_train_examples = dataset.get_examples("train_al")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    model = getattr(models, args.model)(args)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    device = "cuda"
    model.to(device)

    # get optimizer
    regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
                            bool(args.double_neg))
    counter = 0
    best_mrr = None
    best_epoch = None
    logging.info("\t Start training")

    for step in range(args.max_epochs):

        # Train step
        model.train()
        train_loss = optimizer.epoch(train_examples)
        wandb.log({"train_loss": train_loss}, step=step)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

        # Valid step
        model.eval()
        valid_loss = optimizer.calculate_valid_loss(valid_examples)
        wandb.log({"valid_loss": valid_loss}, step=step)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        if (step + 1) % args.valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
            logging.info(format_metrics(valid_metrics, split="valid"))

            valid_mrr = valid_metrics["MRR"]
            wandb.log({"valid_mrr": valid_mrr}, step=step)  
            valid_mr = valid_metrics["MR"]
            wandb.log({"valid_mr": valid_mr}, step=step)  

            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                counter = 0
                best_epoch = step
                logging.info("\t Saving model at epoch {} in {}".format(step, save_dir))
                torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                model.to(device)
            else:
                counter += 1
                if counter == args.patience:
                    logging.info("\t Early stopping")

                    # Added by Xu: Evaluate trainind triples' performance during the first phase
                    train_reciprocal_ranks = model.compute_train_ranks(al_train_examples, filters)
                    ranks_file_path = os.path.join(save_dir, 'train_reciprocal_ranks.pt')  
                    torch.save(train_reciprocal_ranks, ranks_file_path)
                    train_examples, probabilities = dataset.generate_probability_files(train_reciprocal_ranks, al_train_examples)

                    counter = 0
                    logging.info("\t Re-training starting")
                    logging.info("\t Reducing learning rate")
                    
                    for retraining_step in range(step + 1, args.max_epochs):
                        
                        model.train()
                        train_loss = optimizer.epoch1(train_examples, probabilities)
                        wandb.log({"train_loss": train_loss}, step=retraining_step)
                        logging.info("\t Epoch {} | average train loss: {:.4f}".format(retraining_step, train_loss))

                        model.eval()
                        valid_loss = optimizer.calculate_valid_loss(valid_examples)
                        wandb.log({"valid_loss": valid_loss}, step=retraining_step)
                        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(retraining_step, valid_loss))

                        if (retraining_step + 1) % args.valid == 0:
                            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
                            logging.info(format_metrics(valid_metrics, split="valid"))

                            valid_mrr = valid_metrics["MRR"]
                            wandb.log({"valid_mrr": valid_mrr}, step=retraining_step)  
                            valid_mr = valid_metrics["MR"]
                            wandb.log({"valid_mr": valid_mr}, step=retraining_step)  

                            if not best_mrr or valid_mrr > best_mrr:
                                best_mrr = valid_mrr
                                counter = 0
                                best_epoch = retraining_step
                                logging.info("\t Saving model at epoch {} in {}".format(retraining_step, save_dir))
                                torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                                model.to(device)
                            
                            else:
                                counter += 1
                        
                                if counter == args.patience + 10:
                                    logging.info("\t Re-training early stopping ")
                                    break
                    break
                    
                elif counter == args.patience // 2:
                    pass 
                    # logging.info("\t Reducing learning rate")
                    # optimizer.reduce_lr()

    logging.info("\t Optimization finished")
    if not best_mrr:
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    model.to(device)
    model.eval()

    train_reciprocal_ranks = model.compute_train_ranks(al_train_examples, filters)
    ranks_file_path = os.path.join(save_dir, 'train_reciprocal_ranks.pt')  
    torch.save(train_reciprocal_ranks, ranks_file_path)

    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters)) 
    logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    logging.info(format_metrics(test_metrics, split="test"))
    wandb.log({"test": test_metrics})

    wandb.define_metric("MR", step_metric="test_name")
    wandb.define_metric("MRR", step_metric="test_name")

    data1 = [] #MR
    data2 = [] #MRR
    test_names = ["test_easy", "test_medium", "test_hard"]
    test_datasets = ["test_easy", "test_medium", "test_hard"]
    
    for i in range(3):
        test_name = test_names[i]
        test_examples = dataset.get_examples(test_datasets[i])
        
        test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
        logging.info(format_metrics(test_metrics, split=test_name))
        data1.append([test_name, test_metrics["MR"]])
        data2.append([test_name, test_metrics["MRR"]])

    table = wandb.Table(data=data1, columns=["test_names", "MR"])
    fields1 = {"x": "test_names", "value": "MR"}
    my_custom_chart1 = wandb.plot_table(vega_spec_name="xuleqi/new_chart2", data_table=table, fields=fields1)
    wandb.log({"MR of test": my_custom_chart1})

    table = wandb.Table(data=data2, columns=["test_names", "MRR"])
    fields2 = {"x": "test_names", "value": "MRR"}
    my_custom_chart2 = wandb.plot_table(vega_spec_name="xuleqi/new_chart2", data_table=table, fields=fields2)
    wandb.log({"MRR of test": my_custom_chart2})
 
if __name__ == "__main__":
    train(parser.parse_args())

