# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/12 10:36
@Auth ： Hongwei
@File ：main.py
@IDE ：PyCharm
"""
from definitions import *
from ExperimentManager import ExperimentManager, Experiment_part1, Experiment_part2, Experiment_part3, Experiment_part4, parse_args


def run_baseline(dataset_path, trained_model_path, results_path, args, load_model):
    trained_model = trained_model_path + '[BR-1-{}]_part1.pickle'.format(args.balance_ratio)
    experiment = Experiment_part1(dataset_path=dataset_path, model_path=trained_model,
                                  results_path=results_path, args=args)
    evaluation_results = experiment.run(load_model=load_model)
    save_pickle(evaluation_results, results_path + '/[BR-1-{}]_eval_res_part1.pickle'.format(args.balance_ratio))

    trained_model = trained_model_path + '[BR-1-{}]_part2.pickle'.format(args.balance_ratio)
    experiment = Experiment_part2(dataset_path=dataset_path, model_path=trained_model,
                                  results_path=results_path, args=args)
    evaluation_results = experiment.run(load_model=load_model)
    save_pickle(evaluation_results, results_path + '/[BR-1-{}]_eval_res_part2.pickle'.format(args.balance_ratio))

    trained_model = trained_model_path + '[BR-1-{}]_part3.pickle'.format(args.balance_ratio)
    experiment = Experiment_part3(dataset_path=dataset_path, model_path=trained_model,
                                  results_path=results_path, args=args)
    evaluation_results = experiment.run(load_model=load_model)
    save_pickle(evaluation_results, results_path + '/[BR-1-{}]_eval_res_part3.pickle'.format(args.balance_ratio))

    trained_model = trained_model_path + '[BR-1-{}]_part4.pickle'.format(args.balance_ratio)
    experiment = Experiment_part4(dataset_path=dataset_path, model_path=trained_model,
                                  results_path=results_path, args=args)
    evaluation_results = experiment.run(load_model=load_model)
    save_pickle(evaluation_results, results_path + '/[BR-1-{}]_eval_res_part4.pickle'.format(args.balance_ratio))


def run_myModels(dataset_path, trained_model_path, results_path, args, load_model):
    trained_model = trained_model_path + '[BR-1-{}]_myModel.pickle'.format(args.balance_ratio)
    experiment = ExperimentManager(dataset_path=dataset_path, model_path=trained_model,
                                   results_path=results_path, args=args)
    evaluation_results = experiment.run(load_model=load_model)
    save_pickle(evaluation_results, results_path + '/[BR-1-{}]_eval_res_myModel.pickle'.format(args.balance_ratio))


if __name__ == '__main__':
    for outcome_label in ['48h_mortality', 'hospital_mortality']:
        for balance_ratio in [1, 3, 5]:
            args = parse_args(outcome_label=outcome_label, balance_ratio=balance_ratio,
                              n_clusters=4, Alpha=9, Eta=0.95, Beta=0.5, Lambda=0.95)
            dataset_path = DATA_DIR + '/processed_data/{}/{}_folds_datasets_BR-[1-{}].pickle'.format(
                args.outcome_label,
                fold_number,
                args.balance_ratio)
            trained_model_path = MODELS_DIR + '/{}/'.format(args.outcome_label)
            os.makedirs(trained_model_path, exist_ok=True)
            results_path = RESULT_DIR + '/{}/'.format(args.outcome_label)
            os.makedirs(results_path, exist_ok=True)

            run_baseline(dataset_path, trained_model_path, results_path, args, load_model=True)
            run_myModels(dataset_path, trained_model_path, results_path, args, load_model=True)
