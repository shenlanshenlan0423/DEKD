# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/12 10:43
@Auth ： Hongwei
@File ：ExperimentManager.py
@IDE ：PyCharm
"""
from definitions import *
from src.evaluate.evaluate import model_evaluate
from src.utils.main_uitls import train_model


class ExperimentManager:
    def __init__(self, dataset_path, model_path, results_path, args):
        self.dataset = load_pickle(dataset_path)
        self.trained_model_path = model_path
        self.results_path = results_path
        self.args = args

    def print_results(self, evaluation_results):
        model_names = list(evaluation_results[0].keys())
        metrics = ['AUROC', 'AUPRC']
        result_dataframe = pd.DataFrame(columns=['Model'] + metrics)
        result_dataframe['Model'] = model_names
        for i, model_name in enumerate(model_names):
            for j, metric in enumerate(metrics):
                metric_values = [evaluation_results[idx_][model_name][metric] for idx_ in
                                 range(len(evaluation_results))]
                result_dataframe.iloc[i, j + 1] = '{:.4f}±{:.4f}'.format(np.mean(metric_values), np.std(metric_values))
        print(result_dataframe.to_markdown(index=False))

    def run(self, load_model):
        print('\n----------------------------My model Run Begin----------------------------\n')
        my_models, evaluation_results = {}, []
        for idx, fold_idx in enumerate(tqdm(list(self.dataset.keys()))):
            fold_data = self.dataset[fold_idx]
            cols, train_set, val_set, test_set = fold_data['cols'], fold_data['train'], fold_data['val'], fold_data[
                'test']
            X_train, X_val, X_test = train_set[:, :-1], val_set[:, :-1], test_set[:, :-1]
            y_train, y_val, y_test = train_set[:, -1], val_set[:, -1], test_set[:, -1]
            if not load_model:
                delak = train_model(model_name='DELAK', X_train=X_train, y_train=y_train, args=self.args)
                dekd_1 = train_model(model_name='DEKD (Stage 1)', X_train=X_train, y_train=y_train, cols=cols,
                                     args=self.args)
                dekd = train_model(model_name='DEKD', X_train=X_train, y_train=y_train, X_val=X_val,
                                   y_val=y_val, cols=cols, args=self.args, idx=idx + 1)
                my_models[fold_idx] = {
                    'DELAK': delak,
                    'DEKD (Stage 1)': dekd_1,
                    'DEKD': dekd,
                }
            else:
                trained_model_dict = load_pickle(self.trained_model_path)[fold_idx]
                delak = trained_model_dict['DELAK']
                dekd_1 = trained_model_dict['DEKD (Stage 1)']
                dekd = trained_model_dict['DEKD']
            evaluation_result = {
                'DELAK': model_evaluate(model=delak, X_test=X_test, y_test=y_test),
                'DEKD (Stage 1)': model_evaluate(model=dekd_1, X_test=X_test, y_test=y_test),
                'DEKD': model_evaluate(model=dekd, X_test=X_test, y_test=y_test),
            }
            evaluation_results.append(evaluation_result)

        self.print_results(evaluation_results)
        if not load_model:
            save_pickle(my_models, self.trained_model_path)
        return evaluation_results


class Experiment_part1(ExperimentManager):
    def __init__(self, dataset_path, model_path, results_path, args):
        super().__init__(dataset_path, model_path, results_path, args)

    def run(self, load_model):
        print('\n----------------------------Part 1 Run Begin----------------------------\n')
        trained_models, evaluation_results = {}, []
        for idx, fold_idx in enumerate(tqdm(list(self.dataset.keys()))):
            fold_data = self.dataset[fold_idx]
            cols, train_set, val_set, test_set = fold_data['cols'], fold_data['train'], fold_data['val'], fold_data[
                'test']

            X_train, X_val, X_test = train_set[:, :-1], val_set[:, :-1], test_set[:, :-1]
            y_train, y_val, y_test = train_set[:, -1], val_set[:, -1], test_set[:, -1]
            if not load_model:
                # Traditional ML model
                lr = train_model(model_name='LR', X_train=X_train, y_train=y_train)
                dt = train_model(model_name='DT', X_train=X_train, y_train=y_train)
                bg = train_model(model_name='BG', X_train=X_train, y_train=y_train)
                rf = train_model(model_name='RF', X_train=X_train, y_train=y_train)
                ada = train_model(model_name='ADA', X_train=X_train, y_train=y_train)
                xgb = train_model(model_name='XGB', X_train=X_train, y_train=y_train)
                mlp = train_model(model_name='MLP', X_train=X_train, y_train=y_train)
                trained_model_dict = {
                    'LR': lr,
                    'DT': dt,
                    'BG': bg,
                    'RF': rf,
                    'ADA': ada,
                    'XGB': xgb,
                    'MLP': mlp,
                }
                trained_models[fold_idx] = trained_model_dict
            else:
                trained_model_dict = load_pickle(self.trained_model_path)[fold_idx]
                lr = trained_model_dict['LR']
                dt = trained_model_dict['DT']
                bg = trained_model_dict['BG']
                rf = trained_model_dict['RF']
                ada = trained_model_dict['ADA']
                xgb = trained_model_dict['XGB']
                mlp = trained_model_dict['MLP']
            evaluation_result = {
                'LR': model_evaluate(model=lr, X_test=X_test, y_test=y_test),
                'DT': model_evaluate(model=dt, X_test=X_test, y_test=y_test),
                'BG': model_evaluate(model=bg, X_test=X_test, y_test=y_test),
                'RF': model_evaluate(model=rf, X_test=X_test, y_test=y_test),
                'ADA': model_evaluate(model=ada, X_test=X_test, y_test=y_test),
                'XGB': model_evaluate(model=xgb, X_test=X_test, y_test=y_test),
                'MLP': model_evaluate(model=mlp, X_test=X_test, y_test=y_test),
            }
            evaluation_results.append(evaluation_result)

        self.print_results(evaluation_results)
        if not load_model:
            save_pickle(trained_models, self.trained_model_path)
        return evaluation_results


class Experiment_part2(ExperimentManager):
    def __init__(self, dataset_path, model_path, results_path, args):
        super().__init__(dataset_path, model_path, results_path, args)

    def run(self, load_model):
        print('\n----------------------------Part 2 Run Begin----------------------------\n')
        trained_models, evaluation_results = {}, []
        for idx, fold_idx in enumerate(tqdm(list(self.dataset.keys()))):
            fold_data = self.dataset[fold_idx]
            cols, train_set, val_set, test_set = fold_data['cols'], fold_data['train'], fold_data['val'], fold_data[
                'test']
            self.args.cols = cols
            X_train, X_val, X_test = train_set[:, :-1], val_set[:, :-1], test_set[:, :-1]
            y_train, y_val, y_test = train_set[:, -1], val_set[:, -1], test_set[:, -1]
            if not load_model:
                # Traditional static ensemble model
                avge = train_model(model_name='AvgE', X_train=X_train, y_train=y_train, args=self.args)
                maxe = train_model(model_name='MaxE', X_train=X_train, y_train=y_train, args=self.args)
                mine = train_model(model_name='MinE', X_train=X_train, y_train=y_train, args=self.args)
                wauce = train_model(model_name='WAUCE', X_train=X_train, y_train=y_train, args=self.args)
                wauce.predict_val(X_val=X_val, y_val=y_val)  # get AUC of each base classifier for weight calculate
                sb = train_model(model_name='SB', X_train=X_train, y_train=y_train, args=self.args)
                sb.predict_val(X_val=X_val, y_val=y_val)
                trained_model_dict = {
                    'AvgE': avge,
                    'MaxE': maxe,
                    'MinE': mine,
                    'WAUCE': wauce,
                    'SB': sb,
                }
                trained_models[fold_idx] = trained_model_dict
            else:
                trained_model_dict = load_pickle(self.trained_model_path)[fold_idx]
                avge = trained_model_dict['AvgE']
                maxe = trained_model_dict['MaxE']
                mine = trained_model_dict['MinE']
                wauce = trained_model_dict['WAUCE']
                sb = trained_model_dict['SB']
            evaluation_result = {
                'AvgE': model_evaluate(model=avge, X_test=X_test, y_test=y_test),
                'MaxE': model_evaluate(model=maxe, X_test=X_test, y_test=y_test),
                'MinE': model_evaluate(model=mine, X_test=X_test, y_test=y_test),
                'WAUCE': model_evaluate(model=wauce, X_test=X_test, y_test=y_test),
                'SB': model_evaluate(model=sb, X_test=X_test, y_test=y_test),
            }
            evaluation_results.append(evaluation_result)

        self.print_results(evaluation_results)
        if not load_model:
            save_pickle(trained_models, self.trained_model_path)
        return evaluation_results


class Experiment_part3(ExperimentManager):
    def __init__(self, dataset_path, model_path, results_path, args):
        super().__init__(dataset_path, model_path, results_path, args)

    def run(self, load_model):
        print('\n----------------------------Part 3 Run Begin----------------------------\n')
        trained_models, evaluation_results = {}, []
        for idx, fold_idx in enumerate(tqdm(list(self.dataset.keys()))):
            fold_data = self.dataset[fold_idx]
            cols, train_set, val_set, test_set = fold_data['cols'], fold_data['train'], fold_data['val'], fold_data[
                'test']

            X_train, X_val, X_test = train_set[:, :-1], val_set[:, :-1], test_set[:, :-1]
            y_train, y_val, y_test = train_set[:, -1], val_set[:, -1], test_set[:, -1]
            if not load_model:
                # The state-of-the-art techniques for dynamic classifier
                desp = train_model(model_name='DESP', X_train=X_train, y_train=y_train)
                knorau = train_model(model_name='KNORAU', X_train=X_train, y_train=y_train)
                kne = train_model(model_name='KNORAE', X_train=X_train, y_train=y_train)
                meta = train_model(model_name='METADES', X_train=X_train, y_train=y_train)
                trained_model_dict = {
                    'DESP': desp,
                    'KNORAU': knorau,
                    'KNORAE': kne,
                    'METADES': meta,
                }
                trained_models[fold_idx] = trained_model_dict
            else:
                trained_model_dict = load_pickle(self.trained_model_path)[fold_idx]
                desp = trained_model_dict['DESP']
                knorau = trained_model_dict['KNORAU']
                kne = trained_model_dict['KNORAE']
                meta = trained_model_dict['METADES']
            evaluation_result = {
                'DESP': model_evaluate(model=desp, X_test=X_test, y_test=y_test),
                'KNORAU': model_evaluate(model=knorau, X_test=X_test, y_test=y_test),
                'KNORAE': model_evaluate(model=kne, X_test=X_test, y_test=y_test),
                'METADES': model_evaluate(model=meta, X_test=X_test, y_test=y_test),
            }
            evaluation_results.append(evaluation_result)

        self.print_results(evaluation_results)
        if not load_model:
            save_pickle(trained_models, self.trained_model_path)
        return evaluation_results


def get_config(outcome_label, balance_ratio, n_clusters, beta, sample_ratio, Temperature, Alpha):
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Settings")
    parser.add_argument('--outcome_label', type=str, default=outcome_label)
    parser.add_argument('--balance_ratio', type=int, default=balance_ratio)
    parser.add_argument('--n_clusters', type=int, default=n_clusters)
    parser.add_argument('--sample_ratio', type=float, default=sample_ratio)
    parser.add_argument('--beta', type=float, default=beta)
    parser.add_argument('--num_boost_round', type=int, default=380)
    parser.add_argument('--Temperature', type=float, default=Temperature)
    parser.add_argument('--Alpha', type=float, default=Alpha)
    args = parser.parse_args()
    return args
