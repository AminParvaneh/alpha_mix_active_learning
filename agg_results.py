import argparse
import copy
import csv
import json
import numpy as np
import os
import math
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

font_size = 14
font = {'family' : 'serif',
        'size'   : font_size}

matplotlib.rc('font', **font)

matrix_fint_size = 12

exp_colors = {
    'Random': 'tab:gray',
    'Entropy': 'tab:purple',
    'CoreSet': 'tab:green',
    'BALD': 'tab:brown',
    'BADGE': 'tab:red',
    'DFAL': 'tab:blue',
    'CDAL': 'tab:cyan',
    'GCNAL': 'black',
    'Ours': 'tab:orange',
}

exp_markers = {
    'Random': 'd',
    'Entropy': '^',
    'BALD': 'v',
    'CoreSet': '<',
    'BADGE': '>',
    'DFAL': '|',
    'CDAL': '*',
    'GCNAL': 's',
    'Ours': 'o',
}

plot_legends = True
ignore_initial_point = True


def plot_experiment_results(exp_names, rounds, values, value_name, output_path, x_name='#Labels'):
    if plot_legends:
        plt.ylabel(value_name)
    plt.xlabel(x_name)
    for i, name in enumerate(exp_names):
        mean = values[i].mean(axis=0)# * 100
        std = values[i].std(axis=0)# * 100

        if name in exp_colors:
            plt.plot(rounds, mean, linewidth=1.1, label=name, color=exp_colors[name],
                     marker=exp_markers[name], markersize=4)
            plt.fill_between(rounds, mean - std, mean + std, alpha=0.15, color=exp_colors[name])
        else:
            plt.plot(rounds, mean, linewidth=1.1, label=name)
            plt.fill_between(rounds, mean - std, mean + std, alpha=0.15)

    plt.grid()

    if ignore_initial_point:
        plt.ylim(values.mean(axis=1)[:, 1].min() - 1, values.mean(axis=1)[:, -1].max() + 1)

    #plt.setp(xticks=np.arange(0, 80001, 20000), xticklabels=['0', '20k', '40k', '60k', '80k'])
    if plot_legends:
        plt.legend(fontsize=font_size, ncol=2, borderpad=0.2, labelspacing=0.2, columnspacing=0.5, handletextpad=0.3)

    #plt.show()
    plt.savefig(os.path.join(output_path, '%s.pdf' % value_name), bbox_inches='tight')
    plt.close()


def read_configs(setting_path):
    lst = os.listdir(setting_path)
    lst.sort()
    for exp_name in lst:
        exp = os.path.join(setting_path, exp_name)
        if not os.path.isdir(exp) or not os.path.exists(os.path.join(exp, 'args.json')):
            continue
        with open(os.path.join(exp, 'args.json')) as f:
            return json.load(f)


def visualise_general_path(general_directory, exp_list, exp_alternate_names, metric, max_rounds=-1, vis_changes=False):
    general_scores = np.zeros([len(exp_list), len(exp_list)])
    for dataset_name in os.listdir(general_directory):
        dataset_path = os.path.join(general_directory, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        dataset_scores = visualise_dataset_path(dataset_path, exp_list, exp_alternate_names, metric, max_rounds, vis_changes)

        general_scores += dataset_scores
    plot_matrix(general_scores, exp_list, general_directory, metric)


def visualise_dataset_path(dataset_path, exp_list, exp_alternate_names, metric, max_rounds=-1, vis_changes=False):

    dataset_scores = np.zeros([len(exp_list), len(exp_list)])
    for setting_name in os.listdir(dataset_path):
        setting_path = os.path.join(dataset_path, setting_name)
        if not os.path.isdir(setting_path):
            continue

        scores = visualise_setting_path(setting_path, exp_list, exp_alternate_names, metric, max_rounds=max_rounds,
                                        vis_changes=vis_changes)

        dataset_scores += scores
    plot_matrix(dataset_scores, exp_list, dataset_path, metric)

    return dataset_scores


def get_exp_index(exp_sub_name, exp_list, exp_alternate_names):
    if exp_sub_name in exp_list:
        return exp_list.index(exp_sub_name)
    elif exp_sub_name in exp_alternate_names:
        return exp_alternate_names.index(exp_sub_name)
    else:
        return -1
        # raise Exception('experiment name not found: %s' % exp_sub_name)


def visualise_setting_path(setting_path, exp_list, exp_alternate_names, max_rounds=-1, vis_changes=False):
    configs = read_configs(setting_path)
    lbl_count = configs['n_init_lb']
    rounds = [lbl_count]
    reps = len(configs['seeds'])
    n_rounds = configs['n_round'] if max_rounds <= 0 else min(configs['n_round'], max_rounds)
    for i in range(n_rounds):
        lbl_count += configs['n_query']
        rounds.append(lbl_count)

    exp_accs = np.zeros([len(exp_list), reps, len(rounds)], dtype=np.float)
    exp_times = np.zeros([len(exp_list), reps, len(rounds)], dtype=np.float)
    exp_changes = np.zeros([len(exp_list), reps, len(rounds)], dtype=np.float)
    for exp_name in os.listdir(setting_path):
        exp = os.path.join(setting_path, exp_name)
        if os.path.isdir(exp) or len(exp_name) < 4 or exp_name[-4:] != '.csv':
            continue

        exp_sub_name = exp_name[:exp_name.index('_seed')]
        seed = int(exp_name[exp_name.index('_seed') + 5:exp_name.index('.csv')])
        exp_idx = get_exp_index(exp_sub_name, exp_list, exp_alternate_names)
        if exp_idx < 0:
            continue

        if seed not in configs['seeds']:
            continue

        seed_idx = configs['seeds'].index(seed)

        result_reader = csv.reader(open(exp, 'r'), quoting=csv.QUOTE_ALL)
        accs, times, changes = [], [], []
        for row in result_reader:
            accs.append(row[0])
            times.append(row[1])
            if vis_changes:
                changes.append(row[2])
        assert len(accs) >= n_rounds + 1, 'This experiment is not complete: ' + exp
        accs = accs[:n_rounds + 1]
        times = times[:n_rounds + 1]

        try:
            exp_accs[exp_idx][seed_idx] = np.array(accs, dtype=np.float)
            exp_times[exp_idx][seed_idx] = np.array(times, dtype=np.float)
        except Exception as excp:
            print(excp)

        if vis_changes:
            changes = changes[:n_rounds + 1]
            exp_changes[exp_idx][seed_idx] = np.array(changes, dtype=np.float)

    exp_accs = np.array(exp_accs, dtype=np.float)
    exp_times = np.array(exp_times, dtype=np.float)

    scores = get_comp_matrix(exp_list, exp_accs[:, :, 1:])

    plot_matrix(scores, exp_list, setting_path)

    plot_experiment_results(exp_list, rounds, exp_accs * 100, 'Test Accuracy (%)', setting_path)
    plot_experiment_results(exp_list, np.arange(1, len(rounds)).tolist(), exp_times[:, :, 1:], 'Seconds', setting_path, x_name='AL Round')
    if vis_changes:
        plot_experiment_results(exp_list, np.arange(1, len(rounds)).tolist(), exp_changes[:, :, 1:], '# Flipped Samples', setting_path,
                                x_name='AL Round')

    mean = exp_accs.mean(axis=1)
    std = exp_accs.std(axis=1)
    str = ''
    for i in range(mean.shape[0]):
        str += exp_list[i] + '\t'
        for j in range(mean.shape[1]):
            str += ('& ${:.1f}\scriptstyle\pm{:.1f}$\t'.format(mean[i][j] * 100., std[i][j] * 100))
        str += '\\\\\n'
    print(str)

    mean = exp_times.mean(axis=1)
    std = exp_times.std(axis=1)
    str = ''
    for i in range(mean.shape[0]):
        str += exp_list[i] + '\t'
        for j in range(1, mean.shape[1]):
            #str += ('& ${:.0f}\scriptstyle\pm{:.0f}$\t'.format(mean[i][j] * 100., std[i][j] * 100))
            str += ('& ${:.0f}$\t'.format(mean[i][j] * 100.))
        str += '\\\\\n'
    print(str)

    mean_1 = mean.mean(axis=1)
    std_1 = mean.std(axis=1)
    str = ''
    for i in range(mean_1.shape[0]):
        str += exp_list[i] + '\t'
        str += ('& ${:.0f}\scriptstyle\pm{:.0f}$\\\\\n'.format(mean_1[i], std_1[i]))
    print(str)

    return scores


def plot_matrix(matPlot, algs, setting_path):
    plt.clf()

    col_avg = matPlot.sum(axis=0) / (matPlot.shape[0] - 1)

    matPlot = np.round(matPlot * 10) / 10.
    col_avg = np.round(col_avg * 10) / 10.
    #row_avg = np.round(row_avg * 100) / 100.
    min_e = matPlot.min()
    max_e = matPlot.max()

    plt.rcParams["axes.grid"] = False
    fig, ax = plt.subplots()
    fig.set_size_inches(5.5, 5.2)

    ax.tick_params(axis=u'both', which=u'both', length=0)
    im = ax.matshow(matPlot, cmap='cividis', vmin=min_e, vmax=max_e)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    x_names = copy.copy(algs)
    ax.set_xticklabels([0] + [alg for alg in x_names], rotation=45, fontsize=matrix_fint_size)
    ax.set_yticklabels([0] + [alg for alg in algs], rotation=0, fontsize=matrix_fint_size)

    for i in range(len(algs)):
        for j in range(len(algs)):
            text = ax.text(j, i, matPlot[i, j], ha="center", va="center", color="w", fontsize=matrix_fint_size)

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size=1., pad=-0.2)

    for j in range(len(algs)):
        text = ax2.text(j, 0, col_avg[j], ha="center", va="center", color="w", fontsize=matrix_fint_size)

    im = ax2.matshow(np.array([col_avg]), cmap='cividis', vmin=min_e, vmax=max_e)
    ax2.axis('off')

    fig.subplots_adjust(left=0., right=1., top=.88, bottom=0., wspace=0.05, hspace=0)
    cbar_ax = fig.add_axes([.86, 0.04, 0.03, 0.84])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=matrix_fint_size)
    cbar.ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: int(x)))
    cbar.ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: int(x)))

    plt.savefig(os.path.join(setting_path, 'comp_matrix.pdf'))
    plt.close()


def get_comp_matrix(exp_names, exp_accs):  # badge paper method
    scores = np.zeros([len(exp_names), len(exp_names)])
    for i in range(len(exp_names)):
        if exp_accs[i].sum() <= 0:
            continue
        for j in range(i+1, len(exp_names)):
            if i == j or exp_accs[j].sum() <= 0:
                continue
            dif = exp_accs[i] - exp_accs[j]
            dif_mean = dif.mean(axis=0)
            #dif_var = (1 / (dif.shape[0] - 1)) * np.power(dif - np.repeat(np.expand_dims(dif_mean, axis=0), dif.shape[0], axis=0)[2][0], 2).sum(axis=0)
            dif_var = dif.var(axis=0)
            t_score = math.sqrt(dif.shape[0]) * dif_mean / np.sqrt(dif_var)
            scores[i][j] = (t_score > 2.776).sum()
            scores[j][i] = (t_score < -2.776).sum()

    return scores / (exp_accs.shape[2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', help='directory prefix', type=str,
                        default='../../logs/final_experiments/')
    parser.add_argument('--dir_type', type=str, default='general', choices=['general', 'dataset', 'setting'])
    parser.add_argument('--max_rounds', type=int, default=20)
    parser.add_argument('--vis_changes', action='store_const', default=False, const=True)
    args = parser.parse_args()

    exp_list = [
        'Random',
        'Entropy',
        'BALD',
        'CoreSet',
        'GCNAL',
        'CDAL',
        'BADGE',
        'Ours',
    ]

    exp_alternate_names = [
        'RandomSampling',
        'EntropySampling',
        'BALDDropout',
        'CoreSet',
        'GCNSampling',
        'CDALSampling',
        'BadgeSampling',
        'AlphaMixSampling',
    ]

    if args.dir_type == 'setting':
        visualise_setting_path(args.directory, exp_list, exp_alternate_names, max_rounds=args.max_rounds,
                               vis_changes=args.vis_changes)
    elif args.dir_type == 'dataset':
        visualise_dataset_path(args.directory, exp_list, exp_alternate_names, max_rounds=args.max_rounds,
                               vis_changes=args.vis_changes)
    else:
        visualise_general_path(args.directory, exp_list, exp_alternate_names, max_rounds=args.max_rounds,
                               vis_changes=args.vis_changes)
