import pandas as pd
from scipy.stats import ttest_ind
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from copy import deepcopy
import numpy as np
import random

def generate_signatures(df_ini, category_level, pval_cutoff=0.05, num_top_dims=False):

    ''' Generate signatures for column categories '''

    df_t = df_ini.transpose()

    # remove columns with constant values
    df_t = df_t.loc[:, (df_t != df_t.iloc[0]).any()]

    df = row_tuple_to_multiindex(df_t)

    cell_types = sorted(list(set(df.index.get_level_values(category_level).tolist())))

    keep_genes = []
    keep_genes_dict = {}

    for inst_ct in cell_types:

        inst_ct_mat = df.xs(key=inst_ct, level=category_level)
        inst_other_mat = df.drop(inst_ct, level=category_level)

        inst_stats, inst_pvals = ttest_ind(inst_ct_mat, inst_other_mat, axis=0)

        ser_pval = pd.Series(data=inst_pvals, index=df.columns.tolist()).sort_values()

        if num_top_dims == False:
            ser_pval_keep = ser_pval[ser_pval < pval_cutoff]
        else:
            ser_pval_keep = ser_pval[:num_top_dims]

        inst_keep = ser_pval_keep.index.tolist()
        keep_genes.extend(inst_keep)
        keep_genes_dict[inst_ct] = inst_keep

    keep_genes = sorted(list(set(keep_genes)))

    df_gbm = df.groupby(level=category_level).mean().transpose()
    cols = df_gbm.columns.tolist()
    new_cols = []
    for inst_col in cols:
        new_col = (inst_col, category_level + ': ' + inst_col)
        new_cols.append(new_col)
    df_gbm.columns = new_cols

    df_sig = df_gbm.ix[keep_genes]

    if len(keep_genes) == 0:
        print('found no informative dimensions')

    return df_sig, keep_genes, keep_genes_dict

def predict_cats_from_sigs(df_data_ini, df_sig_ini, dist_type='cosine', predict_level='Predict Category',
                           truth_level=1):
    ''' Predict category using signature '''

    keep_rows = df_sig_ini.index.tolist()
    data_rows = df_data_ini.index.tolist()

    common_rows = list(set(data_rows).intersection(keep_rows))

    df_data = deepcopy(df_data_ini.ix[common_rows])
    df_sig = deepcopy(df_sig_ini.ix[common_rows])

    # calculate sim_mat of df_data and df_sig
    cell_types = df_sig.columns.tolist()
    barcodes = df_data.columns.tolist()
    sim_mat = 1 - pairwise_distances(df_sig.transpose(), df_data.transpose(), metric=dist_type)
    df_sim = pd.DataFrame(data=sim_mat, index=cell_types, columns=barcodes).transpose()

    # get the top column value (most similar signature)
    df_sim_top = df_sim.idxmax(axis=1)

    # add predicted category name to top list
    top_list = df_sim_top.get_values()
    top_list = [ predict_level + ': ' + x[0] for x in top_list]

    # add cell type category to input data
    df_cat = deepcopy(df_data)
    cols = df_cat.columns.tolist()
    new_cols = []

    # check whether the columns have the true category available
    has_truth = False
    if type(cols[0]) is tuple:
        has_truth = True


    if has_truth:
        new_cols = [tuple(list(a) + [b]) for a,b in zip(cols, top_list)]
    else:
        new_cols = [tuple([a] + [b]) for a,b in zip(cols, top_list)]

    # transfer new categories
    df_cat.columns = new_cols

    # keep track of true and predicted labels
    y_info = {}
    y_info['true'] = []
    y_info['pred'] = []

    if has_truth:
        y_info['true'] = [x[truth_level].split(': ')[1] for x in cols]
        y_info['pred'] = [x.split(': ')[1] for x in top_list]



    return df_cat, df_sim.transpose(), y_info

def OLD_predict_cats_from_sigs(df_data_ini, df_sig, dist_type='cosine', predict_level='Predict Category',
                           truth_level=1):
    ''' Predict category using signature '''

    keep_rows = df_sig.index.tolist()
    df_data = deepcopy(df_data_ini.ix[keep_rows])
    # print('df_data: ', df_data.shape)

    # calculate sim_mat of df_data and df_sig
    cell_types = df_sig.columns.tolist()
    barcodes = df_data.columns.tolist()
    sim_mat = 1 - pairwise_distances(df_sig.transpose(), df_data.transpose(), metric=dist_type)
    df_sim = pd.DataFrame(data=sim_mat, index=cell_types, columns=barcodes).transpose()
    # print(df_sim.shape)

    # ser_list = []
    top_list = []
    rows = df_sim.index.tolist()

    for inst_row in rows:

        # make ser_data_sim
        inst_ser = df_sim.loc[[inst_row]]
        inst_data = inst_ser.get_values()[0]
        inst_cols = inst_ser.columns.tolist()
        ser_data_sim = pd.Series(data=inst_data, index=inst_cols, name=inst_row).sort_values(ascending=False)

        # define top matching cell type
        top_ct_1 = ser_data_sim.index.tolist()[0]

        # use cell type signature
        found_ct = top_ct_1

        # make binary matrix of ct_max
        inst_zeros = np.zeros((len(inst_cols)))
        max_ser = pd.Series(data=inst_zeros, index=inst_cols, name=inst_row)
        max_ser[found_ct] = 1
        top_list.append(found_ct)
        ser_list.append(max_ser)

    # # make matrix of top cell type identified
    # df_sim_top = pd.concat(ser_list, axis=1).transpose()

    y_info = {}
    y_info['true'] = []
    y_info['pred'] = []

    # add cell type category to input data
    df_cat = deepcopy(df_data)
    cols = df_cat.columns.tolist()

    # check whether the columns have the true category available
    has_truth = False
    if type(cols[0]) is tuple:
        has_truth = True

    new_cols = []
    for i in range(len(cols)):

        if has_truth:
            inst_col = list(cols[i])
            inst_col.append(predict_level + ': ' + top_list[i][0])
            new_col = tuple(inst_col)
        else:
            inst_col = cols[i]
            new_col = (inst_col, predict_level + ': ' + top_list[i][0])

        new_cols.append(new_col)

        if has_truth:
            # store true and predicted lists
            y_info['true'].append(inst_col[truth_level].split(': ')[1])
            y_info['pred'].append(top_list[i][0])

    df_cat.columns = new_cols

    return df_cat, df_sim.transpose(), df_sim.transpose(), y_info


def confusion_matrix_and_correct_series(y_info):
    ''' Generate confusion matrix from y_info '''


    a = deepcopy(y_info['true'])
    true_count = dict((i, a.count(i)) for i in set(a))

    a = deepcopy(y_info['pred'])
    pred_count = dict((i, a.count(i)) for i in set(a))

    sorted_cats = sorted(list(set(y_info['true'])))
    conf_mat = confusion_matrix(y_info['true'], y_info['pred'], sorted_cats)
    df_conf = pd.DataFrame(conf_mat, index=sorted_cats, columns=sorted_cats)

    total_correct = np.trace(df_conf)
    total_pred = df_conf.sum().sum()
    fraction_correct = total_correct/float(total_pred)

    # calculate ser_correct
    correct_list = []
    cat_counts = df_conf.sum(axis=1)
    all_cols = df_conf.columns.tolist()
    for inst_cat in all_cols:
        inst_correct = df_conf[inst_cat].loc[inst_cat] / cat_counts[inst_cat]
        correct_list.append(inst_correct)

    ser_correct = pd.Series(data=correct_list, index=all_cols)

    populations = {}
    populations['true'] = true_count
    populations['pred'] = pred_count

    return df_conf, populations, ser_correct, fraction_correct


def compare_performance_to_shuffled_labels(df_data, category_level, num_shuffles=100,
                                           random_seed=99, pval_cutoff=0.05, dist_type='cosine',
                                           num_top_dims=False, predict_level='Predict Category',
                                           truth_level=1):
    random.seed(random_seed)

    perform_list = []
    num_shuffles = num_shuffles

    for inst_run in range(num_shuffles + 1):

        cols = df_data.columns.tolist()
        rows = df_data.index.tolist()
        mat = df_data.get_values()

        shuffled_cols = deepcopy(cols)
        random.shuffle(shuffled_cols)

        # do not perform shuffling the first time to confirm that we get the same
        # results as the unshuffled dataaset
        if inst_run == 0:
            df_shuffle = deepcopy(df_data)
        else:
            df_shuffle = pd.DataFrame(data=mat, columns=shuffled_cols, index=rows)

        # generate signature on shuffled data
        df_sig, keep_genes, keep_genes_dict = generate_signatures(df_shuffle, category_level,
                                                                      pval_cutoff=pval_cutoff,
                                                                      num_top_dims=num_top_dims)

        # predict categories from signature
        df_pred_cat, df_sig_sim, y_info = predict_cats_from_sigs(df_shuffle, df_sig,
            dist_type=dist_type, predict_level=predict_level, truth_level=truth_level)

        # calc confusion matrix and performance
        df_conf, populations, ser_correct, fraction_correct = confusion_matrix_and_correct_series(y_info)

        # store performances of shuffles
        if inst_run > 0:
            perform_list.append(fraction_correct)
        else:
            print('performance (fraction correct) of unshuffled: ' + str(fraction_correct))

    perform_ser = pd.Series(perform_list)

    return perform_ser


def box_scatter_plot(df, group, columns=False, rand_seed=100, alpha=0.5,
    dot_color='red', num_row=None, num_col=1, figsize=(10,10),
    start_title='Variable Measurements Across', end_title='Groups',
    group_list=False):

    from scipy import stats
    import pandas as pd

    import matplotlib.pyplot as plt
    # %matplotlib inline

    if columns == False:
        columns = df.columns.tolist()

    plt.figure(figsize=figsize)
    figure_title = start_title + ' ' + group + ' ' + end_title
    plt.suptitle(figure_title, fontsize=20)

    # list of arranged dataframes
    dfs = {}

    for col_num in range(len(columns)):
        column = columns[col_num]
        plot_id = col_num + 1

        # group by column name or multiIndex name
        if group in df.columns.tolist():
            grouped = df.groupby(group)
        else:
            grouped = df.groupby(level=group)

        names, vals, xs = [], [] ,[]

        if type(column) is tuple:
            column_title = column[0]
        else:
            column_title = column

        for i, (name, subdf) in enumerate(grouped):

            names.append(name)

            inst_ser = subdf[column]

            column_name = column_title + '-' + str(name)

            inst_ser.name = column_name
            vals.append(inst_ser)

            np.random.seed(rand_seed)
            xs.append(np.random.normal(i+1, 0.04, subdf.shape[0]))

        ax = plt.subplot(num_row, num_col, plot_id)

        plt.boxplot(vals, labels=names)

        ngroup = len(vals)

        clevels = np.linspace(0., 1., ngroup)

        for x, val, clevel in zip(xs, vals, clevels):

            plt.subplot(num_row, num_col, plot_id)
            plt.scatter(x, val, c=dot_color, alpha=alpha)


        df_arranged = pd.concat(vals, axis=1)

        # anova
        anova_data = [df_arranged[col].dropna() for col in df_arranged]
        f_val, pval = stats.f_oneway(*anova_data)

        if pval < 0.01:
            ax.set_title(column_title + ' P-val: ' + '{:.2e}'.format(pval))
        else:
            pval = round(pval * 100000)/100000
            ax.set_title(column_title + ' P-val: ' + str(pval))

        dfs[column] = df_arranged

    return dfs

def rank_cols_by_anova_pval(df, group, columns=False, rand_seed=100, alpha=0.5, dot_color='red', num_row=None, num_col=1,
                     figsize=(10,10)):

    from scipy import stats
    import numpy as np
    import pandas as pd

    # import matplotlib.pyplot as plt
    # %matplotlib inline

    if columns == False:
        columns = df.columns.tolist()

    # plt.figure(figsize=figsize)

    # list of arranged dataframes
    dfs = {}

    pval_list = []

    for col_num in range(len(columns)):
        column = columns[col_num]
        plot_id = col_num + 1

        # group by column name or multiIndex name
        if group in df.columns.tolist():
            grouped = df.groupby(group)
        else:
            grouped = df.groupby(level=group)

        names, vals, xs = [], [] ,[]

        if type(column) is tuple:
            column_title = column[0]
        else:
            column_title = column

        for i, (name, subdf) in enumerate(grouped):
            names.append(name)

            inst_ser = subdf[column]

            column_name = column_title + '-' + str(name)

            inst_ser.name = column_name
            vals.append(inst_ser)

            np.random.seed(rand_seed)
            xs.append(np.random.normal(i+1, 0.04, subdf.shape[0]))


        ngroup = len(vals)

        df_arranged = pd.concat(vals, axis=1)

        # anova
        anova_data = [df_arranged[col].dropna() for col in df_arranged]
        f_val, pval = stats.f_oneway(*anova_data)

        pval_list.append(pval)

    pval_ser = pd.Series(data=pval_list, index=columns)
    pval_ser = pval_ser.sort_values(ascending=True)

    return pval_ser


def row_tuple_to_multiindex(df):
    import pandas as pd

    from copy import deepcopy
    df_mi = deepcopy(df)
    rows = df_mi.index.tolist()
    titles = []
    for inst_part in rows[0]:

        if ': ' in inst_part:
            inst_title = inst_part.split(': ')[0]
        else:
            inst_title = 'Name'
        titles.append(inst_title)

    new_rows = []
    for inst_row in rows:
        inst_row = list(inst_row)
        new_row = []
        for inst_part in inst_row:
            if ': ' in inst_part:
                inst_part = inst_part.split(': ')[1]
            new_row.append(inst_part)
        new_row = tuple(new_row)
        new_rows.append(new_row)

    df_mi.index = new_rows

    df_mi.index = pd.MultiIndex.from_tuples(df_mi.index, names=titles)

    return df_mi
