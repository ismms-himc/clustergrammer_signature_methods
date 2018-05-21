def box_scatter_plot(df, group, columns=False, rand_seed=100, alpha=0.5,
    dot_color='red', num_row=None, num_col=1, figsize=(10,10),
    start_title='Variable Measurements Across', end_title='Groups',
    group_list=False):

    from scipy import stats
    import numpy as np
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
