# Helper functions for creating train, test and validation sets
def split_by_day(df, train_range, test_range):
    # train range and test range should be length 2 arrays of strings formatted "YYYY-MM-DD"
    train_s, train_e = train_range
    test_s, test_e = test_range
    # partitions a DateTimeIndex-indexed dataframe into train and test sets based on day
    train_set = df[train_s:train_e]
    test_set = df[test_s:test_e]

    return train_set, test_set

# Helper functions for splitting dataframes into X / y columns
# accept index is an option tha will allow you to
def select_all_bla(columns, accept_index=False):
    # gets all columns from a dataframe corresponding to bla outputs
    cols = [column for column in columns if "BLA" in column]
    if accept_index:
        cols.append("index")
    return cols

def select_one_bla(columns, bla_number, accept_index=False):
    # given the number of a BLA (1-6) selects the columns related to that
    bla = "BLA_%d" % bla_number
    cols = [column for column in columns if bla in column]
    if accept_index:
        cols.append("index")
    return cols

def select_non_bla(columns, accept_index=False):
    # gets all columns from a dataframe not related to BLA outputs
    # aka the X for an experiment
    cols = [column for column in columns if "BLA" not in column]
    if accept_index:
        cols.append("index")
    return cols

def select_one_quantile(columns, quantile, accept_index=False):
    # given a set of columns, filter out those relating to a single quantile
    # inputs: "median", "5th_quantile", "95th_quantile
    cols = [column for column in columns if quantile in column]
    if accept_index:
        cols.append("index")
    return cols

def split_X_into_quantiles(data, accept_index=False):
    # given an input dataframe, split it into 3 separate numpy matrices by quantile
    X = data[select_non_bla(data.columns)]

    X_95th = X[select_one_quantile(X.columns, "_95th_quantile", accept_index=accept_index)]
    X_median = X[select_one_quantile(X.columns, "_median", accept_index=accept_index)]
    X_5th = X[select_one_quantile(X.columns, "_5th_quantile", accept_index=accept_index)]

    return X_95th.to_numpy(), X_median.to_numpy(), X_5th.to_numpy()


def split_Y_into_beams_and_quantiles(data):
    # given an input dataframe, split into quantiles and return 3 lists of np arrays, where each array contains all the
    # beam outputs corresponding to that quantile
    Y_95th = data[select_one_quantile(data.columns, "_95th_quantile")]
    Y_median = data[select_one_quantile(data.columns, "_median")]
    Y_5th = data[select_one_quantile(data.columns, "_5th_quantile")]

    # convert each dataframe into a list of np arrays, where the ith item corresponds to the ith bla output
    Y_95th = [Y_95th[select_one_bla(Y_95th.columns, i)].to_numpy() for i in range(1, 7)]
    Y_median = [Y_median[select_one_bla(Y_median.columns, i)].to_numpy() for i in range(1, 7)]
    Y_5th = [Y_5th[select_one_bla(Y_5th.columns, i)].to_numpy() for i in range(1, 7)]

    return Y_95th, Y_median, Y_5th