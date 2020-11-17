# Helper functions for creating train, test and validation sets

def split_by_day(df, train_s, train_e, test_s, test_e):
    # partitions a DateTimeIndex-indexed dataframe into train and test sets based on day
    train_set = df[train_s:train_e]
    test_set = df[test_s:test_e]

    return train_set, test_set

# Helper functions for splitting dataframes into X / y columns

def select_one_bla(columns, bla_number):
    # given the number of a BLA (1-6) selects the columns related to that
    bla = "BLA_%d" % bla_number
    return [column for column in columns if bla in column]

def select_non_bla(columns):
    # gets all columns from a dataframe not related to BLA outputs
    # aka the X for an
    return [column for column in columns if "BLA" not in column]

def select_one_quantile(columns, quantile):
    # given a set of columns, filter out those relating to a single quantile
    # inputs: "mean", "5th_quantile", "95th_quantile
    return [column for column in columns if quantile in column]