from glob import glob
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import itertools
import re


#
#   Class for creating sparse matrices from sequential data
#
#   Used clean dataset of web-browsed logs from the paper:
#   A Tool for Classification of Sequential Data. Giacomo Kahn, Yannick Loiseau and Olivier Raynaud
#   http://ceur-ws.org/Vol-1703/paper12.pdf
#
#
#   Using the class you can:
#   1) Read csv files with given name patterns
#   2) Use sliding windows on log reading
#   3) Prepare train-ready data sparse matrices for sklearn
#
#   At the time of this writing, the following sklearn 0.18.1 algorithms accept sparse matrices[1]:
#
#   Logistic Regression, SGDClassifier, DecisionTreeClassifier, BaggingClassifier, RandomForestClassifier,
#   Linear Regression, ElasticNet, AdaBoostRegressor, SVR and many others.[1]
#
#   [1] Full list - https://dziganto.github.io/Sparse-Matrices-For-Efficient-Machine-Learning/
#
class SequentialData2Sparse:

    def __init__(self):
        self.output = None

    #
    #   Function constructs output compressed sparse matrice from prepared data
    #   using scipy.sparse.csr_matrix function
    #
    #
    @staticmethod
    def get_sparse_matrice(data, rownum, colnum):
        index = 0
        col = []
        row = []
        prepared_data = []

        for x in enumerate(data):
            nonzero = np.ma.masked_equal(x[1], 0).compressed()  # masked zero
            counted = np.unique(nonzero, return_counts=True)

            col_item = [xx - 1 for xx in counted[0]]
            row_item = [index] * len(col_item)
            data_item = counted[1]

            col.append(col_item)
            row.append(row_item)
            prepared_data.append(data_item)
            index += 1

        cols = np.array(list(itertools.chain.from_iterable(col)))
        rows = np.array(list(itertools.chain.from_iterable(row)))
        all_data = np.array(list(itertools.chain.from_iterable(prepared_data)))

        output = csr_matrix((all_data, (rows, cols)), shape=(rownum, colnum))

        return output

    #
    #   Test data:
    #   Set of csv files named using pattern user[id].csv
    #   First column - timestamp - log timestamp
    #   Second column - site - visited site
    #
    #   Outputs dataframe  df[rowid,'timestamp','site','user_id']
    #
    def test_csv_read(self, path, pattern='/user*.csv'):

        dataset_files = sorted(glob(path + pattern))
        count = 0
        df = pd.DataFrame()

        # make one dataframe
        for row in dataset_files:

            # pick up user id
            m = re.compile(".*user(.*).csv.*").match(row).groups()

            if len(m) > 0:
                userid = int(m[0])

                if count == 0:
                    df = pd.read_csv(row)
                    df['user_id'] = userid
                else:
                    df_userid = pd.read_csv(row)
                    df_userid['user_id'] = userid

                    df = df.append(df_userid, ignore_index=True)
                count += 1
        return df

    #
    #
    #   Function converts any input dataframe to compressed sparse matrice
    #   Function don't use timestamp column in assumption of data sequentially logged.
    #
    #   window_size - number of logged items in one-hot encoding matrice
    #   session_length - number of logged items used for window
    #
    #   Variable session_length can be more than window_size, for example, if you
    #   want to read data using sliding window.
    #   That sometimes useful in time-related analysis
    #
    #   You can simply rename head in dataframe
    #   df['site'] - web-browsed site
    #   df['user_id'] - id of logged user
    #   df['timestamp'] - date and time of log item
    #
    def convert_dataframe(self, df, session_length=10, window_size=10):
        # make frequency dictionary
        udf = df['site'].value_counts(sort=True)

        unique = {}
        i = 0
        for index, row in udf.iteritems():
            i += 1
            unique[index] = i, row

        n = session_length + 1

        # make output dataframe
        split = 1
        out = []
        line = np.zeros((1, n))
        user_id = 0
        prev_user_id: int = 0
        as_matrix = df.as_matrix()
        window_step = 0
        i = 0

        while i < len(as_matrix):

            # get first user
            if user_id == 0:
                prev_user_id = as_matrix[i][2]

            user_id = as_matrix[i][2]
            site_id = unique[as_matrix[i][1]][0]

            line[0, split - 1] = site_id

            # check end or change user
            if user_id != prev_user_id:
                line[0, split - 1] = 0
                split = session_length

            if (i == (len(as_matrix) - 1)) and (window_step + window_size > (len(as_matrix) - 1)):
                split = session_length

            if split < session_length:
                i += 1

            else:  # limit

                if user_id != prev_user_id:
                    line[0, n - 1] = prev_user_id
                    prev_user_id = user_id
                    window_step = i
                else:
                    line[0, n - 1] = user_id
                    window_step += window_size
                    if i <= (len(as_matrix) - 1):
                        i = window_step
                    else:
                        out.append(line)
                        break

                out.append(line)
                line = np.zeros((1, n))
                split = 0

            split += 1

        data = np.concatenate(out)

        return self.get_sparse_matrice(data[:, :-1], data.shape[0], len(unique.keys())), data[:, -1]


#
# demo
#


PATH_TO_DATA = './data/test'
reader = SequentialData2Sparse()

# read csv data to dataframe
df = reader.test_csv_read(PATH_TO_DATA)

print('Sample data')
print(df.head())
print('-' * 10)

# convert dataframe to sparse matrice
X, y = reader.convert_dataframe(df)

print('Sparse matrice')
print(X)
print('-' * 10)
print('Densed matrice')
print(X.todense())
print('-' * 10)
print('Y matrice')
print(y)
