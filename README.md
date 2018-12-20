# Simple class for CSR matrix generating from text data 

Useful python script for converting csv-log files to compressed sparse row.
  - Compressed sparse matrice preparation 
  - Sliding window reading capability
  - Numpy realization

# Sparse matrices for efficient machine learning

Sparse matrices are regular used in machine learning.
But it have lots of zero values, we can apply special 
algorithms that will do two important things:
- compress the memory footprint of our matrix object
- speed up many machine learning routines

Since storing all those zero values is a waste, we can apply data 
compression techniques to minimize the amount of data we need to store. 
That is not the only benefit, however. Users of sklearn will note that all 
native machine learning algorithms require data matrices to be in-memory. 
Said another way, the machine learning process breaks down when a data 
matrix (usually called a dataframe) does not fit into RAM. One of the 
perks of converting a dense data matrix to sparse is that in many cases 
it is possible to compress it so that it can fit in RAM.[1]

The compressed sparse row (CSR) or compressed row storage (CRS) format 
represents a matrix M by three (one-dimensional) arrays, that respectively contain nonzero values, the extents of rows, and column indices. 
It is similar to COO, but compresses the row indices, hence the name. 
This format allows fast row access and matrix-vector multiplications (Mx).
The CSR format has been in use since at least the mid-1960s,
with the first complete description appearing in 1967.[2]

# Snippet class example

**Trivial task:**
Read provided log files and prepare it for machine learning model that predicts user behaviour

Dataset of web-browsed logs was used from the paper:
[A Tool for Classification of Sequential Data. Giacomo Kahn, Yannick Loiseau and Olivier Raynaud](http://ceur-ws.org/Vol-1703/paper12.pdf)


**Solution:**
Python script reads csv files using sliding window and creates dataframe.
For learning efficiency and memory usage optimization script also converts data frame to compressed sparse matrice.

**Using:**
1. Read csv data to dataframe. By default session size is 10 sites. 
Window is equal to session size (no sliding).
```python
PATH_TO_DATA = './data/test'
reader = SequentialData2Sparse()
df = reader.test_csv_read(PATH_TO_DATA)
print(df.head())
```
```sh
             timestamp             site  user_id
0  2013-11-15 09:28:17           vk.com        1
1  2013-11-15 09:33:04       oracle.com        1
2  2013-11-15 09:52:48       oracle.com        1
3  2013-11-15 11:37:26  geo.mozilla.org        1
4  2013-11-15 11:40:32       oracle.com        1
```
2. Convert dataframe to sparse matrice
```python
X, y = reader.convert_dataframe(df)
print (X) #sparse
print (X.todense()) #densed
print (y) #target values
```
```sh
  (0, 0)	1
  (0, 1)	3
  (0, 2)	1
  (0, 4)	1
  (0, 7)	1
  (0, 8)	1
  
[[1 3 1 0 1 0 0 1 1 1 1]
 [3 0 1 0 0 0 0 0 0 0 0]
 [0 2 1 0 0 2 0 0 0 0 0]
 [4 2 0 2 1 0 1 0 0 0 0]
 [1 1 0 1 0 0 0 0 0 0 0]]
 
 [1. 1. 2. 3. 3.]
...
```
# How to start?
```sh
git clone https://github.com/alm4z/sparse-from-sequential-data.git
cd sparse-from-sequential-data
python data2sparse.py
```

# Contact
If you have any further questions or suggestions, please, do not hesitate to contact  by email at a.sadenov@gmail.com.

# References
1. Great article from David Ziganto - [Sparse Matrices For Efficient Machine Learning
](https://dziganto.github.io/Sparse-Matrices-For-Efficient-Machine-Learning/)

2. Wikipedia article - [Sparse Matrices](https://en.wikipedia.org/wiki/Sparse_matrix) 

# License
MIT
