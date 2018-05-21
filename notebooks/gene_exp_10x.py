def load_gene_exp_to_df(inst_path):
  '''
  Loads gene expression data from 10x in sparse matrix format and returns a
  Pandas dataframe
  '''

  import pandas as pd
  from scipy import io
  from scipy import sparse
  from ast import literal_eval as make_tuple

  # matrix
  Matrix = io.mmread( inst_path + 'matrix.mtx')
  mat = Matrix.todense()

  # genes
  filename = inst_path + 'genes.tsv'
  f = open(filename, 'r')
  lines = f.readlines()
  f.close()

  genes = []
  unique_id = 0
  for inst_line in lines:
      inst_line = inst_line.strip().split()

      if len(inst_line) > 1:
        inst_gene = inst_line[1]
      else:
        inst_gene = inst_line[0]

      genes.append(inst_gene + '_' + str(unique_id))
      unique_id = unique_id + 1

  # barcodes
  filename = inst_path + 'barcodes.tsv'
  f = open(filename, 'r')
  lines = f.readlines()
  f.close()

  cell_barcodes = []
  for inst_bc in lines:
      inst_bc = inst_bc.strip().split('\t')

      # remove dash from barcodes if necessary
      if '-' in inst_bc[0]:
        inst_bc[0] = inst_bc[0].split('-')[0]

      cell_barcodes.append(inst_bc[0])

  # parse tuples if necessary
  try:
      cell_barcodes = [make_tuple(x) for x in cell_barcodes]
  except:
      pass

  try:
      genes = [make_tuple(x) for x in genes]
  except:
      pass

  # make dataframe
  df = pd.DataFrame(mat, index=genes, columns=cell_barcodes)

  return df

def save_gene_exp_to_mtx_dir(inst_path, df):

  import os
  from scipy import io
  from scipy import sparse

  if not os.path.exists(inst_path):
      os.makedirs(inst_path)

  genes = df.index.tolist()
  barcodes = df.columns.tolist()

  save_list_to_tsv(genes, inst_path + 'genes.tsv')
  save_list_to_tsv(barcodes, inst_path + 'barcodes.tsv')

  mat_ge = df.get_values()
  mat_ge_sparse = sparse.coo_matrix(mat_ge)

  io.mmwrite( inst_path + 'matrix.mtx', mat_ge_sparse)

def save_list_to_tsv(inst_list, filename):
    f = open(filename, 'w')
    for inst_line in inst_list:
        f.write(str(inst_line) + '\n')

    f.close()
