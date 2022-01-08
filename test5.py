import pandas as pd
df = [[1, 2], [3, 4]]
fn = "output/test.tsv"
df = pd.DataFrame(df, columns=['index', 'prediction'])
df.to_csv(fn, sep='\t', index=False)
