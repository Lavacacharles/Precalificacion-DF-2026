import pandas as pd

# df = pd.read_csv('data/metadata.csv')
# with open('docs/metadata.md', 'w') as f:
    # f.write(df.to_markdown())

df = pd.read_csv('data/train.csv', low_memory=False)

