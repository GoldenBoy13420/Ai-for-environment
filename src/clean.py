import pandas as pd

def load_full_table(table_name, base_path):
    dfs = []
    
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        path = f"{base_path}/{table_name}25{q}.csv"
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    print(f"{table_name} loaded:", full_df.shape)
    
    return full_df

def group_table(df, name):
    grouped = df.groupby("primaryid").agg(
        lambda x: ", ".join(x.astype(str))
    ).reset_index()
    
    grouped = grouped.rename(
        columns=lambda x: x + "_" + name if x != "primaryid" else x
    )
    
    print(f"{name} grouped")
    
    return grouped


def merge_all(demo, tables):
    df = demo.copy()
    
    for t in tables:
        df = df.merge(t, on="primaryid", how="left")
    
    print("Merging done")
    
    return df

