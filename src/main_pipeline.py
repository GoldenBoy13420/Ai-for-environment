from .transformer import convert_all_tables
from .merger import load_full_table, group_table, merge_all

def run_pipeline():
    
    raw_path = "Data/Datatxt"
    processed_path = "Data/Datacsv"
    
    
    # Step 1: Converting TXT → CSV
    convert_all_tables(raw_path, processed_path)
    
    # Step 2: Loading data
    demo = load_full_table("DEMO", processed_path)
    drug = load_full_table("DRUG", processed_path)
    reac = load_full_table("REAC", processed_path)
    outc = load_full_table("OUTC", processed_path)
    rpsr = load_full_table("RPSR", processed_path)
    ther = load_full_table("THER", processed_path)
    indi = load_full_table("INDI", processed_path)
    #Step 3: Grouping
    drug_g = group_table(drug, "drug")
    reac_g = group_table(reac, "reac")
    outc_g = group_table(outc, "outc")
    rpsr_g = group_table(rpsr, "rpsr")
    ther_g = group_table(ther, "ther")
    indi_g = group_table(indi, "indi")
    
     #Step 4: Merging
    final_df = merge_all(demo, [drug_g, reac_g, outc_g, rpsr_g, ther_g, indi_g ])
    

    
    return final_df