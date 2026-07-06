import sqlite3
import pandas as pd
from src.config import DB_PATH, MATRIX_A, MATRIX_B

def build_severity_matrix():
    print("🚀 Building Severity Matrix A (v6)...")
    conn = sqlite3.connect(DB_PATH)
    
    query = """
    WITH 
    PatientOutcome AS (SELECT primaryid, MAX(CASE WHEN outc_cod IN ('DE', 'LT', 'HO', 'DS', 'CA', 'RI') THEN 1 ELSE 0 END) as is_severe FROM outc_clean GROUP BY primaryid),
    PrimaryDrug AS (SELECT primaryid, MAX(final_drug_name) as primary_suspect_drug, MAX(route) as ps_route FROM drug_clean WHERE role_cod = 'PS' AND final_drug_name != 'none' GROUP BY primaryid),
    Polypharmacy AS (SELECT primaryid, COUNT(final_drug_name) as num_drugs FROM drug_clean GROUP BY primaryid),
    TherapyDuration AS (SELECT primaryid, MAX(dur) as therapy_duration FROM ther_clean WHERE dur IS NOT NULL GROUP BY primaryid),
    ReportSource AS (SELECT primaryid, MAX(rpsr_cod) as rpsr_cod FROM rpsr_clean GROUP BY primaryid),
    PatientIndications AS (SELECT primaryid, COUNT(indi_pt) as num_indications, MAX(indi_pt) as primary_indication FROM indi_clean WHERE indi_pt IS NOT NULL GROUP BY primaryid)
    
    SELECT d.primaryid, d.age, d.wt, d.sex, d.occp_cod, d.rept_cod, d.is_test_set, p.primary_suspect_drug, p.ps_route,
           COALESCE(poly.num_drugs, 1) as num_drugs, t.therapy_duration, rs.rpsr_cod, COALESCE(ind.num_indications, 0) as num_indications, ind.primary_indication, COALESCE(o.is_severe, 0) as is_severe
    FROM demo_clean d
    INNER JOIN PrimaryDrug p ON d.primaryid = p.primaryid
    LEFT JOIN Polypharmacy poly ON d.primaryid = poly.primaryid
    LEFT JOIN TherapyDuration t ON d.primaryid = t.primaryid
    LEFT JOIN ReportSource rs ON d.primaryid = rs.primaryid
    LEFT JOIN PatientIndications ind ON d.primaryid = ind.primaryid
    LEFT JOIN PatientOutcome o ON d.primaryid = o.primaryid
    WHERE d.age IS NOT NULL AND p.primary_suspect_drug IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    df.to_sql(MATRIX_A, conn, if_exists='replace', index=False)
    conn.close()
    print(f"✅ Matrix A Built! Shape: {df.shape}")

def build_reaction_matrix():
    print("🚀 Building Adverse Event Matrix B using Pandas...")
    conn = sqlite3.connect(DB_PATH)
    
    demo = pd.read_sql("SELECT primaryid, age, wt, sex, occp_cod, rept_cod, is_test_set FROM demo_clean WHERE age IS NOT NULL", conn)
    reac = pd.read_sql("SELECT primaryid, pt FROM reac_clean WHERE pt IS NOT NULL", conn)
    drug = pd.read_sql("SELECT primaryid, final_drug_name, role_cod, route FROM drug_clean", conn)
    ther = pd.read_sql("SELECT primaryid, dur FROM ther_clean WHERE dur IS NOT NULL", conn)
    rpsr = pd.read_sql("SELECT primaryid, rpsr_cod FROM rpsr_clean", conn)
    indi = pd.read_sql("SELECT primaryid, indi_pt FROM indi_clean WHERE indi_pt IS NOT NULL", conn)

    reac_target = reac.sort_values(by='pt').drop_duplicates(subset=['primaryid'], keep='first')
    reac_target.rename(columns={'pt': 'target_reaction'}, inplace=True)

    drug['final_drug_name'] = drug['final_drug_name'].astype(str)
    drug['route'] = drug['route'].fillna('Unknown').astype(str)

    drug_ps = drug[(drug['role_cod'] == 'PS') & (drug['final_drug_name'] != 'none')]
    drug_primary = drug_ps.groupby('primaryid').agg(
        primary_suspect_drug=('final_drug_name', 'max'),
        ps_route=('route', 'max')
    ).reset_index()

    polypharmacy = drug.groupby('primaryid').size().reset_index(name='num_drugs')
    therapy = ther.groupby('primaryid').agg(therapy_duration=('dur', 'max')).reset_index()
    report_src = rpsr.groupby('primaryid').agg(rpsr_cod=('rpsr_cod', 'max')).reset_index()
    indications = indi.groupby('primaryid').agg(num_indications=('indi_pt', 'size'), primary_indication=('indi_pt', 'max')).reset_index()

    df = demo.merge(drug_primary, on='primaryid', how='inner')
    df = df.merge(reac_target, on='primaryid', how='inner')
    df = df.merge(polypharmacy, on='primaryid', how='left')
    df['num_drugs'] = df['num_drugs'].fillna(1)
    df = df.merge(therapy, on='primaryid', how='left')
    df = df.merge(report_src, on='primaryid', how='left')
    df = df.merge(indications, on='primaryid', how='left')
    df['num_indications'] = df['num_indications'].fillna(0)

    df.to_sql(MATRIX_B, conn, if_exists='replace', index=False)
    conn.close()
    print(f"✅ Matrix B Built! Shape: {df.shape}")

if __name__ == "__main__":
    build_severity_matrix()
    build_reaction_matrix()