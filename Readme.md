# Drug Safety Intelligence System

## Overview
This project aims to analyze adverse drug events and predict the severity of harm caused by medications using real-world data.

## Objective
- Understand drug safety patterns  
- Analyze relationships between drugs and adverse reactions  
- Build a system to predict harm severity (Mild, Moderate, Severe)

## Data Source
The dataset is obtained from the FDA Adverse Event Reporting System (FAERS).

- Source: FDA  
- Data Link: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html  

## Dataset Used
We used FAERS data for the year **2025**.

The dataset includes multiple tables:
- DEMO (patient information)  
- DRUG (drug details)  
- REAC (adverse reactions)  
- THER (therapy duration)  
- OUTC (outcomes)  
- RPSR (report sources)

## Data Processing
- Converted raw `.txt` files to `.csv`  
- Cleaned and handled missing values  
- Grouped records by `primaryid`  
- Integrated all tables into a single dataset  

## Tools & Technologies
- Python  
- Pandas  
- Jupyter Notebook  
- Git & GitHub  

## Project Structure
Project/
│
├── Data/ # Datatxt and Datacsv (ignored in Git)
├── Notebooks/ # Data integration notebook
├── README.md

## Notes
- The dataset is not included in this repository due to size limitations.  
- You can download the data from the official FDA link above.