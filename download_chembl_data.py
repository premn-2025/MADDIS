#!/usr/bin/env python3
"""
Download ChEMBL Bioactivity Data for QSAR Model Training

Downloads IC50 data for 5 drug targets:
- EGFR (CHEMBL203)
- COX2 (CHEMBL230)
- BACE1 (CHEMBL4822)
- JAK2 (CHEMBL2971)
- THROMBIN (CHEMBL204)

Outputs: CSV files in data/ with SMILES and pIC50 values
"""

import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem

def download_target_data(target_id: str, target_name: str, output_dir: str = "data") -> pd.DataFrame:
    """Download bioactivity data from ChEMBL REST API for a specific target.
    
    Uses direct REST API calls with pagination for reliability.
    
    Args:
        target_id: ChEMBL target ID (e.g., 'CHEMBL203')
        target_name: Human-readable name (e.g., 'EGFR')
        output_dir: Directory to save CSV files
        
    Returns:
        DataFrame with columns ['smiles', 'pIC50']
    """
    import requests as req
    
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    
    all_records = []
    offset = 0
    limit = 1000  # Page size
    max_records = 10000  # Safety cap
    
    print(f"  Querying ChEMBL REST API for {target_name} ({target_id})...")
    
    while offset < max_records:
        params = {
            'target_chembl_id': target_id,
            'standard_type': 'IC50',
            'standard_relation': '=',
            'assay_type': 'B',
            'standard_units': 'nM',
            'limit': limit,
            'offset': offset,
            'format': 'json',
        }
        
        try:
            resp = req.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  API request failed at offset {offset}: {e}")
            break
        
        activities = data.get('activities', [])
        if not activities:
            break
        
        for act in activities:
            smiles = act.get('canonical_smiles')
            value = act.get('standard_value')
            if smiles and value:
                all_records.append({
                    'canonical_smiles': smiles,
                    'standard_value': value,
                })
        
        total_count = data.get('page_meta', {}).get('total_count', '?')
        print(f"    Page {offset // limit + 1}: got {len(activities)} records "
              f"(total so far: {len(all_records)}, API total: {total_count})")
        
        # Check if there are more pages
        if data.get('page_meta', {}).get('next') is None:
            break
        
        offset += limit
    
    df = pd.DataFrame(all_records)
    print(f"  Raw records: {len(df)}")
    
    if len(df) == 0:
        print(f"  WARNING: No IC50 data. Trying Ki as fallback...")
        offset = 0
        all_records = []
        while offset < max_records:
            params = {
                'target_chembl_id': target_id,
                'standard_type': 'Ki',
                'standard_relation': '=',
                'assay_type': 'B',
                'standard_units': 'nM',
                'limit': limit,
                'offset': offset,
                'format': 'json',
            }
            try:
                resp = req.get(base_url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  Ki API request failed: {e}")
                break
            activities = data.get('activities', [])
            if not activities:
                break
            for act in activities:
                smiles = act.get('canonical_smiles')
                value = act.get('standard_value')
                if smiles and value:
                    all_records.append({
                        'canonical_smiles': smiles,
                        'standard_value': value,
                    })
            if data.get('page_meta', {}).get('next') is None:
                break
            offset += limit
        df = pd.DataFrame(all_records)
        print(f"  Raw Ki records: {len(df)}")
    
    if len(df) == 0:
        print(f"  ERROR: No data available for {target_name}")
        return pd.DataFrame(columns=['smiles', 'pIC50'])
    
    # Clean data
    df = df[df['canonical_smiles'].notna()].copy()
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df[df['standard_value'].notna()].copy()
    
    # Remove extreme outliers (IC50 < 0.01 nM or > 100,000 nM)
    df = df[(df['standard_value'] >= 0.01) & (df['standard_value'] <= 100000)].copy()
    
    print(f"  After filtering: {len(df)} records")
    
    # Convert IC50 (nM) to pIC50
    df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)
    
    # Remove duplicates: keep median pIC50 per SMILES
    df = df.groupby('canonical_smiles')['pIC50'].median().reset_index()
    df.columns = ['smiles', 'pIC50']
    
    # Validate SMILES with RDKit
    valid_rows = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            # Additional filters: reasonable drug-like molecules
            num_atoms = mol.GetNumHeavyAtoms()
            if 5 <= num_atoms <= 100:  # Skip tiny fragments and huge molecules
                valid_rows.append(row)
    
    result_df = pd.DataFrame(valid_rows)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{target_name}_chembl.csv")
    result_df.to_csv(output_path, index=False)
    
    print(f"  {target_name}: {len(result_df)} valid compounds saved to {output_path}")
    print(f"  pIC50 range: {result_df['pIC50'].min():.2f} - {result_df['pIC50'].max():.2f}")
    print(f"  pIC50 median: {result_df['pIC50'].median():.2f}")
    
    return result_df


def main():
    """Download ChEMBL data for all 5 targets."""
    
    targets = {
        'CHEMBL203': 'EGFR',
        'CHEMBL230': 'COX2',
        'CHEMBL4822': 'BACE1',
        'CHEMBL2971': 'JAK2',
        'CHEMBL204': 'THROMBIN'
    }
    
    print("=" * 60)
    print("ChEMBL Bioactivity Data Download")
    print(f"Targets: {', '.join(targets.values())}")
    print("=" * 60)
    
    results = {}
    
    for target_id, target_name in targets.items():
        print(f"\n--- {target_name} ({target_id}) ---")
        try:
            df = download_target_data(target_id, target_name)
            results[target_name] = len(df)
        except Exception as e:
            print(f"  ERROR downloading {target_name}: {e}")
            results[target_name] = 0
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    total = 0
    for target_name, count in results.items():
        status = "OK" if count > 100 else "LOW" if count > 0 else "FAILED"
        print(f"  {target_name:10s}: {count:5d} compounds  [{status}]")
        total += count
    print(f"  {'TOTAL':10s}: {total:5d} compounds")
    print("=" * 60)
    
    if total == 0:
        print("\nERROR: No data downloaded. Check your internet connection.")
        sys.exit(1)
    
    print("\nNext step: python train_qsar_models.py")


if __name__ == "__main__":
    main()
