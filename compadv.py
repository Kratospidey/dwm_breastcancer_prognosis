#!/usr/bin/env python3
"""
Reworked METABRIC Data Integration Script with Advanced Memory Optimizations,
Optimized Imputation, Conditional Normalization (Clinical data preserved),
VAE-based Dimensionality Reduction for Omics Data, and SHAP-based Feature Selection.

Overview:
- Reads individual METABRIC files (clinical, mRNA, CNA, methylation, gene panel, mutations).
- Applies a VarianceThreshold filter to drop near-constant numerical features before imputation.
- Uses smart imputation on numerical data with either IterativeImputer or (optionally) KNNImputer.
- Optionally applies autoencoder-based imputation.
- Normalizes continuous features via Z-score normalization, unless disabled (for clinical data).
- Encodes categorical variables via one-hot encoding.
- Reduces high-dimensional omics features using a VAE. Clinical numeric columns are excluded.
- Uses a RandomForest classifier with SHAP to select top features.
- Merges processed datasets on SAMPLE_ID and exports a combined file.
- Filters the clinical data to match the final integrated dataset.
- Saves all outputs at the end with their designated names.

Usage:
    python metabric_integration_reworked.py [--use-cudf] [--autoencoder] [--knn]

References:
• VarianceThreshold: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html  
• PyTorch VAE Tutorial: https://pytorch.org/tutorials/beginner/variational_autoencoder.html  
• SHAP Documentation: https://shap.readthedocs.io/en/latest/
"""

import os
import sys
import argparse
import numpy as np
import pandas as host_pd  # using host pandas for internal processing
import logging

# Fix for IterativeImputer (experimental)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import shap

import torch
import torch.nn as nn
import torch.optim as optim

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Global variable for backend (pandas vs cuDF)
USE_CUDF = False

def import_backend(use_cudf: bool):
    global USE_CUDF, pd
    USE_CUDF = use_cudf
    if USE_CUDF:
        try:
            import cudf as pd
            logger.info("Using cuDF as the DataFrame backend.")
        except ImportError:
            logger.error("cuDF requested but not installed; falling back to pandas.")
            import pandas as pd
            USE_CUDF = False
    else:
        import pandas as pd
        logger.info("Using pandas as the DataFrame backend.")
    return pd

def make_unique(names):
    seen = {}
    unique_names = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique_names.append(name)
    return unique_names

def series_to_list(s):
    try:
        return s.to_pandas().tolist() if USE_CUDF and hasattr(s, "to_pandas") else s.tolist()
    except AttributeError:
        return list(s)

def read_file(filename, sep='\t', comment='#', engine=None):
    skiprows = None
    header = 'infer'
    if "data_clinical_patient" in filename or "data_clinical_sample" in filename:
        skiprows = 4
        comment = None
        header = 0
    if USE_CUDF and skiprows is None:
        skiprows = 0
    logger.info(f"Reading file: {filename} (skiprows={skiprows}, comment={comment})")
    try:
        if USE_CUDF:
            return pd.read_csv(filename, sep=sep, skiprows=skiprows, comment=comment)
        else:
            return host_pd.read_csv(filename, sep=sep, skiprows=skiprows, comment=comment,
                               header=header, engine=engine or 'python')
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        sys.exit(1)

def read_wide_file_and_transpose(filename, id_cols=None):
    logger.info(f"Reading and transposing wide file: {filename}")
    df = read_file(filename, sep='\t', engine='python')
    if USE_CUDF:
        df.columns = [col.strip() for col in df.columns]
    else:
        df.columns = df.columns.str.strip()
    if id_cols is None:
        id_cols = [0]
    if USE_CUDF:
        gene_names = host_pd.Series(df.iloc[:, id_cols].to_pandas().astype(str).agg("_".join, axis=1))
    else:
        gene_names = df.iloc[:, id_cols].astype(str).agg("_".join, axis=1)
    gene_names_list = series_to_list(gene_names)
    unique_gene_names = make_unique(gene_names_list)
    if USE_CUDF:
        import cudf
        gene_names = cudf.Series(unique_gene_names)
    else:
        gene_names = host_pd.Series(unique_gene_names)
    data_df = df.drop(columns=[df.columns[i] for i in id_cols])
    data_df = data_df.astype('float32')
    df_transposed = data_df.transpose()
    df_transposed.columns = gene_names
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'SAMPLE_ID'}, inplace=True)
    logger.info(f"Transposed file {filename}: {df_transposed.shape[0]} samples and {df_transposed.shape[1]-1} data columns.")
    return df_transposed

def aggregate_mutations(filename):
    logger.info(f"Aggregating mutations from: {filename}")
    mut_df = read_file(filename, sep='\t', engine='python')
    if 'SAMPLE_ID' not in mut_df.columns:
        if 'Tumor_Sample_Barcode' in mut_df.columns:
            mut_df = mut_df.rename(columns={'Tumor_Sample_Barcode': 'SAMPLE_ID'})
            logger.info("Renamed 'Tumor_Sample_Barcode' to 'SAMPLE_ID'.")
        else:
            logger.error("No sample ID column found in mutations file.")
            return None
    if 'Hugo_Symbol' not in mut_df.columns:
        logger.error("Expected column 'Hugo_Symbol' not found in mutations file.")
        return None
    if USE_CUDF:
        mut_df = mut_df.to_pandas()
    agg = mut_df.groupby('SAMPLE_ID')['Hugo_Symbol'].apply(lambda x: ','.join(x.astype(str).unique())).reset_index()
    agg.rename(columns={'Hugo_Symbol': 'Aggregated_Mutations'}, inplace=True)
    logger.info("Completed aggregating mutations.")
    return agg

def final_cleaning(df):
    logger.info("Performing final cleaning...")
    df.columns = [str(col).strip() if col is not None else "" for col in df.columns]
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace("NA", np.nan)
    return df

def smart_impute(df, strategy='iterative', use_knn=False):
    logger.info("Performing smart imputation...")
    if USE_CUDF and hasattr(df, "to_pandas"):
        df = df.to_pandas()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        vt = VarianceThreshold(threshold=1e-5)
        try:
            df_num = host_pd.DataFrame(vt.fit_transform(df[num_cols]),
                                       index=df.index,
                                       columns=host_pd.Index(np.array(num_cols)[vt.get_support()]))
            df.update(df_num)
        except ValueError:
            logger.warning("No numeric features passed the variance threshold.")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    imputer = KNNImputer() if use_knn else IterativeImputer(random_state=42, max_iter=10)
    if num_cols:
        df[num_cols] = imputer.fit_transform(df[num_cols])
    for col in cat_cols:
        df[col] = df[col].fillna("Missing")
    return df

class AutoencoderImputer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(AutoencoderImputer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

def autoencoder_impute(df, epochs=30, batch_size=32, hidden_dim=64, lr=1e-3):
    logger.info("Performing autoencoder-based imputation...")
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return df
    data = df[num_cols].values.astype(np.float32)
    col_means = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(col_means, inds[1])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_data = torch.tensor(data).to(device)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AutoencoderImputer(input_dim=data.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch_data = batch[0]
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_data.size(0)
        logger.info(f"Autoencoder Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataset):.4f}")
    model.eval()
    with torch.no_grad():
        imputed_data = model(torch.tensor(data).to(device)).cpu().numpy()
    df[num_cols] = imputed_data
    return df

def normalize_features(df, feature_list):
    logger.info("Normalizing continuous features...")
    if not feature_list:
        logger.warning("No features provided for normalization.")
        return df
    scaler = StandardScaler()
    df[feature_list] = scaler.fit_transform(df[feature_list])
    return df

def encode_categorical(df, categorical_cols, encoding_strategy='onehot'):
    logger.info("Encoding categorical variables...")
    if encoding_strategy == 'onehot' and categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[categorical_cols])
        encoded_df = host_pd.DataFrame(encoded,
                                  columns=encoder.get_feature_names_out(categorical_cols),
                                  index=df.index)
        df = host_pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    elif encoding_strategy == 'binary' and categorical_cols:
        try:
            import category_encoders as ce
        except ImportError:
            logger.error("category_encoders not installed; please install via 'pip install category_encoders'.")
            sys.exit(1)
        encoder = ce.BinaryEncoder(cols=categorical_cols)
        df = encoder.fit_transform(df)
    return df

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

def reduce_dimensions_vae(df, n_components=100, columns_to_reduce=None,
                          epochs=30, batch_size=32, hidden_dim=128, lr=1e-3, max_input_features=1000):
    if columns_to_reduce is None:
        return df
    
    logger.info(f"Reducing dimensions to {n_components} latent features using VAE for specified columns...")
    data = df[columns_to_reduce].values.astype(np.float32)
    if data.shape[1] > max_input_features:
        variances = np.var(data, axis=0)
        top_indices = np.argsort(variances)[-max_input_features:]
        data = data[:, top_indices]
        columns_to_reduce = [columns_to_reduce[i] for i in top_indices]
        logger.info(f"Input features reduced to top {max_input_features} by variance.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_data = torch.tensor(data).to(device)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = data.shape[1]
    model = VAE(input_dim=input_dim, latent_dim=n_components, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch_data = batch[0]
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_data)
            loss = vae_loss(recon_batch, batch_data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"VAE Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.4f}")
    
    model.eval()
    with torch.no_grad():
        tensor_all = torch.tensor(data).to(device)
        mu, _ = model.encode(tensor_all)
        latent_repr = mu.cpu().numpy()
    
    latent_df = host_pd.DataFrame(latent_repr, columns=[f"VAE_{i}" for i in range(n_components)],
                             index=df.index)
    df = df.drop(columns=columns_to_reduce)
    df = host_pd.concat([df, latent_df], axis=1)
    return df

def impute_and_process_individual(df, autoencoder_flag=False, numeric_cols=None, categorical_cols=None, use_knn=False, scale_numeric=True):
    logger.info("Processing individual DataFrame: imputation, normalization, and encoding.")
    df = final_cleaning(df)
    df = smart_impute(df, strategy='iterative', use_knn=use_knn)
    if autoencoder_flag:
        df = autoencoder_impute(df)
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in ['SAMPLE_ID']]
    if scale_numeric and numeric_cols:
        df = normalize_features(df, numeric_cols)
    df = encode_categorical(df, categorical_cols, encoding_strategy='onehot')
    return df

def process_and_export_file(file_key, file_path, id_cols, output_folder, autoencoder_flag=False, use_knn=False):
    logger.info(f"Processing {file_key} from {file_path}...")
    if file_key in ['clinical_patient', 'clinical_sample']:
        df = read_file(file_path)
    elif file_key in ['mrna_raw', 'mrna_zscore', 'cna', 'methylation']:
        df = read_wide_file_and_transpose(file_path, id_cols=id_cols)
    elif file_key == 'mutations':
        df = aggregate_mutations(file_path)
    else:
        df = read_file(file_path)
    df = impute_and_process_individual(df, autoencoder_flag=autoencoder_flag, use_knn=use_knn)
    return df

def merge_processed_datasets(processed_dfs):
    merged = processed_dfs.get('clinical')
    keys = ['mrna_raw', 'mrna_zscore', 'cna', 'methylation', 'gene_panel', 'mutations']
    for key in keys:
        if processed_dfs.get(key) is not None:
            logger.info(f"Merging {key} into the combined dataset...")
            merged = merged.merge(processed_dfs[key], on='SAMPLE_ID', how='inner')
    logger.info(f"Combined dataset contains {merged.shape[0]} records and {merged.shape[1]} columns.")
    return merged

def prompt_user_for_selection():
    print("\n*** METABRIC Data Integration ***")
    print("Select the data types to include:")
    selections = {}
    all_choice = input("Include all files? (y/n): ").strip().lower()
    if all_choice == 'y':
        selections = {
            'clinical': True,
            'mrna_raw': True,
            'mrna_zscore': True,
            'cna': True,
            'methylation': True,
            'gene_panel': True,
            'mutations': True
        }
    else:
        selections['clinical'] = (input("Include Clinical Data? (y/n): ").strip().lower() == 'y')
        selections['mrna_raw'] = (input("Include Raw Gene Expression Data? (y/n): ").strip().lower() == 'y')
        selections['mrna_zscore'] = (input("Include Gene Expression Z-scores? (y/n): ").strip().lower() == 'y')
        selections['cna'] = (input("Include CNA Data? (y/n): ").strip().lower() == 'y')
        selections['methylation'] = (input("Include Methylation Data? (y/n): ").strip().lower() == 'y')
        selections['gene_panel'] = (input("Include Gene Panel Matrix Data? (y/n): ").strip().lower() == 'y')
        selections['mutations'] = (input("Include Mutation Data? (y/n): ").strip().lower() == 'y')
    return selections

def main():
    parser = argparse.ArgumentParser(description="Reworked METABRIC Data Integration Script")
    parser.add_argument("--use-cudf", action="store_true", help="Use cuDF as the DataFrame backend")
    parser.add_argument("--autoencoder", action="store_true", help="Use autoencoder-based imputation")
    parser.add_argument("--knn", action="store_true", help="Use KNNImputer instead of IterativeImputer")
    args = parser.parse_args()
    global pd
    pd = import_backend(args.use_cudf)
    
    output_folder = "compiled"
    os.makedirs(output_folder, exist_ok=True)
    
    logger.info("Starting reworked METABRIC data integration script...")
    selections = prompt_user_for_selection()
    processed_dfs = {}
    
    # Process clinical data
    if selections.get('clinical', False):
        logger.info("Processing clinical data...")
        patient_df = read_file('data_clinical_patient.txt')
        sample_df = read_file('data_clinical_sample.txt')
        clinical_df = pd.merge(sample_df, patient_df, on='PATIENT_ID', how='inner')
        if 'PATIENT_ID' in clinical_df.columns:
            clinical_df = clinical_df.drop(columns=['PATIENT_ID'])
        processed_dfs['clinical'] = impute_and_process_individual(
            final_cleaning(clinical_df),
            autoencoder_flag=args.autoencoder,
            use_knn=args.knn,
            scale_numeric=False  # Preserve clinical values
        )
    else:
        logger.error("Clinical data is required. Exiting.")
        sys.exit(1)
    
    file_map = {
        'mrna_raw': ('data_mrna_illumina_microarray.txt', [0]),
        'mrna_zscore': ('data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt', [0]),
        'cna': ('data_cna.txt', [0, 1]),
        'methylation': ('data_methylation_promoters_rrbs.txt', [0]),
        'gene_panel': ('data_gene_panel_matrix.txt', None),
        'mutations': ('data_mutations.txt', None)
    }
    
    for key, (filepath, id_cols) in file_map.items():
        if selections.get(key, False):
            if key in ['mrna_raw', 'mrna_zscore', 'cna', 'methylation']:
                df = read_wide_file_and_transpose(filepath, id_cols=id_cols)
            elif key == 'mutations':
                df = aggregate_mutations(filepath)
            else:
                df = read_file(filepath)
            df = final_cleaning(df)
            processed_dfs[key] = impute_and_process_individual(df, autoencoder_flag=args.autoencoder, use_knn=args.knn)
        else:
            processed_dfs[key] = None
    
    merged_df = merge_processed_datasets(processed_dfs)
    
    # Exclude clinical numeric columns from VAE reduction.
    clinical_numeric = set(col.upper() for col in ['GRADE', 'TUMOR_SIZE', 'TUMOR_STAGE', 
                                                   'TMB_NONSYNONYMOUS', 'LYMPH_NODES_EXAMINED_POSITIVE', 
                                                   'NPI', 'COHORT', 'AGE_AT_DIAGNOSIS', 'RFS_MONTHS'])
    all_numeric = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    omic_numeric = [col for col in all_numeric if col.upper() not in clinical_numeric]
    
    if omic_numeric:
        merged_df = reduce_dimensions_vae(merged_df, n_components=50, columns_to_reduce=omic_numeric,
                                           epochs=30, batch_size=32, hidden_dim=128, lr=1e-3, max_input_features=1000)
    
    # (Optional) Apply SHAP-based feature selection on merged data if desired.
    target_col = 'Overall Survival Status'
    if target_col in merged_df.columns:
        y = merged_df[target_col].apply(lambda x: 1 if str(x).strip().lower() in ['1', 'deceased', 'died'] else 0)
        X = merged_df.drop(columns=[target_col])
        selected_features = shap_feature_selection(X, y, num_features=100)
        merged_df = merged_df[selected_features + [target_col]]
    
    # Filter clinical data to only include samples present in the integrated dataset.
    processed_dfs['clinical'] = processed_dfs['clinical'][processed_dfs['clinical']['SAMPLE_ID'].isin(merged_df['SAMPLE_ID'])]
    
    # Save all processed files at the end.
    for key, df in processed_dfs.items():
        if df is not None:
            out_file = os.path.join(output_folder, f"{key}_processed.csv")
            df.to_csv(out_file, index=False)
            logger.info(f"Exported processed {key} file to {out_file}.")
    
    # Save the integrated dataset.
    merged_out = os.path.join(output_folder, "integrated_metabric_compiled.csv")
    merged_df.to_csv(merged_out, index=False)
    logger.info(f"Exported combined integrated CSV file to {merged_out}.")

if __name__ == '__main__':
    main()
