import numpy as np
import pandas as pd
from load_data import load_cached
from train import als_update
from tqdm import tqdm

if __name__=="__main__":
    #np.random.seed(42)
    num_epochs = 50
    lambdas = (10, 1, 1)
    rank = 12
    
    factors_stock, factors_kg, kg_data, stock_tensors, kg_tensors, entity_embeddings, common_symbols_list, stock_only_symbols_list, stock_train, stock_true, common_symbols, symbol_to_id, missing_idxs = load_cached()

    for epoch in tqdm(range(num_epochs), desc="Training"):
        if 5 <= epoch < 15:
            beta_k = 0.3 
        else:
            beta_k = 0.0 
        total_rmse = als_update(factors_stock, factors_kg, kg_data, stock_tensors, kg_tensors, entity_embeddings, common_symbols_list, stock_only_symbols_list, stock_train, stock_true, common_symbols, symbol_to_id, missing_idxs, lambdas, beta_k, rank)
        print(f"Epoch {epoch + 1}/{num_epochs}, Total RMSE: {total_rmse:.4f}")

    print("Training complete")

