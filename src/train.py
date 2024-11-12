import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.linalg import svd, solve_sylvester
from tqdm import tqdm
from scipy.sparse import csr_matrix
from tqdm.contrib.concurrent import process_map
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from multiprocessing import Pool
from joblib import Parallel, delayed

def calculate_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def predict(stock_tensors, factors_stock):
    predictions = {}
    for symbol, tensor in stock_tensors.items():
        idx = list(stock_tensors.keys()).index(symbol)
        U_k = factors_stock[0][idx]
        S_k = factors_stock[1][idx]
        V = factors_stock[2]
        predicted_tensor = U_k @ S_k @ V.T
        predictions[symbol] = predicted_tensor
    return predictions

def evaluate_rmse(stock_true, predicted_tensors, missing_idxs):
    rmse_values = []
    for symbol, indices in missing_idxs.items():
        true_tensor = stock_true[symbol]
        pred_tensor = predicted_tensors[symbol]
        for row, col in indices:
            if row < true_tensor.shape[0] and col < true_tensor.shape[1] and row < pred_tensor.shape[0] and col < pred_tensor.shape[1]:
                true_value = true_tensor[row, col]
                pred_value = pred_tensor[row, col]
                rmse_values.append((true_value, pred_value))
            else:
                print(f"Skipping index ({row}, {col}) for symbol {symbol} due to out of bounds.")
    
    true_values = np.array([x[0] for x in rmse_values])
    pred_values = np.array([x[1] for x in rmse_values])

    print("true_values:", true_values)
    print("pred_values:", pred_values)
    rmse = calculate_rmse(pred_values, true_values)
    return rmse

# Function to update U_k
def update_U_k(T_k, V, S_k, Q_k, H, lambda_u, U_k_old, beta_k, rank):
    U_k = (T_k @ V @ S_k + lambda_u * Q_k @ H) @ np.linalg.inv(S_k @ V.T @ V @ S_k + lambda_u * np.eye(rank))
    U_k += beta_k * (U_k - U_k_old)  
    return U_k


#Function to update S_k
def update_S_k(T_k, U_k, V, G_k, M_k, R, D_k, lambda_l, lambda_r, S_k_old, beta_k, rank):
     D_k_inv = np.diag(1 / (np.diag(D_k) + 1e-8 ))
     S_k_num = (U_k.T @ T_k @ V) + (M_k.T @ G_k @ R) + lambda_r * (np.ones((rank, G_k.shape[1])) @ R) + lambda_r*(np.ones((rank, G_k.shape[1])) @ D_k_inv @ G_k.T @ M_k)

     S_k_den = ((U_k.T @ U_k) * (V.T @ V)) + ((M_k.T @ M_k) * (R.T @ R)) + lambda_l * np.eye(U_k.shape[1]) + lambda_r * np.eye(U_k.shape[1]) * G_k.shape[1]

     S_k = np.diag(np.diag(S_k_num) @ np.linalg.inv(S_k_den))
     S_k += beta_k * (S_k - S_k_old)
     return S_k

# vectorizing method to update S_k
# khatri_rao product

#def khatri_rao(A, B):
#    r = A.shape[1]
#    result = np.zeros((A.shape[0] * B.shape[0], r))
#    for i in range(r):
#        result[:, i] = np.kron(A[:, i], B[:, i])
#    return result

#def update_S_k(T_k, U_k, V, G_k, M_k, R, D_k, lambda_l, lambda_r, S_k_old, beta_k, rank):
#    D_k_inv = np.linalg.inv(np.diag(np.diag(D_k) + 1e-8))  # Inverse of D_k with regularization for numerical stability
#    T_k_vector = T_k.T.reshape(1, -1)
#    G_k_vector = G_k.T.reshape(1, -1)

#    S_k_num = (T_k_vector @ khatri_rao(V, U_k)) + (G_k_vector @ khatri_rao(R, M_k)) + lambda_r * (np.ones((1, G_k.shape[1])) @ (R + D_k_inv @ G_k.T @ M_k))
#    S_k_den = ((V.T @ V) * (U_k.T @ U_k)  + (R.T @ R) * (M_k.T @ M_k)) + lambda_l * np.eye(U_k.shape[1]) + lambda_r * R.shape[0] * np.eye(U_k.shape[1])

#    S_k = S_k_num @ np.linalg.inv(S_k_den)
#    S_k = np.diag(S_k[0])
#    S_k += beta_k * (S_k - S_k_old)
    
#    return S_k


# Function to update V
def update_V(stock_tensors, factors_stock, lambda_l, V_old, beta_k, rank):
    V_numerator = np.zeros((factors_stock[2].shape[0], factors_stock[2].shape[1]))
    V_denominator = np.zeros((factors_stock[2].shape[1], factors_stock[2].shape[1]))
    for idx, T_k in enumerate(stock_tensors.values()):
        U_k = factors_stock[0][idx]
        S_k = factors_stock[1][idx]
        V_numerator += T_k.T @ U_k @ S_k
        V_denominator += S_k @ U_k.T @ U_k @ S_k
    V = V_numerator @ np.linalg.inv(V_denominator + lambda_l * np.eye(rank))
    V += beta_k * (V - V_old) 
    return V
    
# Function to update M_k
def update_M_k(G_k, M_k, S_k, R, D_k, lambda_r, lambda_l, M_k_old, beta_k, rank):
    D_k_inv = np.diag(1 / (np.diag(D_k) + 1e-8 ))
    A = lambda_r * ((G_k @ D_k_inv.T @ D_k_inv @ G_k.T)) + lambda_l * np.eye(M_k.shape[0])
    B = S_k @ R.T @ R @ S_k
    C = G_k @ R @ S_k + (lambda_r *  G_k @ D_k_inv @ np.ones((G_k.shape[1], rank)) @ S_k) - (lambda_r *  G_k @ D_k_inv @ R)
 
    M_k = solve_sylvester(A, B, C)
    return M_k

# Function to update R
def update_R(kg_tensors, factors_kg, factors_stock, lambda_l, lambda_r, R_old, beta_k, num_companies, rank):
    R_numerator = np.zeros((factors_kg[2].shape[0], factors_kg[2].shape[1]))
    R_denominator = np.zeros((factors_kg[2].shape[1], factors_kg[2].shape[1]))
    for idx, G_k in enumerate(kg_tensors.values()):
        M_k = factors_kg[0][idx]
        S_k = factors_stock[1][idx]
        D_k = factors_kg[4][idx]
        D_k_inv = np.diag(1 / (np.diag(D_k) + 1e-8 ))
        R_numerator += G_k.T @ M_k @ S_k + lambda_r * ((np.ones((G_k.shape[1], rank))) @ S_k - D_k_inv @ G_k.T @  M_k)  
        R_denominator +=  S_k @ M_k.T @ M_k @ S_k
    R = R_numerator @ np.linalg.inv(R_denominator + (lambda_l + lambda_r * num_companies) * np.eye(rank))
    return R

# Function to update QH
def update_QH(U_dict, H):
    K = len(U_dict)
    R = H.shape[0]
    Q_update = []
    H_tmp = np.zeros((K, R, R))
    H_update = np.zeros((R, R))

    for idx, U_k in U_dict.items():
        Z, _, P = np.linalg.svd(U_k @ H.T, full_matrices=False)
        Q_update.append((idx, Z @ P))
        H_update += Q_update[-1][1].T @ U_k

    H_update /= K
    return Q_update, H_update

# ALS update method for each epoch
def als_update(factors_stock, factors_kg, kg_data, stock_tensors, kg_tensors, entity_embeddings, common_symbols_list, stock_only_symbols_list, stock_train, stock_true, common_symbols, symbol_to_id, missing_idxs, lambdas, beta_k, rank):
    """ 
    Perform an ALS update on the stock and knowledge graph tensors for a single epoch.
    """ 
    lambda_u, lambda_r, lambda_l = lambdas

    total_rmse = 0

    U_dict = {}
    M_dict = {}

    for idx, symbol in enumerate(stock_tensors.keys()):
        T_k = stock_tensors[symbol]
        V = factors_stock[2]
        Q_k = factors_stock[3][idx]
        H = factors_stock[4]

        symbol_id = symbol_to_id[symbol] if symbol in common_symbols_list else None
        G_k = kg_tensors[symbol_id] if symbol_id else None

        # Update U_k
        S_k = factors_stock[1][idx]
        U_k_old = factors_stock[0][idx]
        U_k = update_U_k(T_k, V, S_k, Q_k, H, lambda_u, U_k_old, beta_k, rank)
        factors_stock[0][idx] = U_k
        U_dict[idx] = U_k

        # Update S_k
        S_k_old = S_k
        if symbol_id in kg_tensors:
            kg_idx = list(kg_tensors.keys()).index(symbol_id)
            M_k = factors_kg[0][kg_idx]
            R = factors_kg[2]
            D_k = factors_kg[4][kg_idx]
            S_k = update_S_k(T_k, U_k, V, G_k, M_k, R, D_k, lambda_l, lambda_r, S_k_old, beta_k, rank)
            factors_stock[1][idx] = S_k

        # Update M_k using Sylvester equation
        if symbol_id in kg_tensors:
            M_k_old = M_k
            M_k = factors_kg[0][kg_idx]
            R = factors_kg[2]
            D_k = factors_kg[4][kg_idx]
            M_k = update_M_k(G_k, M_k, S_k, R, D_k, lambda_r, lambda_l, M_k_old, beta_k, rank)
            factors_kg[0][kg_idx] = M_k
            M_dict[kg_idx] = M_k
            
            # entity embedding upade 
            unique_entities = sorted(set(kg_data[kg_data['tail'] == symbol_id]['head'].unique()))
            entity_idx = {entity: idx for idx, entity in enumerate(unique_entities)}
            for entity, i in entity_idx.items():
                entity_embeddings[entity] = M_k[i]

    # Update V
    V_old = factors_stock[2]
    factors_stock[2] = update_V(stock_tensors, factors_stock, lambda_l, V_old, beta_k, rank)

    # Update R
    R_old = factors_kg[2]
    factors_kg[2] = update_R(kg_tensors, factors_kg, factors_stock, lambda_l, lambda_r, R_old, beta_k, len(stock_tensors), rank)

    # Update Q_k and H
    Q_update, H_update = update_QH(U_dict, factors_stock[4])
    for idx, Q_k in Q_update:
        factors_stock[3][idx] = Q_k
        factors_stock[4] = H_update

    # Predict and calculate RMSE for the current update
    predicted_tensors = predict(stock_tensors, factors_stock)

    rmse = evaluate_rmse(stock_true, predicted_tensors, missing_idxs)
    total_rmse += rmse

    return total_rmse

