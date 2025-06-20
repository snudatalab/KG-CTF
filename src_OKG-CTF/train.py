import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.linalg import solve_sylvester
from tqdm import tqdm
from collections import defaultdict

def khatri_rao(A, B):
    """
    Compute the Khatri-Rao product of matrices A and B.
    """
    r = A.shape[1]
    result = np.zeros((A.shape[0] * B.shape[0], r))
    for i in range(r):
        result[:, i] = np.kron(A[:, i], B[:, i])
    return result

def calculate_rmse(predictions, targets):
    """
    Calculate the RMSE between predictions and targets.
    """
    return np.sqrt(((predictions - targets) ** 2).mean())

def predict(stock_tensors, factors_stock):
    """
    Generate predictions based on the current factors.
    """
    predictions = {}
    for symbol, tensor in stock_tensors.items():
        idx = list(stock_tensors.keys()).index(symbol)
        U_k = factors_stock[0][idx]
        S_k = factors_stock[1][idx]
        V = factors_stock[2]
        predictions[symbol] = U_k @ S_k @ V.T
    return predictions

def evaluate_rmse(stock_true, predicted_tensors, missing_idxs):
    """
    Evaluate the RMSE for missing indices.
    """
    rmse_values = []
    for symbol, indices in missing_idxs.items():
        true_tensor = stock_true[symbol]
        pred_tensor = predicted_tensors[symbol]
        for row, col in indices:
            if row < true_tensor.shape[0] and col < true_tensor.shape[1]:
                true_value = true_tensor[row, col]
                pred_value = pred_tensor[row, col]
                rmse_values.append((true_value, pred_value))

    true_values = np.array([x[0] for x in rmse_values])
    pred_values = np.array([x[1] for x in rmse_values])
    return calculate_rmse(pred_values, true_values)


def evaluate_local_rmse(stock_true, predicted_tensors, missing_idxs, N_initial, N_stream, block_idx, not_streaming_symbols):
    rmse_values = []
    for symbol, indices in missing_idxs.items():
        if symbol in not_streaming_symbols:
            continue

        true_tensor = stock_true[symbol]
        pred_tensor = predicted_tensors[symbol]

        # 초기 블록과 스트리밍 블록의 시작 및 끝 인덱스 계산
        start_idx = N_initial[symbol]
        end_idx = start_idx + ((block_idx + 1) * N_stream)

        # 슬라이싱된 블록 데이터
        block_true_tensor = true_tensor[start_idx:end_idx]
        block_pred_tensor = pred_tensor[start_idx:end_idx]

        # missing_idxs에서 유효한 row, col 값만 처리
        for row, col in indices:
            if row < block_true_tensor.shape[0] and col < block_true_tensor.shape[1]:
                true_value = block_true_tensor[row, col]
                pred_value = block_pred_tensor[row, col]
                rmse_values.append((true_value, pred_value))

    # 유효한 값들만 RMSE 계산
    if rmse_values:
        true_values = np.array([x[0] for x in rmse_values])
        pred_values = np.array([x[1] for x in rmse_values])
        return calculate_rmse(pred_values, true_values)
    else:
        # 유효한 값이 없으면 RMSE를 0 또는 NaN으로 반환
        return np.nan



def update_U_k(T_k, V, S_k, Q_k, H, lambda_u, U_k_pre, beta_k, rank):
    """
    Update U_k using ALS logic.
    """

    #print("T_k:", T_k.shape)
    #print("S_k:", S_k.shape)
    #print("Q_k:", Q_k.shape)
    #print("V:", V.shape)

    U_k = (T_k @ V @ S_k + lambda_u * Q_k @ H) @ np.linalg.pinv(S_k @ V.T @ V @ S_k + lambda_u * np.eye(rank))
    #print("U_k:", U_k.shape)
    U_k += beta_k * (U_k - U_k_pre)
    return U_k


# def update_S_k(T_k_old, T_k, U_k_old, U_k, V, G_k, M_k, R, D_k, lambda_l, lambda_r, lambda_o, lambda_k, S_k_pre, beta_k):
#     """
#     Update S_k using vectorized computation.
#     """
#     T_k_old_vec = T_k_old.T.reshape(1, -1)  # vec(T_k_old)
#     T_k_new_vec = T_k.T.reshape(1, -1) # vec(T_k)
#     G_k_vec = G_k.T.reshape(1, -1)

#     #print("T_k_old:", T_k_old.shape)
#     #print("T_k:", T_k.shape)
#     #print("U_k_old:", U_k_old.shape)
#     #print("V:", V.shape)


#     # Numerator
#     numerator = (
#         lambda_o *(T_k_old_vec @ khatri_rao(V, U_k_old)) +
#         T_k_new_vec @ khatri_rao(V, U_k) +
#         lambda_k * (G_k_vec @ khatri_rao(R, M_k)) +
#         lambda_r * (np.ones((1, G_k.shape[1])) @ (R + np.diag(1 / (np.diag(D_k) + 1e-8)) @ G_k.T @ M_k))
#     )

#     # Denominator
#     denominator = (
#         lambda_o*((V.T @ V) * (U_k_old.T @ U_k_old)) +
#         (V.T @ V) * (U_k.T @ U_k) +
#         lambda_k * ((R.T @ R) * (M_k.T @ M_k)) +
#         lambda_l * np.eye(U_k.shape[1]) +
#         lambda_r * R.shape[0] * np.eye(U_k.shape[1])
#     )

#     S_k = numerator @ np.linalg.pinv(denominator)
#     S_k = np.diag(S_k[0])  # Ensure diagonal S_k
#     S_k += beta_k * (S_k - S_k_pre)
#     return S_k

#Function to update S_k
def update_S_k(T_k_old, T_k, U_k_old, U_k, V, G_k, M_k, R, D_k, lambda_l, lambda_r, lambda_o, lambda_k, S_k_pre, beta_k):
    D_k_inv = np.diag(1 / (np.diag(D_k) + 1e-8 ))
    S_k_num = lambda_o *((U_k_old.T @ T_k_old @ V))+ (U_k.T @ T_k @ V) + lambda_k *((M_k.T @ G_k @ R)) + lambda_r * (np.ones((V.shape[1], G_k.shape[1])) @ R) + lambda_r*(np.ones((V.shape[1], G_k.shape[1])) @ D_k_inv @ G_k.T @ M_k)

    S_k_den = (lambda_o*((U_k_old.T @ U_k_old) * (V.T @ V)) + (U_k.T @ U_k) * (V.T @ V)) + (lambda_k * (M_k.T @ M_k) * (R.T @ R)) + lambda_l * np.eye(U_k.shape[1]) + lambda_r * np.eye(U_k.shape[1]) * G_k.shape[1]

    S_k = np.diag(np.diag(S_k_num) @ np.linalg.pinv(S_k_den))
    S_k += beta_k * (S_k - S_k_pre)
    return S_k

def update_V(stock_old_tensors, factors_old_stock, stock_tensors, factors_stock, lambda_l, lambda_o, V_pre, beta_k, rank):
    """
    Update V based on current and old tensors.
    """
    V_numerator = np.zeros((factors_stock[2].shape[0], factors_stock[2].shape[1]))
    V_denominator = np.zeros((factors_stock[2].shape[1], factors_stock[2].shape[1]))
    
    #for idx, T_k in enumerate(stock_tensors.values()):
    for idx, (symbol, T_k) in enumerate(stock_tensors.items()):
        U_k = factors_stock[0][idx]
        S_k = factors_stock[1][idx]
        #U_k_old = factors_old_stock[0][idx]
        U_k_old = factors_old_stock[symbol]["U_old"]
        T_k_old = stock_old_tensors.get(list(stock_tensors.keys())[idx])
        
        #print("T_k_old:", T_k_old.shape)
        #print("T_k:", T_k.shape)
        #print("U_k_old:", U_k_old.shape)
        #print("S_k:", S_k.shape)
        
        V_numerator += lambda_o*(T_k_old.T @ U_k_old @ S_k) + T_k.T @ U_k @ S_k
        V_denominator += lambda_o*(S_k @ U_k_old.T @ U_k_old @ S_k) + S_k @ U_k.T @ U_k @ S_k

    V = V_numerator @ np.linalg.pinv(V_denominator + lambda_l * np.eye(rank))
    V += beta_k * (V - V_pre)
    return V



# Function to update M_k
def update_M_k(G_k, M_k_pre, S_k, R, D_k, lambda_r, lambda_l, lambda_k, beta_k, rank):
    D_k_inv = np.diag(1 / (np.diag(D_k) + 1e-8 ))
    A = lambda_r * ((G_k @ D_k_inv.T @ D_k_inv @ G_k.T)) + lambda_l * np.eye(M_k_pre.shape[0])
    B = lambda_k * (S_k @ R.T @ R @ S_k)
    C = lambda_k * (G_k @ R @ S_k) + (lambda_r *  G_k @ D_k_inv @ np.ones((G_k.shape[1], rank)) @ S_k) - (lambda_r *  G_k @ D_k_inv @ R)
 
    M_k = solve_sylvester(A, B, C)
    M_k += beta_k * (M_k - M_k_pre)
    return M_k

# Function to update R
def update_R(kg_tensors, factors_kg, factors_stock, lambda_l, lambda_r, lambda_k, beta_k, rank, symbol_to_id, common_symbols_list, stock_tensors, R_pre):
    R_numerator = np.zeros((factors_kg[2].shape[0], factors_kg[2].shape[1]))
    R_denominator = np.zeros((factors_kg[2].shape[1], factors_kg[2].shape[1]))
    
    for idx, (symbol, tensor) in enumerate(stock_tensors.items()):
        symbol_id = symbol_to_id[symbol]
        kg_idx = list(kg_tensors.keys()).index(symbol_id)
        G_k = kg_tensors[symbol_id]
        M_k = factors_kg[0][kg_idx]
        S_k = factors_stock[1][idx]  # Use stock index `idx` for S_k
        D_k = factors_kg[4][kg_idx]
        D_k_inv = np.diag(1 / (np.diag(D_k) + 1e-8))
        
        # Update numerator and denominator
        R_numerator += lambda_k * (G_k.T @ M_k @ S_k) + lambda_r * (
            (np.ones((G_k.shape[1], rank))) @ S_k - D_k_inv @ G_k.T @ M_k
        )
        R_denominator += lambda_k * (S_k @ M_k.T @ M_k @ S_k)
    
    # Compute the updated R matrix
    R = R_numerator @ np.linalg.inv(R_denominator + (lambda_l + lambda_r * len(stock_tensors)) * np.eye(rank))
    R += beta_k * (R - R_pre)
    return R




def update_H(U_dict, U_update_old, Q_update, Q_update_old, lambda_o):
    """
    Update H factor matrix based on Q_update and U_dict.
    """
    K = len(U_dict)  # Number of companies
    H_shape = (U_dict[0].shape[1], U_dict[0].shape[1])  # (rank, rank)
    H_update = np.zeros(H_shape)

    for idx, Q_k in Q_update:
        Q_k_old = Q_update_old[idx]
        U_k_old = U_update_old[idx]
        U_k = U_dict[idx]

        #print("U_k:", U_k.shape)
        #print("Q_k:", Q_k.shape)
        #print("Q_k_old:", Q_k_old.shape)
        #print("U_k_old:", U_k_old.shape)

        H_update += Q_k.T @ U_k + lambda_o*(Q_k_old.T @ U_k_old)

    H_update /= K
    return H_update

def update_Q(U_dict, H):
    """
    Update Q_k for each symbol (U_k).
    """
    Q_update = []
    for idx, U_k in U_dict.items():
        Z, _, P = np.linalg.svd(U_k @ H.T, full_matrices=False)
        Q_update.append((idx, Z @ P))
    return Q_update

def als_update(factors_old_stock, factors_stock, factors_kg, kg_data, stock_old_tensors, stock_tensors, kg_tensors, 
               entity_embeddings, common_symbols_list, stock_only_symbols_list, stock_train, stock_true, 
               common_symbols, symbol_to_id, missing_idxs, lambdas, beta_k, rank):
    """
    Perform ALS update for the current block.
    """
    lambda_u, lambda_r, lambda_l, lambda_o, lambda_k = lambdas
    total_rmse = 0

    U_dict = {}
    M_dict = {}
    #Q_update_old = factors_old_stock[3]
    #U_update_old = factors_old_stock[0]

    Q_update_old = [factors_old_stock[symbol]["Q_old"] for symbol in stock_tensors.keys()]
    U_update_old = [factors_old_stock[symbol]["U_old"] for symbol in stock_tensors.keys()]


    # Update factors for each stock tensor
    for idx, symbol in enumerate(stock_tensors.keys()):
        T_k = stock_tensors[symbol]
        #print("!!T_k:", T_k.shape)
        T_k_old = stock_old_tensors[symbol]
        #print("!!T_k_old:", T_k_old.shape)

        symbol_id = symbol_to_id[symbol] if symbol in common_symbols_list else None
        G_k = kg_tensors[symbol_id]
        
        kg_idx = list(kg_tensors.keys()).index(symbol_id)
        #복구할려면 M_k 다시 주석 풀기
        M_k = factors_kg[0][kg_idx]
        # #추가#################################
        # unique_entities = sorted(set(kg_data[kg_data['tail'] == symbol_id]['head'].unique()))
        # M_k = np.array([entity_embeddings[entity] for entity in unique_entities])
        # factors_kg[0][kg_idx] = M_k
        # #추가#################################


        R = factors_kg[2]
        D_k = factors_kg[4][kg_idx]


        U_k_pre = factors_stock[0][idx]
        #print("!!U_k_pre:", U_k_pre.shape)
        V = factors_stock[2]
        #print("!!V:", V.shape)
        Q_k = factors_stock[3][idx]
        #print("!!Q_k:", Q_k.shape)
        H = factors_stock[4]
        #print("!!H:", H.shape)
        S_k = factors_stock[1][idx]

        U_k = update_U_k(T_k, V, S_k, Q_k, H, lambda_u, U_k_pre, beta_k, rank)
        factors_stock[0][idx] = U_k

        #U_k_old = factors_old_stock[0][idx]
        U_k_old = factors_old_stock[symbol]["U_old"]
        U_dict[idx] = U_k

        S_k_pre = factors_stock[1][idx]  # Make a writable copy of S_k
        #print("!!S_k_pre:", S_k_pre.shape)

        # Update S_k
        S_k = update_S_k(
            T_k_old, T_k, U_k_old, U_k, V, G_k, M_k, R, D_k, lambda_l, lambda_r, lambda_o, lambda_k, S_k_pre, beta_k
        )

        factors_stock[1][idx] = S_k

        # Update M_k
        #M_k = factors_kg[0][kg_idx]
        R = factors_kg[2]
        D_k = factors_kg[4][kg_idx]
        M_k_pre = factors_kg[0][kg_idx]
        M_k = update_M_k(G_k, M_k_pre, S_k, R, D_k, lambda_r, lambda_l, lambda_k, beta_k, rank)
        factors_kg[0][kg_idx] = M_k
        M_dict[kg_idx] = M_k

        # 엔티티 임베딩 업데이트
        #추가#################################
        #unique_entities = sorted(set(kg_data[kg_data['tail'] == symbol_id]['head'].unique()))
        # entity_idx = {entity: idx for idx, entity in enumerate(unique_entities)}
        # for entity, i in entity_idx.items():
        #     entity_embeddings[entity] = M_k[i]
        #추가#################################

    # Update V
    V_pre = factors_stock[2]
    V = update_V(stock_old_tensors, factors_old_stock, stock_tensors, factors_stock, lambda_l, lambda_o, V_pre, beta_k, rank)
    factors_stock[2] = V

    # Update R
    R_pre = factors_kg[2]
    factors_kg[2] = update_R(kg_tensors, factors_kg, factors_stock, lambda_l, lambda_r, lambda_k, beta_k, rank, symbol_to_id, common_symbols_list, stock_tensors, R_pre)



    # Update Q and H
    Q_update = update_Q(U_dict, factors_stock[4])
    factors_stock[3] = [q[1] for q in Q_update]
    factors_stock[4] = update_H(U_dict, U_update_old, Q_update, Q_update_old, lambda_o)


    # Predict and evaluate RMSE
    predicted_tensors = predict(stock_tensors, factors_stock)
    #print("predicted_tensors:", predicted_tensors["A"].shape)


    #rmse = evaluate_rmse(stock_true, predicted_tensors, missing_idxs)
    #total_rmse += rmse

    #return total_rmse
    return predicted_tensors

