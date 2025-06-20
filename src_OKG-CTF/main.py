import numpy as np
import pandas as pd
from load_data import load_cached
from train import als_update
from train import calculate_rmse, evaluate_rmse, evaluate_local_rmse
from load_data import initialize_factors
from tqdm import tqdm
from collections import OrderedDict
import time
import random
import os

if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 5  # Number of epochs
    lambdas = (10, 0.1, 0.1, 0.01, 0.05)
    rank = 5

    # Load cached data
    (
        factors_stock, factors_kg, kg_data, all_stock_tensors, stock_tensors, kg_tensors, entity_embeddings, 
        common_symbols_list, stock_only_symbols_list, stock_train, stock_true, common_symbols, symbol_to_id, missing_idxs,
        factors_old_stock, stock_old_tensors, streaming_data
    ) = load_cached()
    
    #print("stock_true", stock_true)

    """
    #모든 슬라이스 스트리밍하는 기존코드 시작
    # Create initial blocks and streaming blocks dynamically for each symbol
    streaming_blocks = {}
    N_initial = {}

    for sym, tensor in all_stock_tensors.items():
        if tensor.shape[0] > 0:
            total_rows = tensor.shape[0]
            N_initial[sym] = max(1, int(0.4 * total_rows))  # 40% of the total rows
            N_stream = 1000  # Define block size

            # Split initial block and streaming blocks
            initial_block = tensor[:N_initial[sym]]
            remaining_blocks = [
                tensor[start:start + N_stream] for start in range(N_initial[sym], total_rows, N_stream)
            ]

            streaming_blocks[sym] = {
                "initial": initial_block,
                "streaming": remaining_blocks
            }

    # Display block counts for each symbol
    for sym, blocks in streaming_blocks.items():
        print(f"Symbol: {sym}, Initial block size: {blocks['initial'].shape[0]}, Number of streaming blocks: {len(blocks['streaming'])}")
    #모든 슬라이스 스트리밍하는 기존코드 끝
    """

    ##################################################################################################
    # 랜덤하게 30% 심볼 선택
    # 슬라이스 일부만 스트리밍하는 수정코드 시작
    total_symbols = list(all_stock_tensors.keys())
    random.seed(42)  # 재현성을 위한 시드 고정
    not_streaming_symbols = random.sample(total_symbols, int(len(total_symbols) * 0.3))

    streaming_blocks = {}
    N_initial = {}

    for sym, tensor in all_stock_tensors.items():
        if sym not in not_streaming_symbols and tensor.shape[0] > 0:
            total_rows = tensor.shape[0]
            N_initial[sym] = max(1, int(0.4 * total_rows))  # 40% of the total rows
            N_stream = 20  # Define block size

            # Split initial block and streaming blocks
            initial_block = tensor[:N_initial[sym]]
            remaining_blocks = [
                tensor[start:start + N_stream] for start in range(N_initial[sym], total_rows, N_stream)
            ]

            streaming_blocks[sym] = {
                "initial": initial_block,
                "streaming": remaining_blocks
            }

        elif tensor.shape[0] > 0:
            # 스트리밍 대상이 아닌 심볼은 전체 데이터를 하나의 초기 블록으로 처리
            streaming_blocks[sym] = {
                "initial": tensor,
                "streaming": []
            }
        
    # Display block counts for each symbol with streaming status
    for sym, blocks in streaming_blocks.items():
        is_streaming = "Yes" if sym in not_streaming_symbols else "No"
        print(f"Symbol: {sym}, Streaming: {is_streaming}, Initial block size: {blocks['initial'].shape[0]}, Number of streaming blocks: {len(blocks['streaming'])}")

    #슬라이스 일부만 스트리밍하는 수정코드 끝
    ##################################################################################################

    total_rmse = 0
    cumulative_predictions = {symbol: [] for symbol in all_stock_tensors.keys()}

    # 초기 블록 학습 시작
    for sym, blocks in streaming_blocks.items():
        
        #print("stock_true", stock_true)
        
        
        #cumulative_predictions[sym] = []
        print(f"\nProcessing symbol: {sym}, Total streaming blocks: {len(blocks['streaming'])}")

        # Process the initial block
        initial_block = blocks['initial']
        print(f"Processing initial block for symbol {sym}")

        stock_tensors = OrderedDict({sym: initial_block})
        block_shapes = [tensor.shape for tensor in stock_tensors.values()]
        kg_tensor_shapes = [tensor.shape for tensor in kg_tensors.values()]
        
        ##################################################################################################
        #추가 시작
        factors_stock = initialize_factors(len(stock_tensors), rank, block_shapes, kg_tensor_shapes, kg_tensors)[0]
        #추가 끝
        ##################################################################################################
        
        # Epoch loop for the initial block
        for epoch in tqdm(range(num_epochs), desc=f"Training Initial Block for {sym}"):
            beta_k = 0.3 if 5 <= epoch < 15 else 0.5
            #beta_k = 0

            predicted_tensors = als_update(
                factors_old_stock, factors_stock, factors_kg, kg_data, stock_old_tensors,
                stock_tensors, kg_tensors, entity_embeddings, common_symbols_list,
                stock_only_symbols_list, stock_train, stock_true, common_symbols,
                symbol_to_id, missing_idxs, lambdas, beta_k, rank
            )
            #print(f"Epoch {epoch + 1}/{num_epochs}, Total RMSE: {total_rmse:.4f}")

        print("len(initial_stock_tensors):", len(stock_tensors))
        cumulative_predictions[sym].append(predicted_tensors[sym])
        stock_old_tensors[sym] = initial_block

        # Update `factors_old_stock` with the initial block factors
        idx = list(stock_tensors.keys()).index(sym)

        #factors_old_stock[sym]["U_old"] = factors_stock[0][idx]
        #factors_old_stock[sym]["Q_old"] = factors_stock[3][idx]
        
        #0123추가
        factors_old_stock[sym]["U_old"] = factors_stock[0][idx]
        factors_old_stock[sym]["S"] = factors_stock[1][idx]
        factors_old_stock[sym]["V"] = factors_stock[2]
        factors_old_stock[sym]["Q_old"] = factors_stock[3][idx]
        factors_old_stock[sym]["H"] = factors_stock[4]
        #0123추가

   ################################################
   # 스트리밍 블록 학습 시작
   # 시간 기반으로 반복
    num_streaming_blocks = max(len(blocks['streaming']) for blocks in streaming_blocks.values())  # 가장 긴 스트리밍 블록 수를 기준으로 반복
    block_times = []
    local_error = []

    output_file = '/home/duduuman/O_data_JPN/src/OKG-CTF/log.txt'
    folder_name = os.path.basename(os.path.dirname(output_file))

    # ★ 로그 파일에 전반부 메타 정보 먼저 기록
    with open(output_file, "a") as f:
        f.write("\n###############################################\n")
        f.write(f"Method Name: {folder_name}\n")
        f.write(f"N_stream: {20}\n")
        f.write(f"lambdas: {lambdas}\n")
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"Num Streaming Blocks: {num_streaming_blocks}\n")
        f.write("Block Times & Local Errors:\n")  # 아래에서 각 블록마다 한 줄씩 기록

    for block_idx in range(num_streaming_blocks):
        print(f"\nProcessing streaming block {block_idx + 1}/{num_streaming_blocks}")
        
        count = 0
        start_time = time.time()  # Start timing
        # 모든 심볼에 대해 현재 시점의 블록을 처리
        for sym, blocks in streaming_blocks.items():
            if block_idx >= len(blocks['streaming']):  # 현재 심볼에 스트리밍 데이터가 더 이상 없다면 건너뜀
                continue

            # 현재 스트리밍 블록
            new_block = blocks['streaming'][block_idx]
            print(f"Processing block {block_idx + 1} for symbol {sym}")

            # Prepare stock_tensors for the current block
            stock_tensors = OrderedDict({sym: new_block})
            new_block_shapes = [tensor.shape for tensor in stock_tensors.values()]

            # Initialize new factors for the streaming block
            new_factors = initialize_factors(len(stock_tensors), rank, new_block_shapes, kg_tensor_shapes, kg_tensors)[0]

            # Update factors_old_stock with the latest factors
            idx = list(stock_tensors.keys()).index(sym)
            
            #0123추가
            new_factors[1][idx] = factors_old_stock[sym]["S"]
            new_factors[2] = factors_old_stock[sym]["V"]
            new_factors[4] = factors_old_stock[sym]["H"]
            #0123추가
            
            # Epoch loop for the current block
            for epoch in tqdm(range(num_epochs), desc=f"Training Block {block_idx + 1} for {sym}"):
                beta_k = 0.5 if 5 <= epoch < 15 else 0.5
                #beta_k = 0

                #start_time = time.time()  # Start timing

                predicted_tensors = als_update(
                    factors_old_stock, new_factors, factors_kg, kg_data, stock_old_tensors,
                    stock_tensors, kg_tensors, entity_embeddings, common_symbols_list,
                    stock_only_symbols_list, stock_train, stock_true, common_symbols,
                    symbol_to_id, missing_idxs, lambdas, beta_k, rank
                )
                
                #end_time = time.time()  # End timing
                #print(f"Time taken for epoch {epoch + 1}: {end_time - start_time:.2f} seconds")
                
            print("len(stream_stock_tensors):", len(stock_tensors))
            cumulative_predictions[sym].append(predicted_tensors[sym])

            # Concatenate old and new factors
            factors_old_stock[sym]["U_old"] = np.vstack([factors_old_stock[sym]["U_old"], new_factors[0][idx]])
            factors_old_stock[sym]["Q_old"] = np.vstack([factors_old_stock[sym]["Q_old"], new_factors[3][idx]])
            
            #0123추가
            factors_old_stock[sym]["S"] = new_factors[1][idx]
            factors_old_stock[sym]["V"] = new_factors[2]
            factors_old_stock[sym]["H"] = new_factors[4]
            #0123추가
        
            # Update `stock_old_tensors` after processing the block
            stock_old_tensors[sym] = np.vstack([stock_old_tensors[sym], new_block])
            
            count +=1
            
        end_time = time.time()  # End timing
        block_time = end_time - start_time  # 블록 처리 시간 계산
        block_times.append(block_time)  # 블록 처리 시간을 리스트에 추가
        print(f"Time taken for block {block_idx + 1}: {block_time:.2f} seconds")

        ########로컬 에러 추가############################################################################
        local_combined_predictions = {
            symbol: np.vstack(blocks) for symbol, blocks in cumulative_predictions.items() if len(blocks) > 0
        }

        local_total_rmse = evaluate_local_rmse(stock_true, local_combined_predictions, missing_idxs, N_initial, N_stream, block_idx, not_streaming_symbols)
        print(local_total_rmse)
        local_error.append(local_total_rmse) 
        ########로컬 에러 추가###########################################################################
            
        # ★★★★★ 여기서 각 블록마다 파일에 한 줄씩 기록
        with open(output_file, "a") as f:
            f.write(f"Block {block_idx+1}: {block_time:.2f} seconds, Local RMSE: {local_total_rmse:.4f}\n")

        print(f"Time taken for block {block_idx + 1}: {block_time:.2f} seconds")
        print(f"Local RMSE for block {block_idx+1}: {local_total_rmse:.4f}")
    
    # Combine cumulative predictions after processing all streaming blocks
    combined_predictions = {
        symbol: np.vstack(blocks) for symbol, blocks in cumulative_predictions.items() if len(blocks) > 0
    }

    for symbol, blocks in combined_predictions.items():
        print(f"Final shape for symbol {symbol}: {blocks.shape}")

    #print("stock_true", stock_true.shape)
    # Evaluate RMSE
    total_rmse = evaluate_rmse(stock_true, combined_predictions, missing_idxs)
    print(f"Final Total RMSE: {total_rmse:.4f}")
    print("sym_count:", count)

    print("\nAll streaming blocks processed. Training complete.")
    
    
    # 마지막으로 최종 RMSE 등 추가 정보 로그 파일에 기록
    with open(output_file, "a") as f:
        f.write(f"\nGlobal Total RMSE: {total_rmse:.4f}\n")
        f.write("Full Block Times:\n")
        for i, time_val in enumerate(block_times):
            f.write(f"  Block {i+1}: {time_val:.2f} seconds\n")
        f.write("Full Local Errors:\n")
        for i, err_val in enumerate(local_error):
            f.write(f"  Block {i+1}: {err_val:.4f}\n")
        f.write("\n###############################################\n")

    print(f"\nAll logs saved to {os.path.abspath(output_file)}")
    
    
    
    
    
    
    
    