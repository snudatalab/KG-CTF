import numpy as np
import pandas as pd
import pickle
import os
from collections import OrderedDict
from tqdm import tqdm
from joblib import Parallel, delayed


# StockTensorManager to handle stock tensors
def create_single_stock_tensor(args):
    """ 
    Create a single stock tensor for a given stock symbol.
    
    Args:
        args (tuple): A tuple containing a stock symbol (str), the stock data grouped by the symbol (pd.DataFrame), 
                      and the features to include in the tensor (list).
        
    Returns:
        tuple: The stock symbol and the corresponding tensor as a numpy array.
    """
    symbol, group, features = args
    unique_dates = sorted(group['Date'].unique())
    tensor = np.zeros((len(unique_dates), len(features)))
    date_idx = {date: idx for idx, date in enumerate(unique_dates)}
    group = group.set_index('Date')
    for date, row in group.iterrows():
        tensor[date_idx[date], :] = row[features].values
    return symbol, tensor

def create_stock_tensor_parallel(stock_data, common_symbols):
    """ 
    Create stock tensors in parallel for multiple stock symbols.
    
    Args:
        stock_data (pd.DataFrame): DataFrame containing stock data.
        common_symbols (list): List of stock symbols that are common across data.
        
    Returns:
        tuple: A dictionary of stock tensors and a list of common stock indices.
    """
    #features = [col for col in stock_data.columns if col not in ['ID','Date', 'volatility_kchi', 'volatility_kcli', 'trend_psar_down', 'trend_psar_down_indicator']]
    features = ['Low', 'Open', 'Volume', 'High', 'Close']
    all_symbols = sorted(stock_data['ID'].unique())
    common_stock_indices = [all_symbols.index(symbol) for symbol in common_symbols if symbol in all_symbols]
    grouped = stock_data.groupby('ID')
    result = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(create_single_stock_tensor)((symbol, group, features)) for symbol, group in tqdm(grouped)
    )
    stock_tensors = dict(result)
    return stock_tensors, common_stock_indices

# KGTensorManager to handle knowledge graph tensors
def create_single_kg_tensor(args):
    """ 
    Create a single tensor for a given symbol in the knowledge graph (KG).
    
    Args:
        args (tuple): A tuple containing the symbol_id (str), the group of relations (pd.DataFrame), 
                      fixed relations (list), and the entity embeddings (dict).
        
    Returns:
        tuple: The symbol_id, the KG tensor, and the entity matrix for the symbol.
    """
    symbol_id, group, fixed_relations, entity_embeddings = args
    unique_entities = sorted(set(group['head'].unique()))
    entity_idx = {entity: idx for idx, entity in enumerate(unique_entities)}
    relation_idx = {relation: idx for idx, relation in enumerate(fixed_relations)}
    tensor = np.zeros((len(unique_entities), len(fixed_relations)))  
    for _, row in group.iterrows():
        head = row['head']
        relation = row['relation']
        tail = row['tail']
        tensor[entity_idx[head], relation_idx[relation]] = 1
        
    M_k = np.array([entity_embeddings[entity] for entity in unique_entities])
    
    return symbol_id, tensor, M_k


def create_kg_tensor_parallel(kg_data, common_symbols, entity_embeddings):
    """ 
    Create knowledge graph tensors in parallel for multiple stock symbols.
    
    Args:
        kg_data (pd.DataFrame): DataFrame containing KG data.
        common_symbols (pd.DataFrame): DataFrame with stock symbols and IDs.
        entity_embeddings (dict): Dictionary containing entity embeddings for each entity.
        
    Returns:
        tuple: A dictionary of KG tensors and a dictionary of entity matrices for each symbol.
    """
    all_ids = common_symbols['id'].unique()
    grouped = kg_data.groupby('tail')
    fixed_relations = sorted(kg_data['relation'].unique())

    result = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(create_single_kg_tensor)((symbol_id, group, fixed_relations, entity_embeddings))
        for symbol_id, group in tqdm(grouped) if symbol_id in all_ids
    )

    kg_tensors = {res[0]: res[1] for res in result}
    kg_entity_matrices = {res[0]: res[2] for res in result}
    return kg_tensors, kg_entity_matrices

def create_symbols_from_kg_tensors(stock_data, kg_tensors, common_symbols, symbol_to_id):
    """ 
    Separate stock symbols into common and stock-only categories.
    
    Args:
        stock_data (pd.DataFrame): DataFrame containing stock data.
        kg_tensors (dict): Dictionary of KG tensors.
        common_symbols (pd.DataFrame): DataFrame with stock symbols and IDs.
        symbol_to_id (dict): Dictionary mapping stock symbols to IDs.
        
    Returns:
        tuple: List of common stock symbols and list of stock-only symbols.
    """
    all_symbols = sorted(stock_data['ID'].unique())
    id_to_symbol = {v: k for k, v in dict(zip(common_symbols['symbol'], common_symbols['id'])).items()}
    common_symbol_ids = set(kg_tensors.keys())
    common_symbols_list = []
    stock_only_symbols_list = []
    for symbol in all_symbols:
        symbol_id = symbol_to_id.get(symbol)
        if symbol_id in common_symbol_ids:
            common_symbols_list.append(symbol)
        else:
            stock_only_symbols_list.append(symbol)
    return common_symbols_list, stock_only_symbols_list

def initialize_factors(num_companies, rank, block_shapes, kg_tensor_shapes, kg_tensors):
    """ 
    Initialize the factors for both stock and knowledge graph tensors using Xavier initialization.
    
    Args:
        num_companies (int): Number of stock companies.
        rank (int): Rank of the factor matrices.
        block_shapes (list): List of shapes for current block tensors (not entire dataset).
        kg_tensor_shapes (list): List of shapes for KG tensors.
        kg_tensors (dict): Dictionary of KG tensors.
        
    Returns:
        tuple: Initialized factors for stock and KG tensors.
    """

    def xavier_init(shape):
        if isinstance(shape, int):  
            return np.random.randn(shape) * np.sqrt(2 / shape)
        elif len(shape) == 1:  
            return np.random.randn(shape[0]) * np.sqrt(2 / shape[0])
        else:  
            return np.random.randn(*shape) * np.sqrt(2 / (shape[0] + shape[1]))

    # Initialize factors for current block of stock data
    factors_stock = [
        [xavier_init((shape[0], rank)) for shape in block_shapes],  # U_k for block data
        [np.diag(xavier_init((rank))) for _ in range(num_companies)],  # Diagonal S_k
        xavier_init((block_shapes[0][1], rank)),  # Shared V
        [xavier_init((shape[0], rank)) for shape in block_shapes],  # Q_k for block data
        xavier_init((rank, rank))  # Shared H
    ]

    # Initialize factors for knowledge graph data
    factors_kg = [
        [xavier_init((shape[0], rank)) for shape in kg_tensor_shapes],  # M_k
        [xavier_init((shape[0], rank)) for shape in kg_tensor_shapes],  # M_k alternative
        xavier_init((kg_tensor_shapes[0][1], rank)),  # R
        xavier_init((rank, rank)),  # Shared part for KG
        [np.diag(np.sum(G_k, axis=0)) for G_k in kg_tensors.values()],  # D_k
        np.ones((kg_tensor_shapes[0][1], rank))  # Placeholder
    ]

    return factors_stock, factors_kg


#====================================================================================================
# Load data

def initialize_data():

    trn_file = '/home/duduuman/O_data_JPN/data_filtered/stock_train_30.csv'
    true_file = '/home/duduuman/O_data_JPN/data_filtered/stock_true.csv'
    missing_idxs = np.load('/home/duduuman/O_data_JPN/data_filtered/missing_idxs_30.npy', allow_pickle=True).item()
    aux_file = '/home/duduuman/O_data_JPN/data_filtered/updated_relations.txt'
    common_symbols_file = '/home/duduuman/O_data_JPN/data_filtered/common_symbols2id.txt'
    rank = 5
    
    stock_train = pd.read_csv(trn_file, dtype={'ID': str})
    stock_true = pd.read_csv(true_file, dtype={'ID': str})
    common_symbols = pd.read_csv(common_symbols_file, sep='\t', header=None, names=['symbol', 'id'])

    symbol_to_id = dict(zip(common_symbols['symbol'], common_symbols['id']))
    id_to_symbol = {v: k for k, v in symbol_to_id.items()}

    kg_data = pd.read_csv(aux_file, sep='\t', header=None, names=['head', 'relation', 'tail'])
    
    entity_embeddings = {entity: np.random.rand(rank) for entity in kg_data['head'].unique()}  
    if not os.path.exists("./cached"):
        os.makedirs("cached")
    with open("./cached/stock_train.pkl", "wb") as f:
        pickle.dump(stock_train, f)
    with open("./cached/stock_true.pkl", "wb") as f:
        pickle.dump(stock_true, f)
    with open("./cached/common_symbols.pkl", "wb") as f:
        pickle.dump(common_symbols, f)
    with open("./cached/kg_data.pkl", "wb") as f:
        pickle.dump(kg_data, f)
    with open("./cached/entity_embeddings.pkl", "wb") as f:
        pickle.dump(entity_embeddings, f)
    with open("./cached/symbol_to_id.pkl", "wb") as f:
        pickle.dump(symbol_to_id, f)

    return stock_train, stock_true, common_symbols, kg_data, entity_embeddings, symbol_to_id, missing_idxs

def read_data():
    if not os.path.exists("./cached"):    
        stock_train, stock_true, common_symbols, kg_data, entity_embeddings, symbol_to_id, missing_idxs = initialize_data() 
        return stock_train, stock_true, common_symbols, kg_data, entity_embeddings, symbol_to_id, missing_idxs

    with open("./cached/stock_train.pkl", "rb") as f:
        stock_train = pickle.load(f)
    with open("./cached/stock_true_par.pkl", "rb") as f:
        stock_true= pickle.load(f)
    with open("./cached/common_symbols.pkl", "rb") as f:
        common_symbols= pickle.load(f)
    with open("./cached/kg_data.pkl", "rb") as f:
        kg_data=pickle.load(f)
    with open("./cached/entity_embeddings.pkl", "rb") as f:
        entity_embeddings=pickle.load(f)
    with open("./cached/symbol_to_id.pkl", "rb") as f:
        symbol_to_id=pickle.load(f)
    missing_idxs = np.load('/home/duduuman/O_data_JPN/data_filtered/missing_idxs_30.npy', allow_pickle=True).item()
    return stock_train, stock_true, common_symbols, kg_data, entity_embeddings, symbol_to_id, missing_idxs

################################################

def load_data():
    print("Loading data with streaming setup...")
    stock_train, stock_true, common_symbols, kg_data, entity_embeddings, symbol_to_id, missing_idxs = read_data()
    id_to_symbol = {v: k for k, v in symbol_to_id.items()}

    print("Preparing tensors...")
    stock_tensors, common_stock_indices = create_stock_tensor_parallel(stock_train, common_symbols['symbol'])
    stock_true, _ = create_stock_tensor_parallel(stock_true, common_symbols['symbol'])
    kg_tensors, kg_entity_matrices = create_kg_tensor_parallel(kg_data, common_symbols, entity_embeddings)

    common_symbols_list, stock_only_symbols_list = create_symbols_from_kg_tensors(stock_train, kg_tensors, common_symbols, symbol_to_id)

    ordered_stock_tensors = OrderedDict()
    ordered_kg_tensors = OrderedDict()

    for symbol in stock_tensors.keys():
        symbol_id = symbol_to_id[symbol]
        ordered_stock_tensors[symbol] = stock_tensors[symbol]
        ordered_kg_tensors[symbol_id] = kg_tensors[symbol_id]

    all_stock_tensors = ordered_stock_tensors
    kg_tensors = ordered_kg_tensors

    num_companies = len(all_stock_tensors)

    # Initial settings for streaming
    initial_percentage = 0.4 # Initial 50%
    #streaming_percentage = 0.2  # Incremental 20%

    #N_initial = int(initial_percentage * list(all_stock_tensors.values())[0].shape[0])
    #N_stream = int(streaming_percentage * list(all_stock_tensors.values())[0].shape[0])
    N_stream =500

    stock_tensors = OrderedDict()
    stock_old_tensors = {}
    streaming_data = {}

    for sym, tensor in all_stock_tensors.items():
        N_initial = int(initial_percentage * tensor.shape[0])
        stock_tensors[sym] = tensor[:N_initial]
        stock_old_tensors[sym] = np.empty((0, tensor.shape[1]))
        streaming_data[sym] = tensor[N_initial:]

    print("N_initial per symbol:", {sym: tensor.shape[0] for sym, tensor in stock_tensors.items()})


    # Initialize factors for the initial block
    block_shapes = [tensor.shape for tensor in stock_tensors.values()]
    kg_tensor_shapes = [tensor.shape for tensor in kg_tensors.values()]
    rank = 5

    factors_stock, factors_kg = initialize_factors(num_companies, rank, block_shapes, kg_tensor_shapes, kg_tensors)



    # # Initialize storage for old factors
    # factors_old_stock = [
    #     [np.empty((0, rank)) for _ in range(num_companies)],  # U_old_list
    #     np.copy(factors_stock[1]),  # Shared S
    #     np.copy(factors_stock[2]),  # Shared V
    #     [np.empty((0, rank)) for _ in range(num_companies)],  # Q_old_list
    #     np.copy(factors_stock[4])  # Shared H
    # ]

    # Initialize storage for old factors as a dictionary
    factors_old_stock = {
        symbol: {
            "U_old": np.empty((0, rank)),  # U_old_list
            "S": np.copy(factors_stock[1][idx]),            # Shared S
            "V": np.copy(factors_stock[2]),  # V 초기화
            "Q_old": np.empty((0, rank)),  # Q_old_list
            "H": np.copy(factors_stock[4])              # Shared H
        }
        for idx, symbol in enumerate(all_stock_tensors.keys())
    }



    # Cache all data
    # 데이터 캐싱
    if not os.path.exists("./cached"):
        os.makedirs("./cached")
    with open("./cached/factors_stock.pkl", "wb") as f:
        pickle.dump(factors_stock, f)
    with open("./cached/factors_kg.pkl", "wb") as f:
        pickle.dump(factors_kg, f)
    with open("./cached/all_stock_tensors.pkl", "wb") as f:
        pickle.dump(all_stock_tensors, f)    
    with open("./cached/stock_tensors.pkl", "wb") as f:
        pickle.dump(stock_tensors, f)
    with open("./cached/stock_true_par.pkl", "wb") as f:
        pickle.dump(stock_true, f)
    with open("./cached/kg_tensors.pkl", "wb") as f:
        pickle.dump(kg_tensors, f)
    with open("./cached/common_symbols_list.pkl", "wb") as f:
        pickle.dump(common_symbols_list, f)
    with open("./cached/stock_symbols_list.pkl", "wb") as f:
        pickle.dump(stock_only_symbols_list, f)
    with open("./cached/factors_old_stock.pkl", "wb") as f:
        pickle.dump(factors_old_stock, f)
    with open("./cached/stock_old_tensors.pkl", "wb") as f:
        pickle.dump(stock_old_tensors, f)
    with open("./cached/streaming_data.pkl", "wb") as f:
        pickle.dump(streaming_data, f)

    # 추가 반환: streaming_data와 N_stream
    return (
        factors_stock, factors_kg, kg_data, all_stock_tensors, stock_tensors, kg_tensors, entity_embeddings, 
        common_symbols_list, stock_only_symbols_list, stock_train, stock_true, common_symbols, symbol_to_id, missing_idxs,
        factors_old_stock, stock_old_tensors, streaming_data
    )

def load_cached():
    if not os.path.exists("./cached/factors_stock.pkl"):
        return load_data()

    with open("./cached/factors_stock.pkl", "rb") as f:
        factors_stock=pickle.load(f)
    with open("./cached/factors_kg.pkl", "rb") as f:
        factors_kg=pickle.load(f)
    with open("./cached/all_stock_tensors.pkl", "rb") as f:
        all_stock_tensors=pickle.load(f)    
    with open("./cached/stock_tensors.pkl", "rb") as f:
        stock_tensors=pickle.load(f)
    with open("./cached/kg_tensors.pkl", "rb") as f:
        kg_tensors=pickle.load(f)
    with open("./cached/common_symbols_list.pkl", "rb") as f:
        common_symbols_list=pickle.load(f)
    with open("./cached/stock_symbols_list.pkl", "rb") as f:
        stock_only_symbols_list=pickle.load(f)
    with open("./cached/stock_true_par.pkl", "rb") as f:
        stock_true = pickle.load(f)

    stock_train, stock_true_tmp, common_symbols, kg_data, entity_embeddings, symbol_to_id, missing_idxs = read_data()
    if stock_true_tmp is not None:
        stock_true = stock_true_tmp

    with open("./cached/factors_old_stock.pkl", "rb") as f:
        factors_old_stock = pickle.load(f)
    with open("./cached/stock_old_tensors.pkl", "rb") as f:
        stock_old_tensors = pickle.load(f)
    with open("./cached/streaming_data.pkl", "rb") as f:
        streaming_data = pickle.load(f)

    return (factors_stock, factors_kg, kg_data, all_stock_tensors, stock_tensors, kg_tensors, entity_embeddings, 
            common_symbols_list, stock_only_symbols_list, stock_train, stock_true, common_symbols, symbol_to_id, missing_idxs,
            factors_old_stock, stock_old_tensors, streaming_data)