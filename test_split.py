import numpy as np

def test_split():
    k_agg = np.zeros((1, 144, 512))
    n_past_k = 24
    
    print(f"k_agg shape: {k_agg.shape}")
    print(f"n_past_k: {n_past_k}")
    
    try:
        if k_agg.shape[0] % n_past_k == 0:
            print("Trying split axis 0")
            chunks = np.split(k_agg, n_past_k, axis=0)
            print("Success axis 0")
        else:
            print(f"Cannot split axis 0: {k_agg.shape[0]} % {n_past_k} != 0")
            
        if len(k_agg.shape) > 1 and k_agg.shape[1] % n_past_k == 0:
            print("Trying split axis 1")
            chunks = np.split(k_agg, n_past_k, axis=1)
            print(f"Success axis 1. Chunk shape: {chunks[0].shape}")
        else:
            print(f"Cannot split axis 1: {k_agg.shape[1]} % {n_past_k} != 0")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_split()

