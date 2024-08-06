import numpy as np
import math

class Simulate:
    def __init__(self, mac_size=128):
        self.mac_size = mac_size
    
    def run_simulate(self, configs):
        print("=====================================================")
        print("************* Systolic Array Simulation *************")
        print("=====================================================")
        print()
        
        for i in range(len(configs)):
            m, k, n = configs[i]
            
            print("[Case " + str(i + 1) + "]")
            if m == k and k == n:
                print(f": M = K = N = {m}")
            else:
                print(f": M = {m}, K = {k}, N = {n}")
            
            cycle = self.run_once(i, m, k, n)

            print(f"Cycles for computation : {cycle} cycles\n")
        
        print("=====================================================")
    
    def run_once(self, i, m, k, n):
        cycle = 0
        
        weight_matrix = np.random.random((m, k))
        input_matrix = np.random.random((k, n))
        output_matrix = np.zeros((m, n))
        
        mac_array = np.zeros((self.mac_size, self.mac_size))
        mac_register = np.zeros((self.mac_size, self.mac_size))
        
        # Number of iterations
        N_row = math.ceil(m / self.mac_size)
        N_col = math.ceil(k / self.mac_size)
        
        for mac_col in range(N_col):
            for mac_row in range(N_row):
                
                # Fill the MAC array with weight values (128 x 128)
                for i in range(self.mac_size):
                    # Transfer weight values to adjacent units
                    for j in range(0, self.mac_size - 1):
                        mac_array[j] = mac_array[j + 1]
                    tgt_row = mac_row * self.mac_size + i
                    tgt_col = mac_col * self.mac_size
                    if tgt_row >= m:
                        mac_array[self.mac_size - 1] = np.zeros(self.mac_size)
                    elif tgt_col + self.mac_size > k:
                        stream_weights = weight_matrix[tgt_row][tgt_col : tgt_col + self.mac_size]
                        mac_array[self.mac_size - 1] = np.pad(stream_weights, (0, self.mac_size - len(stream_weights)), mode='constant', constant_values=0)
                    else:
                        mac_array[self.mac_size - 1] = weight_matrix[tgt_row][tgt_col : tgt_col + self.mac_size]
                    
                    cycle += 1
                
                # Reconstruct the input partition (Instead of input FIFO)
                input_partition = self.reconstruct_input(input_matrix[mac_col * self.mac_size : (mac_col + 1) * self.mac_size])
                
                # Implement matrix multiplication and produce output
                # for i in range(self.mac_size):
                #     for j in range(i):
                #         for k in range(i):
                #             mac_register[j][k + 1] = mac_array[j][k] * input_partition[j][k]
                
                # print(mac_register)
                
                for i in range(n + self.mac_size - 1):
                    
                    cycle += 1
                
                for i in range(self.mac_size):
                    
                    cycle += 1
                
                # Reset the MAC array and register
                mac_array = np.zeros((self.mac_size, self.mac_size))
                mac_register = np.zeros((self.mac_size, self.mac_size))
        
        return cycle

    def reconstruct_input(self, input_array):
        rows, cols = input_array.shape
        new_rows = max(rows, self.mac_size)
        # input_partition = np.zeros((rows + cols - 1, rows), dtype=input_array.dtype)
        input_partition = np.zeros((new_rows + cols - 1, new_rows))
        
        for i in range(rows):
            for j in range(cols):
                input_partition[i + j][i] = input_array[i][j]
        
        return input_partition


if __name__ == '__main__':
    simulate = Simulate(128)
    # simulate.run_simulate([
    #     (128, 128, 128)
    # ])
    simulate.run_simulate([
        (128, 128, 128),
        (256, 256, 256),
        (200, 128, 128),
        (128, 200, 200),
    ])
