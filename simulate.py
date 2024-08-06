import numpy as np
import math

class SystolicArraySim:
    def __init__(self, weight_matrix, activation_matrix, mac_size=128):
        self.weight_matrix = weight_matrix
        self.activation_matrix = activation_matrix
        self.mac_size = mac_size
        
        self.m, self.k = weight_matrix.shape
        _, self.n = activation_matrix.shape
    
    def run_simulate(self, i):
        cycle = 0
        
        output_matrix = np.zeros((self.m, self.n))
        
        mac_register = np.zeros((self.mac_size, self.mac_size))
        psum_stream = np.zeros((self.mac_size, self.mac_size))
        pe_in_compute = np.zeros((self.mac_size, self.mac_size))
        
        # Number of iterations (row, col)
        num_iter = (math.ceil(self.m / self.mac_size), math.ceil(self.k / self.mac_size))
        
        for N_col in range(num_iter[1]):
            for N_row in range(num_iter[0]):
                
                # 1. Fill the MAC array with weight values (128 x 128)
                for i in range(self.mac_size):
                    # Transfer weight values to adjacent units
                    for j in reversed(range(1, self.mac_size)):
                        mac_register[j] = mac_register[j - 1]
                    
                    new_row = N_row * self.mac_size
                    new_col = N_col * self.mac_size + i
                    
                    if new_col >= self.k:
                        mac_register[0] = np.zeros(self.mac_size)
                    elif new_row + self.mac_size > self.m:
                        new_weights = self.weight_matrix[new_row:, new_col]
                        mac_register[0] = np.pad(new_weights,
                                                 (0, self.mac_size - len(new_weights)),
                                                 mode='constant',
                                                 constant_values=0)
                    else:
                        mac_register[0] = self.weight_matrix[new_row : new_row + self.mac_size, new_col]
                    
                    cycle += 1
                
                # Reconstruct the input partition (Instead of input FIFO)
                input_stream = self.reconstruct_input(self.activation_matrix[N_row * self.mac_size : (N_row + 1) * self.mac_size])
                
                # Implement matrix multiplication and produce output
                for i in range(self.n):
                    for j in range(self.mac_size):
                        
                        if input_stream[i][j] is None:
                            break
                        
                        cycle += 1
                
                for i in range(self.n, self.n + self.mac_size - 1):
                    for j in range(self.mac_size):
                        
                        if input_stream[i][j] is None:
                            continue
                        
                        cycle += 1
                
                
                for i in range(self.mac_size):
                    
                    cycle += 1
                
                # Reset the MAC array and register
                mac_register = np.zeros((self.mac_size, self.mac_size))
                psum_stream = np.zeros((self.mac_size, self.mac_size))
        
        return output_matrix, cycle

    def reconstruct_input(self, input_array):
        rows, cols = input_array.shape
        input_stream = np.full((cols + self.mac_size - 1, self.mac_size), None, dtype='float64')
        
        for i in range(rows):
            for j in range(cols):
                k = self.mac_size - i - 1
                input_stream[j + k][k] = input_array[i][j]
        
        return input_stream
    
    def compute_utilization(self, mac_register):
        
        return


if __name__ == "__main__":
    simulate = SystolicArraySim(mac_size=128)
    # simulate.run_simulate([
    #     (128, 128, 128)
    # ])
    simulate.run_simulate([
        (128, 128, 128),
        (256, 256, 256),
        (200, 128, 128),
        (128, 200, 200),
    ])
