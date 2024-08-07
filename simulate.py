import numpy as np
import math

class SystolicArraySim:
    def __init__(self, weight_matrix, activation_matrix, mac_size=128):
        self.weight_matrix = weight_matrix
        self.activation_matrix = activation_matrix
        self.mac_size = mac_size
        
        self.m, self.k = weight_matrix.shape
        _, self.n = activation_matrix.shape
        
        self.cycle = 0
        self.utilization = 0
    
    def run_simulate(self, i):
        output_matrix = np.zeros((self.m, self.n))
        
        mac_register = np.zeros((self.mac_size, self.mac_size))
        
        input_stream = np.zeros((self.mac_size, self.mac_size))
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
                    
                    self.compute_cycle_and_utilization(input_stream)
                
                # Reconstruct the input partition (Instead of input FIFO)
                # and prepare the output FIFO
                input_fifo = self.reconstruct_input(self.activation_matrix[N_row * self.mac_size : (N_row + 1) * self.mac_size])
                output_fifo = np.zeros(input_fifo.shape)
                
                # Implement matrix multiplication and produce output
                for i in range(self.mac_size - 1):
                    
                    input_stream[:, 1:] = input_stream[:, :-1]
                    input_stream[:, 0] = input_fifo[i]
                    
                    mac_multiply = mac_register * input_stream
                    mac_accumulate = mac_multiply + psum_stream
                    psum_stream[1:, :] = mac_accumulate[:-1, :]
                    psum_stream[0] = 0
                        
                    self.compute_cycle_and_utilization(input_stream)
                
                for i in range(self.mac_size - 1, self.n + self.mac_size - 1):
                    k = i - self.mac_size + 1
                    
                    input_stream[:, 1:] = input_stream[:, :-1]
                    input_stream[:, 0] = input_fifo[i]
                    
                    mac_multiply = mac_register * input_stream
                    mac_accumulate = mac_multiply + psum_stream
                    psum_stream[1:, :] = mac_accumulate[:-1, :]
                    psum_stream[0] = 0
                    
                    output_fifo[k] = mac_accumulate[-1]
                    
                    self.compute_cycle_and_utilization(input_stream)
                
                for i in range(self.mac_size):
                    
                    input_stream[:, 1:] = input_stream[:, :-1]
                    input_stream[:, 0] = 0
                    
                    mac_multiply = mac_register * input_stream
                    mac_accumulate = mac_multiply + psum_stream
                    psum_stream[1:, :] = mac_accumulate[:-1, :]
                    psum_stream[0] = 0
                    
                    output_fifo[k] = mac_accumulate[-1]
                    
                    self.compute_cycle_and_utilization(input_stream)
                
                # Reconstruct the output FIFO into output matrix
                output_computation_result = self.reconstruct_output(N_row, output_fifo)
                if (N_row + 1) * self.mac_size > self.m:
                    output_matrix[N_row * self.mac_size : self.m] += output_computation_result
                else:
                    output_matrix[N_row * self.mac_size : (N_row + 1) * self.mac_size] += output_computation_result
                
                # Reset the MAC array and register
                mac_register = np.zeros((self.mac_size, self.mac_size))
                
                input_stream = np.zeros((self.mac_size, self.mac_size))
                psum_stream = np.zeros((self.mac_size, self.mac_size))
                pe_in_compute = np.zeros((self.mac_size, self.mac_size))
        
        return output_matrix, self.cycle, self.utilization / self.cycle

    def reconstruct_input(self, input_array):
        rows, cols = input_array.shape
        reconstructed_input = np.zeros((cols + self.mac_size - 1, self.mac_size))
        
        for i in range(rows):
            for j in range(cols):
                k = self.mac_size - i - 1
                reconstructed_input[j + k][k] = input_array[i][j]
        
        return reconstructed_input
    
    def reconstruct_output(self, N_row, output_array):
        if (N_row + 1) * self.mac_size > self.m:
            reconstructed_output = np.zeros((self.m - (N_row * self.mac_size), self.n))
        else:
            reconstructed_output = np.zeros((self.mac_size, self.n))
        
        return reconstructed_output
    
    def compute_cycle_and_utilization(self, input_stream):
        self.cycle += 1
        self.utilization += np.count_nonzero(input_stream) / (self.mac_size * self.mac_size)
        
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
