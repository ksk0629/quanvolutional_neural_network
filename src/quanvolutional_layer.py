import numpy as np
from tqdm.auto import tqdm

from quanvolutional_filter import QuanvolutionalFilter


class QuanvolutionalLayer:
    """Quanvolutional layer class.
    """
    
    def __init__(self, kernel_size: tuple[int, int], num_filters: int, padding_mode: str | None="constant"):
        # Store the arguments to class variables.
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.padding_mode = padding_mode
        
        # Get the quanvolutional filters.
        self.quanvolutional_filters = [QuanvolutionalFilter(self.kernel_size) for _ in range(self.num_filters)]
    
    def run_for_dataset(self, dataset: np.ndarray, shots: int) -> np.ndarray:
        all_outputs = [
            self.run_single_channel(data=data, shots=shots)
            for data in tqdm(dataset, leave=True, desc="Dataset")
        ]
        return np.array(all_outputs)

    def run_single_channel(self, data: np.ndarray, shots:int) -> np.ndarray:
        # Perform padding to make the output shape as same as the input  accodring to the mode.
        padded_data = np.pad(data, ((self.kernel_size[0] // 2,), (self.kernel_size[1] // 2,)), mode=self.padding_mode)

        # Make the strided data.
        new_shape = (padded_data.shape[0] - 3 + 1, padded_data.shape[1] - 3 + 1) + (3, 3)
        new_strides = padded_data.strides * 2
        strided_data = np.lib.stride_tricks.as_strided(padded_data, new_shape, new_strides)
        
        # Reshape the strided data to feed to the quanvolutional filters.
        reshaped_strided_data = strided_data.reshape((-1, self.kernel_size[0] * self.kernel_size[1]))

        # Perform the quanvolutional filters.
        outputs = np.empty([len(self.quanvolutional_filters), *data.shape])
        for index, quanvolutional_filter in enumerate(tqdm(self.quanvolutional_filters, leave=False, desc="Filters")):
            vectorized_quanvolutional_filter_run = np.vectorize(quanvolutional_filter.run,
                                                                signature="(n),()->()")            
            outputs[index, :, :] = vectorized_quanvolutional_filter_run(reshaped_strided_data, shots).reshape(data.shape)
        return outputs

    def run_for_dataset_and_save(self, dataset: np.ndarray, shots: int, filename: str):
        outputs = self.run_for_dataset(dataset=dataset, shots=shots)
        np.save(filename, outputs)
