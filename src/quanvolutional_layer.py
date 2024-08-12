from quanvolutional_filter import QuanvolutionalFilter


class QuanvolutionalLayer:
    """Quanvolutional layer class.
    """
    
    def __init__(self, kernel_size: tuple[int, int], num_filters: int):
        # Store the arguments to class variables.
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        
        # Get the quanvolutional filters.
        self.quanvolutional_filters = [QuanvolutionalFilter(self.kernel_size) for _ in range(self.num_filters)]
