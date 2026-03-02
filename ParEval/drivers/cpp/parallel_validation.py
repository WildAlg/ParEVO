""" Validate that code snippets are using the correct parallelism model for
    computation. Covers OpenMP, and MPI.
    author: Daniel Nichols
    date: November 2023
"""
# std imports
from abc import ABC, abstractmethod


class Validator(ABC):
    parallelism_model: str

    def __init__(self, parallelism_model: str):
        self.parallelism_model = parallelism_model

    @abstractmethod
    def validate(self, source: str) -> bool:
        """ Validate that the given source is using the correct parallelism model. """
        raise NotImplementedError("validate() not implemented for this validator.")

    def must_contain(self, source: str, substr: str) -> bool:
        """ Check if the given source contains the given substring. """
        return substr in source


class OMPValidator(Validator):

    def __init__(self):
        super().__init__("omp")

    """ Validate that the given source uses OpenMP. """
    def validate(self, source: str) -> bool:
        return self.must_contain(source, "#pragma omp")


class MPIValidator(Validator):
    
    def __init__(self):
        super().__init__("mpi")

    """ Validate that the given source uses MPI. """
    def validate(self, source: str) -> bool:
        return self.must_contain(source, "MPI_") or self.must_contain(source, "MPI.")


class MPIandOMPValidator(Validator):
    
    def __init__(self):
        super().__init__("mpi+omp")
        self.mpi_validator = MPIValidator()
        self.omp_validator = OMPValidator()

    """ Validate that the given source uses MPI and OpenMP. """
    def validate(self, source: str) -> bool:
        return self.mpi_validator.validate(source) and self.omp_validator.validate(source)


class EmptyValidator(Validator):

    def __init__(self):
        super().__init__("empty")

    """ Always returns true. """
    def validate(self, source: str) -> bool:
        return True


class ParlayValidator(Validator):

    def __init__(self):
        super().__init__("parlay")

    """ Validate that the given source uses OpenMP. """
    def validate(self, source: str) -> bool:
        return self.must_contain(source, "parlay")
    
class RustValidator(Validator):

    def __init__(self):
        super().__init__("rust")

    """ Validate that the given source uses Rust. """
    def validate(self, source: str) -> bool:
        # Normalize code to avoid whitespace issues
        # code = source.lower()
        print("code: ", source)
        
        # 1. Check for library usage (High Confidence)
        if self.must_contain(source, "use rayon") or self.must_contain(source, "rayon::"):
            return True
            
        # 2. Check for common parallel iterator methods
        # Note: Technically another library could name a method 'par_iter', 
        # but in the Rust ecosystem, this is almost exclusively Rayon.
        rayon_methods = [
            "par_",
            "join",         # rayon::join
            "scope"         # rayon::scope
        ]
        
        for method in rayon_methods:
            if self.must_contain(source, method):
                return True
                
        return False

        # return self.must_contain(source, "pub fn") and self.must_contain(source, ".par_iter()")