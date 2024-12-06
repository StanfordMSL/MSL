from abc import ABC, abstractmethod

class BaseController(ABC):
    @abstractmethod
    def control(self, **inputs):
        """
        Abstract control method to be implemented by subclasses.
        Args:
            **inputs: Key-value pairs representing the inputs.
        """
        pass