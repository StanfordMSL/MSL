from abc import ABC, abstractmethod

class BaseController(ABC):
    """
    Abstract base class for controllers.

    Methods:
        control(**inputs): Abstract method to be implemented by subclasses.
    """

    @abstractmethod
    def control(self, **inputs):
        """
        Abstract control method to be implemented by subclasses.
        Args:
            **inputs: Key-value pairs representing the inputs.
        """
        pass