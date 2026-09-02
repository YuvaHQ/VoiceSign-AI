from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np
from src.models.schemas import RecognitionResult, SignLanguageEnum

class SignRecognizerInterface(ABC):
    @abstractmethod
    def recognize_sign(
        self,
        input_sequence: Union[List[float], List[List[float]], np.ndarray],
        language: SignLanguageEnum,
    ) -> RecognitionResult:
        pass

    @abstractmethod
    def recognize_custom_sign(
        self,
        user_id: str,
        input_sequence: Union[List[float], List[List[float]], np.ndarray],
    ) -> Optional[RecognitionResult]:
        pass
