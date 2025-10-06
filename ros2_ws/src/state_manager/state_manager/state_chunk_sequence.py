import math
from collections import deque
from dataclasses import dataclass
from itertools import islice


@dataclass
class StateChunk:
    """Representation for one unit of state chunk information."""
    chunk: str
    token_len: int
    ts: float


class StateChunkSequence:
    """Optimized for storing and partially clearing large state chunk sequences.

    Example:
        >>> seq = StateChunkSequence(100)
        >>> for _ in range(3):
        >>>    seq.append(chunk)
        >>> seq.clear(0.5)
        >>> print(len(seq))  # --> 1 (most recent chunk remains)

    NOTE: The state chunk sequence is expected to be manually cleared before
    reaching max token length.

    Example:
        >>> if len(seq) >= state_max_tokens:
        >>>     seq.clear(0.5)

    Raises:
        RuntimeError if adding state chunks beyond queue capacity would result
            in eviction and thus break the state chunk sequence order.
    """

    def __init__(self, max_length: int) -> None:
        """
        Args:
            max_length: Number of state chunks should be larger than possible
                max tokens. Recommend setting max_length := max_tokens for
                safety (assuming worst-case: 1 state chunk = 1 token).
        """
        self.max_state_chunks = max_length

        self.seq = deque(maxlen=self.max_state_chunks)
        self.token_len = 0

    def __len__(self) -> int:
        """Returns the sequence token length (NOT number of state chunks!)"""
        return self.token_len

    def get_token_len(self):
        return self.__len__()

    def get_num_state_chunks(self):
        return len(self.seq)

    def append(self, state_chunk: StateChunk) -> None:
        # Check if append operation will evict oldest element
        if len(self.seq) == self.max_state_chunks:
            raise RuntimeError(
                f'StateChunkSequence at max capacity ({self.max_state_chunks})'
                'Must call clear() before appending enw state chunks.')

        self.seq.append(state_chunk)
        self.token_len += state_chunk.token_len

    def clear(self, ratio: float = 0.5) -> None:
        """Clears the sequence of length N to contain the K newest elements.

        A new sequence is created without an intermediate list using islict to
        maximize performance.

        Args:
            ratio: Ratio of elements to keep starting from the most recent.
                ratio=0.0 clears all elements
                ratio=1.0 keeps all elements
                ratio=0.5 keeps newest 50% of elements
        """
        assert 0 <= ratio <= 1, 'Ratio must be between 0 and 1'
        N = len(self.seq)
        K = math.floor(N * ratio)
        skip = max(0, N - K)

        self.seq = deque(
            islice(self.seq, skip, None),
            maxlen=self.max_state_chunks,
        )

        # Recalculate cleared sequence token length
        self.token_len = sum(
            [state_chunk.token_len for state_chunk in self.seq])
