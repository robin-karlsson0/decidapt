import pytest
from state_manager.state_chunk_sequence import StateChunk, StateChunkSequence


class TestStateChunk:
    """Test the StateChunk dataclass."""

    def test_state_chunk_creation(self):
        """Test creating a StateChunk."""
        chunk = StateChunk(chunk="test chunk", token_len=10)
        assert chunk.chunk == "test chunk"
        assert chunk.token_len == 10


class TestStateChunkSequence:
    """Test the StateChunkSequence class."""

    def test_initialization(self):
        """Test that StateChunkSequence initializes correctly."""
        seq = StateChunkSequence(max_length=100)
        assert seq.max_state_chunks == 100
        assert len(seq) == 0
        assert seq.get_token_len() == 0
        assert seq.get_num_state_chunks() == 0

    def test_append_single_chunk(self):
        """Test appending a single state chunk."""
        seq = StateChunkSequence(max_length=100)
        chunk = StateChunk(chunk="test", token_len=5)

        seq.append(chunk)

        assert len(seq) == 5
        assert seq.get_token_len() == 5
        assert seq.get_num_state_chunks() == 1

    def test_append_multiple_chunks(self):
        """Test appending multiple state chunks."""
        seq = StateChunkSequence(max_length=100)
        chunks = [
            StateChunk(chunk="chunk1", token_len=10),
            StateChunk(chunk="chunk2", token_len=15),
            StateChunk(chunk="chunk3", token_len=20),
        ]

        for chunk in chunks:
            seq.append(chunk)

        assert len(seq) == 45  # 10 + 15 + 20
        assert seq.get_token_len() == 45
        assert seq.get_num_state_chunks() == 3

    def test_append_at_max_capacity_raises_error(self):
        """Test that appending beyond max capacity raises RuntimeError."""
        seq = StateChunkSequence(max_length=3)

        # Fill to capacity
        for i in range(3):
            seq.append(StateChunk(chunk=f"chunk{i}", token_len=10))

        # Attempting to add one more should raise RuntimeError
        with pytest.raises(RuntimeError,
                           match="StateChunkSequence at max capacity"):
            seq.append(StateChunk(chunk="overflow", token_len=10))

    def test_clear_with_ratio_half(self):
        """Test clearing half of the sequence (ratio=0.5)."""
        seq = StateChunkSequence(max_length=100)

        # Add 4 chunks
        for i in range(4):
            seq.append(StateChunk(chunk=f"chunk{i}", token_len=10))

        assert seq.get_num_state_chunks() == 4
        assert len(seq) == 40

        # Clear with ratio 0.5 (keep newest 50%)
        seq.clear(ratio=0.5)

        # Should keep 2 newest chunks (chunks 2 and 3)
        assert seq.get_num_state_chunks() == 2
        assert len(seq) == 20

    def test_clear_with_ratio_zero(self):
        """Test clearing all elements (ratio=0.0)."""
        seq = StateChunkSequence(max_length=100)

        # Add chunks
        for i in range(5):
            seq.append(StateChunk(chunk=f"chunk{i}", token_len=10))

        assert seq.get_num_state_chunks() == 5
        assert len(seq) == 50

        # Clear all
        seq.clear(ratio=0.0)

        assert seq.get_num_state_chunks() == 0
        assert len(seq) == 0

    def test_clear_with_ratio_one(self):
        """Test keeping all elements (ratio=1.0)."""
        seq = StateChunkSequence(max_length=100)

        # Add chunks
        for i in range(5):
            seq.append(StateChunk(chunk=f"chunk{i}", token_len=10))

        original_count = seq.get_num_state_chunks()
        original_length = len(seq)

        # Keep all
        seq.clear(ratio=1.0)

        assert seq.get_num_state_chunks() == original_count
        assert len(seq) == original_length

    def test_clear_with_various_ratios(self):
        """Test clearing with various ratios."""
        for ratio in [0.25, 0.33, 0.75, 0.9]:
            seq = StateChunkSequence(max_length=100)

            # Add 10 chunks
            for i in range(10):
                seq.append(StateChunk(chunk=f"chunk{i}", token_len=5))

            seq.clear(ratio=ratio)

            # Check that approximately the correct number remain
            expected_count = int(10 * ratio)
            assert seq.get_num_state_chunks() == expected_count
            assert len(seq) == expected_count * 5

    def test_clear_recalculates_token_length(self):
        """Test that clear() correctly recalculates token length."""
        seq = StateChunkSequence(max_length=100)

        # Add chunks with different token lengths
        seq.append(StateChunk(chunk="chunk1", token_len=10))
        seq.append(StateChunk(chunk="chunk2", token_len=20))
        seq.append(StateChunk(chunk="chunk3", token_len=30))
        seq.append(StateChunk(chunk="chunk4", token_len=40))

        assert len(seq) == 100

        # Clear to keep last 2 chunks (30 + 40 = 70 tokens)
        seq.clear(ratio=0.5)

        assert len(seq) == 70
        assert seq.get_num_state_chunks() == 2

    def test_clear_invalid_ratio_raises_assertion(self):
        """Test that invalid ratios raise AssertionError."""
        seq = StateChunkSequence(max_length=100)
        seq.append(StateChunk(chunk="test", token_len=10))

        # Test ratio > 1
        with pytest.raises(AssertionError,
                           match="Ratio must be between 0 and 1"):
            seq.clear(ratio=1.5)

        # Test ratio < 0
        with pytest.raises(AssertionError,
                           match="Ratio must be between 0 and 1"):
            seq.clear(ratio=-0.5)

    def test_append_after_clear(self):
        """Test appending chunks after clearing."""
        seq = StateChunkSequence(max_length=100)

        # Add and clear
        for i in range(5):
            seq.append(StateChunk(chunk=f"chunk{i}", token_len=10))
        seq.clear(ratio=0.0)

        # Should be able to append again
        seq.append(StateChunk(chunk="new_chunk", token_len=15))

        assert seq.get_num_state_chunks() == 1
        assert len(seq) == 15

    def test_workflow_example_from_docstring(self):
        """Test the example workflow from the class docstring."""
        seq = StateChunkSequence(100)
        chunk = StateChunk(chunk="example", token_len=10)

        # Append 3 chunks
        for _ in range(3):
            seq.append(chunk)

        assert seq.get_num_state_chunks() == 3
        assert len(seq) == 30

        # Clear to keep newest 50%
        seq.clear(0.5)

        # Should have 1 chunk remaining (floor(3 * 0.5) = 1)
        assert seq.get_num_state_chunks() == 1
        assert len(seq) == 10

    def test_clear_on_empty_sequence(self):
        """Test that clearing an empty sequence doesn't cause errors."""
        seq = StateChunkSequence(max_length=100)

        # Should not raise any errors
        seq.clear(ratio=0.5)

        assert seq.get_num_state_chunks() == 0
        assert len(seq) == 0

    def test_max_capacity_workflow(self):
        """Test the recommended workflow for managing capacity."""
        max_tokens = 50
        seq = StateChunkSequence(max_length=max_tokens)

        # Simulate adding chunks and checking capacity
        for i in range(10):
            chunk = StateChunk(chunk=f"chunk{i}", token_len=5)

            # Check if we need to clear before appending
            if seq.get_num_state_chunks() >= max_tokens:
                seq.clear(0.5)

            seq.append(chunk)

        # Should have managed capacity without errors
        assert seq.get_num_state_chunks() <= max_tokens

    def test_chunks_with_zero_token_length(self):
        """Test handling chunks with zero token length."""
        seq = StateChunkSequence(max_length=100)

        seq.append(StateChunk(chunk="empty", token_len=0))
        seq.append(StateChunk(chunk="normal", token_len=10))
        seq.append(StateChunk(chunk="empty2", token_len=0))

        assert seq.get_num_state_chunks() == 3
        assert len(seq) == 10

    def test_clear_preserves_newest_elements(self):
        """Test that clear() keeps the newest (most recent) elements."""
        seq = StateChunkSequence(max_length=100)

        # Add chunks with identifiable token lengths
        for i in range(10):
            seq.append(StateChunk(chunk=f"chunk{i}", token_len=i + 1))

        # Total: 1+2+3+4+5+6+7+8+9+10 = 55
        assert len(seq) == 55

        # Keep newest 3 elements (ratio = 3/10 = 0.3)
        seq.clear(ratio=0.3)

        # Should keep chunks 7, 8, 9 with token_len 8, 9, 10
        assert seq.get_num_state_chunks() == 3
        assert len(seq) == 27  # 8 + 9 + 10

    def test_large_sequence(self):
        """Test with a larger sequence to verify performance."""
        seq = StateChunkSequence(max_length=10000)

        # Add 1000 chunks
        for i in range(1000):
            seq.append(StateChunk(chunk=f"chunk{i}", token_len=10))

        assert seq.get_num_state_chunks() == 1000
        assert len(seq) == 10000

        # Clear to 10%
        seq.clear(ratio=0.1)

        assert seq.get_num_state_chunks() == 100
        assert len(seq) == 1000
