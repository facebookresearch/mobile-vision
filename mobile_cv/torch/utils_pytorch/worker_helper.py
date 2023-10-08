def compute_chunk_indices(rank_idx: int, total_ranks: int, total_elements: int):
    """
    Compute the start and end indices for a given rank.

    :param rank_idx: Current rank index.
    :param total_ranks: Total number of ranks.
    :param total_elements: Total number of elements in the list.
    :return: (start_idx, end_idx) for the given rank.
    """

    # Calculate base chunk size and the number of chunks that should be 1 larger.
    base_chunk_size, remainder = divmod(total_elements, total_ranks)

    if rank_idx < remainder:
        # This rank has one of the larger chunks.
        start_idx = rank_idx * (base_chunk_size + 1)
        end_idx = start_idx + base_chunk_size + 1
    else:
        # This rank has a base-size chunk.
        start_idx = rank_idx * base_chunk_size + remainder
        end_idx = start_idx + base_chunk_size

    return start_idx, end_idx
