"""Equivalent Viterbi for the sparse transition matrix used by librosa pYIN."""

import numpy as np
from numba import njit


@njit(cache=True)
def decode(log_prob, offsets, predecessors, weights, zero_weight, prior):
    frames, states = log_prob.shape
    back = np.empty((frames, states), dtype=np.uint16)
    previous = log_prob[0] + prior
    current = np.empty(states, dtype=np.float64)
    for t in range(1, frames):
        best_state = previous.argmax()
        zero_score = previous[best_state] + zero_weight
        for dest in range(states):
            choice, score = best_state, zero_score
            for edge in range(offsets[dest], offsets[dest + 1]):
                state = predecessors[edge]
                candidate = previous[state] + weights[edge]
                if candidate > score or (candidate == score and state < choice):
                    choice, score = state, candidate
            back[t, dest] = choice
            current[dest] = log_prob[t, dest] + score
        previous, current = current, previous
    path = np.empty(frames, dtype=np.uint16)
    path[-1] = previous.argmax()
    for t in range(frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def install():
    import librosa.sequence

    original = librosa.sequence.viterbi
    cached_transition = None
    cached_epsilon = None
    sparse = None

    def viterbi(prob, transition, *, p_init=None, return_logp=False):
        nonlocal cached_transition, cached_epsilon, sparse
        mono_axis = prob.ndim == 3 and prob.shape[0] == 1
        if (prob.ndim != 2 and not mono_axis) or p_init is None or return_logp or prob.shape[-2] > 65535:
            return original(prob, transition, p_init=p_init, return_logp=return_logp)
        if mono_axis:
            prob = prob[0]
        epsilon = np.finfo(prob.dtype).tiny
        if cached_epsilon != epsilon or not np.array_equal(cached_transition, transition):
            dest, predecessors = np.nonzero(transition.T)
            offsets = np.r_[0, np.cumsum(np.bincount(dest, minlength=prob.shape[0]))]
            weights = np.log(transition[predecessors, dest] + epsilon)
            cached_transition, cached_epsilon = transition.copy(), epsilon
            sparse = offsets, predecessors, weights
        path = decode(np.ascontiguousarray(np.log(prob.T + epsilon)), *sparse, np.log(epsilon),
                      np.log(p_init + epsilon))
        return path[None] if mono_axis else path

    librosa.sequence.viterbi = viterbi
