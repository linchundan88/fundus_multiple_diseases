def input_norm(x):
    # if not isinstance(x, np.ndarray):
    #     x = np.asarray(x, dtype=np.float16)

    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def input_norm_reverse(x):
    x /= 2
    x += 0.5
    x *= 255.

    return x