import numpy as np
import matplotlib.pyplot as plt

def snippets(ts, snippet_size, num_snippets=2, window_size=None):
    ts = np.array(ts, dtype=float)
    time_series_len = len(ts)
    n = len(ts)

    if not isinstance(snippet_size, int) or snippet_size < 4:
        raise ValueError("snippet_size must be an integer >= 4")

    if n < (2 * snippet_size):
        raise ValueError("Time series is too short relative to snippet length")

    if not window_size:
        window_size = int(np.floor(snippet_size / 2))

    if window_size >= snippet_size:
        raise ValueError("window_size must be smaller than snippet_size")

    # Pad the end of the time series with zeros
    num_zeros = int(snippet_size * np.ceil(n / snippet_size) - n)
    ts = np.append(ts, np.zeros(num_zeros))

    # Compute all profiles
    indices = np.arange(0, len(ts) - snippet_size)
    distances = []

    for i in indices:
        snippet = ts[i : (i + snippet_size - 1)]
        distance = mpdist_vector(ts, snippet, int(window_size))
        distances.append(distance)

    distances = np.array(distances)

    # Find N snippets
    snippets = []
    minis = np.inf
    total_min = None
    for _ in range(num_snippets):
        minims = np.inf
        for i in range(len(indices)):
            s = np.sum(np.minimum(distances[i, :], minis))
            if minims > s:
                minims = s
                index = i

        minis = np.minimum(distances[index, :], minis)
        actual_index = indices[index]
        snippet = ts[actual_index : actual_index + snippet_size]
        snippet_distance = distances[index]
        snippets.append(
            {"index": actual_index, "snippet": snippet, "distance": snippet_distance}
        )

        if total_min is None:
            total_min = snippet_distance
        else:
            total_min = np.minimum(total_min, snippet_distance)

    # Compute the fraction of each snippet
    for snippet in snippets:
        mask = snippet["distance"] <= total_min
        arr = np.arange(len(mask))
        max_index = time_series_len - snippet_size
        snippet["neighbors"] = list(filter(lambda x: x <= max_index, arr[mask]))
        if max_index in snippet["neighbors"]:
            last_m_indices = list(range(max_index + 1, time_series_len))
            snippet["neighbors"].extend(last_m_indices)
        snippet["fraction"] = mask.sum() / (len(ts) - snippet_size)
        total_min = total_min - mask

    return snippets

def mpdist_vector(ts, snippet, window_size):
    snippet_len = len(snippet)
    ts_len = len(ts)
    distances = np.zeros(ts_len - snippet_len + 1)
    
    for i in range(ts_len - snippet_len + 1):
        subsequence = ts[i:i + snippet_len]
        distance = np.sum(np.abs(subsequence - snippet))
        distances[i] = distance
    
    return distances

def plot_snippets(ts, snippets):
    with plt.rc_context(
        {
            "lines.linewidth": 2,
            "font.family": "serif",
            "font.serif": "DejaVu Serif",
            "font.size": 36,
        }
    ):
        fig, (ax_main, ax_labels) = plt.subplots(
            2, figsize=(16, 6), gridspec_kw={"height_ratios": [16, 2]}
        )
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        ax_main.plot(ts, color="gray")

        labels = np.zeros_like(ts)
        for i, snippet in enumerate(snippets):
            color = color_cycle[i]

            neighbors = np.array(snippet["neighbors"])
            for neighbor_index in neighbors:
                labels[neighbor_index] = i

            snippet_start = snippet["index"]
            snippet_end = snippet_start + len(snippet["snippet"])
            ax_main.plot(
                np.arange(snippet_start, snippet_end),
                ts[snippet_start:snippet_end],
                c=color,
                label=f'Snippet {i}: {snippet["fraction"]:.2f}',
            )

        img = ax_labels.imshow([range(len(color_cycle))], cmap="tab10", aspect="auto")
        img.set_data([labels])

        ax_main.set_xlim(0, len(ts))
        ax_labels.axis("off")
        ax_main.legend(prop={"size": 16}, loc="upper right")
        plt.tight_layout()
        plt.show()
    return ax_main