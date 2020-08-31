"""
Utility for plotting the non-nested versus nested cross-validation.
"""

# Author: Alex Wozniakowski
# License: MIT

import matplotlib.pyplot as plt


def plot_cv_comparison(non_nested_scores, nested_scores, n_trials, fontsize='14',
                       save_plot=False, path=None):
    """Generate plots that illustrate nested versus non-nested cross-validation."""

    score_difference = non_nested_scores - nested_scores
    mean = score_difference.mean()
    std = score_difference.std()

    print(f'Average difference of {mean:.6f} with standard deviation of {std:.6f}.')

    # Plot the nested and non-nested cross-validation
    # scores for each random trial.
    plt.figure()
    plt.subplot(211)
    non_nested_scores_line, = plt.plot(non_nested_scores, color='r')
    nested_line, = plt.plot(nested_scores, color='b')
    plt.ylabel('Score', fontsize=fontsize)
    plt.legend([non_nested_scores_line, nested_line], ['Non-nested', 'Nested'],
               loc='best')
    plt.title('Non-nested versus nested cross-validation',
              x=.5, y=1.1, fontsize=fontsize)

    # Plot the scoring difference, where a positive difference
    # implies that non_nested_scores > nested_scores.
    plt.subplot(212)
    difference_plot = plt.bar(range(n_trials), score_difference)
    plt.xlabel('Trial')
    plt.ylabel('Score difference', fontsize=fontsize)
    plt.legend([difference_plot], ['Non-nested - Nested'], loc='best')

    if save_plot:
        assert path is not None and isinstance(path, str)
        plt.savefig(path)
    else:
        plt.show(block=True)
