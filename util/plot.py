from matplotlib import pyplot as plt
from IPython import display


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    """Plot x and log(y)."""
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()

def multiple_semilogy(
        x_vals_list, y_val_list, x_label_list, y_label_list, x2_vals_list=None, y2_vals_list=None,
        legend_list=None, figsize=(12, 12)
):
    assert len(x_vals_list) == len(y_val_list)
    assert len(x_vals_list) == len(x_label_list)
    assert len(x_vals_list) == len(y_label_list)
    set_figsize(figsize)
    fig_num = len(x_vals_list)
    width = fig_num ** 0.5
    if int(width) < width:
        width += 1
    plt.figure()
    for i in range(len(x_vals_list)):
        # row_num = i / width
        # col_num = i % width
        ax = plt.subplot(width, width, 1 + i)
        ax.set_xlabel(x_label_list[i])
        ax.set_ylabel(y_label_list[i])
        ax.semilogy(x_vals_list[i], y_val_list[i])
        if x2_vals_list and y2_vals_list:
            ax.semilogy(x2_vals_list[i], y2_vals_list[i], linestyle=':')
            plt.legend(legend_list[i])
    # plt.subplots_adjust(wspace=1, hspace=1)
    plt.tight_layout()  # 设置默认的间距
    plt.show()

def set_figsize(figsize=(3.5, 2.5)):
    """Set matplotlib figure size."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')