def set_axis_attrs(ax, **kwargs):
    if kwargs.get("title"):
        ax.set_title(kwargs["title"])

    if kwargs.get("xlabel"):
        ax.set_xlabel(kwargs["xlabel"])

    if kwargs.get("ylabel"):
        ax.set_ylabel(kwargs["ylabel"])

    if kwargs.get("xlabel_size"):
        ax.xaxis.label.set_size(kwargs["xlabel_size"])

    if kwargs.get("ylabel_size"):
        ax.yaxis.label.set_size(kwargs["ylabel_size"])

    if kwargs.get("title_size"):
        ax.title.set_size(kwargs["title_size"])

    if kwargs.get("xlabel_color"):
        ax.xaxis.label.set_color(kwargs["xlabel_color"])

    if kwargs.get("ylabel_color"):
        ax.yaxis.label.set_color(kwargs["ylabel_color"])

    if kwargs.get("title_color"):
        ax.title.set_color(kwargs["title_color"])

    return ax
