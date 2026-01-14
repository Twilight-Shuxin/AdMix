import matplotlib.pyplot as plt
from enum import Enum
from plotting.constants import color_list, line_style_list

"""
    Trace plot data with format:
        label: Name of data to be plotted
        data = [(timestamp_1, data_1), (timestamp_2, data_2) ...]
"""
class TracePlotData():
    def __init__(self, label, data):
        self.label = label
        self.x, self.y = zip(*data)


"""
    Trace axis information including:
        label: Name of this axis; if not provided default is the label of first given TracePlotData
        plot_type: LINE / BAR (HACK: TracePlotData plotted on same axis is forced to be plotted as either bar / line)
        plot_data_list: list of TracePlotData to be plotted on this axis
        y_tag: A tuple describing this axis: (name, unit) 
"""
class TraceAxisData():
    class PlotType(Enum):
        LINE = 1
        BAR = 2

    def __init__(self, *plot_data_list, label = None, y_tag, plot_type = PlotType.LINE):
        self.label = label
        self.plot_data_list = list(plot_data_list)
        self.y_tag_name, self.y_tag_unit = y_tag
        self.plot_type = plot_type
        if((not label) and self.plot_data_list):
            self.label = self.plot_data_list[0].label
    
    def append(self, plot_data):
        self.plot_data_list.append(plot_data)
        
"""
    Plot trace graph with list of TraceAxisData, save to 'plot_file'.
"""
def plot_trace_graph(plot_file, *axis_list):
    fig, ax = plt.subplots(figsize=(10, 5))
    titles = []
    handles = []
    labels = []
    ax.set_xlabel('Time (s)')
    idy = 0

    for idx, axis_data in enumerate(axis_list):
        for plot_data in axis_data.plot_data_list:
            match axis_data.plot_type:
                case TraceAxisData.PlotType.LINE:
                    ax.plot(plot_data.x, plot_data.y, linestyle = line_style_list[idy], color = color_list[idy], label = plot_data.label)
                case TraceAxisData.PlotType.BAR: 
                    ax.bar(plot_data.x, plot_data.y, label = plot_data.label, color = color_list[idy], alpha=0.5)
            idy += 1
        
        y_tag_name = axis_data.y_tag_name
        y_tag_unit = axis_data.y_tag_unit
        ax.set_ylabel((y_tag_name + ' (' + y_tag_unit + ')') if(y_tag_unit) else y_tag_name)
        ax.tick_params(axis='y')
        titles.append(axis_data.label)

        h, l = ax.get_legend_handles_labels()
        handles = handles + h
        labels = labels + l
        if(not idx):
            ax.grid(True)
        if idx < len(axis_list) - 1:
            ax = ax.twinx()

    ax.legend(handles, labels, loc="upper right")
    plt.title(' & '.join(titles))
    fig.tight_layout()
    plt.savefig(plot_file)