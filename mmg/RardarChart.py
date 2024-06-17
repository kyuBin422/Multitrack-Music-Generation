import re
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def string2num(input_strings):
    all_extracted_numbers = []
    for input_string in input_strings:
        # Regular expression to find all numbers in the string
        numbers = re.findall(r"[\d\.]+", input_string)

        # Convert extracted strings to floats and store in a list
        extracted_numbers = [float(num) for num in numbers]
        all_extracted_numbers.append(extracted_numbers)
    return np.array(all_extracted_numbers)


def array2loss(array):
    array = array / np.sum(array, 1).reshape(-1, 1)
    array = np.mean(array, 0)
    return array


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


if __name__ == '__main__':
    input_string_transformer = [
        'Individual losses: type=0.0013, beat: 0.3053, position: 0.1966, pitch: 0.5547, duration: 0.3634, instrument: 0.3895timesignaturenumerator: 0.0349timesignaturedenominator: 0.0096tempo: 0.0233velocity: 0.1168',
        'Individual losses: type=0.0020, beat: 0.7136, position: 0.2434, pitch: 0.5861, duration: 0.4088, instrument: 0.4460timesignaturenumerator: 0.0636timesignaturedenominator: 0.0224tempo: 0.1653velocity: 0.1506',
        'Individual losses: type=0.0013, beat: 0.6640, position: 0.2536, pitch: 0.6235, duration: 0.4122, instrument: 0.4460timesignaturenumerator: 0.0532timesignaturedenominator: 0.0163tempo: 0.0142velocity: 0.1572',
        'Individual losses: type=0.0013, beat: 0.6788, position: 0.2423, pitch: 0.5933, duration: 0.4041, instrument: 0.4405timesignaturenumerator: 0.0511timesignaturedenominator: 0.0159tempo: 0.0112velocity: 0.1500']
    input_string_transformer = string2num(input_string_transformer)
    input_string_mamba = [
        'Individual losses: type=0.0012, beat: 0.1224, position: 0.1831, pitch: 0.5366, duration: 0.3388, instrument: 0.3693timesignaturenumerator: 0.0295timesignaturedenominator: 0.0144tempo: 0.0081velocity: 0.1113',
        'Individual losses: type=0.0009, beat: 0.0793, position: 0.1701, pitch: 0.4969, duration: 0.3215, instrument: 0.3431timesignaturenumerator: 0.0130timesignaturedenominator: 0.0035tempo: 0.0019velocity: 0.1022',
        'Individual losses: type=0.0009, beat: 0.0686, position: 0.1621, pitch: 0.4704, duration: 0.3127, instrument: 0.3203timesignaturenumerator: 0.0078timesignaturedenominator: 0.0027tempo: 0.0023velocity: 0.0994',
        'Individual losses: type=0.0011, beat: 0.0611, position: 0.1577, pitch: 0.4561, duration: 0.3140, instrument: 0.3027timesignaturenumerator: 0.0093timesignaturedenominator: 0.0025tempo: 0.0103velocity: 0.1088']
    input_string_mamba = string2num(input_string_mamba)
    input_string_retnet = [
        'Individual losses: type=0.0005, beat: 0.1309, position: 0.1757, pitch: 0.5338, duration: 0.3347, instrument: 0.3746timesignaturenumerator: 0.1818timesignaturedenominator: 0.0144tempo: 0.0068velocity: 0.1107',
        'Individual losses: type=0.0003, beat: 0.0773, position: 0.1729, pitch: 0.5146, duration: 0.3290, instrument: 0.3523timesignaturenumerator: 0.0095timesignaturedenominator: 0.0034tempo: 0.0038velocity: 0.1094',
        'Individual losses: type=0.0006, beat: 0.6675, position: 0.2401, pitch: 0.5775, duration: 0.4143, instrument: 0.4405timesignaturenumerator: 0.0451timesignaturedenominator: 0.0087tempo: 0.0246velocity: 0.1327',
        'Individual losses: type=0.0004, beat: 0.6172, position: 0.2371, pitch: 0.5931, duration: 0.3933, instrument: 0.4340timesignaturenumerator: 0.0522timesignaturedenominator: 0.0255tempo: 0.0363velocity: 0.1370']
    input_string_retnet = string2num(input_string_retnet)
    input_string_transformer_single = [
        'Individual losses: type=0.0024, beat: 0.0743, position: 0.1493, pitch: 0.4557, duration: 0.3047, instrument: 0.2589timesignaturenumerator: 0.0164timesignaturedenominator: 0.0112tempo: 0.0068velocity: 0.1056',
        'Individual losses: type=0.0003, beat: 0.0566, position: 0.1341, pitch: 0.4190, duration: 0.2909, instrument: 0.2182timesignaturenumerator: 0.0066timesignaturedenominator: 0.0031tempo: 0.0309velocity: 0.0993',
        'Individual losses: type=0.0003, beat: 0.0514, position: 0.1274, pitch: 0.3944, duration: 0.2756, instrument: 0.1914timesignaturenumerator: 0.0043timesignaturedenominator: 0.0012tempo: 0.0051velocity: 0.0907',
        'Individual losses: type=0.0003, beat: 0.0461, position: 0.1160, pitch: 0.3746, duration: 0.2664, instrument: 0.1723timesignaturenumerator: 0.0030timesignaturedenominator: 0.0008tempo: 0.0020velocity: 0.0859']
    input_string_transformer_single = string2num(input_string_transformer_single)
    input_string_mamba_single = [
        'Individual losses: type=0.0013, beat: 0.0548, position: 0.1283, pitch: 0.4091, duration: 0.2861, instrument: 0.2091timesignaturenumerator: 0.0040timesignaturedenominator: 0.0026tempo: 0.0031velocity: 0.0946',
        'Individual losses: type=0.0012, beat: 0.0433, position: 0.1072, pitch: 0.3460, duration: 0.2592, instrument: 0.1659timesignaturenumerator: 0.0037timesignaturedenominator: 0.0022tempo: 0.0027velocity: 0.0858',
        'Individual losses: type=0.0010, beat: 0.0357, position: 0.0914, pitch: 0.3065, duration: 0.2408, instrument: 0.1360timesignaturenumerator: 0.0024timesignaturedenominator: 0.0013tempo: 0.0022velocity: 0.0798',
        'Individual losses: type=0.0010, beat: 0.0318, position: 0.0831, pitch: 0.2824, duration: 0.2271, instrument: 0.1152timesignaturenumerator: 0.0021timesignaturedenominator: 0.0012tempo: 0.0024velocity: 0.0757']
    input_string_mamba_single = string2num(input_string_mamba_single)
    input_string_retnet_single = [
        'Individual losses: type=0.0005, beat: 0.0553, position: 0.1263, pitch: 0.4010, duration: 0.2795, instrument: 0.1943timesignaturenumerator: 0.0031timesignaturedenominator: 0.0012tempo: 0.0039velocity: 0.0919',
        'Individual losses: type=0.0003, beat: 0.0461, position: 0.1104, pitch: 0.3450, duration: 0.2615, instrument: 0.1596timesignaturenumerator: 0.0039timesignaturedenominator: 0.0009tempo: 0.0015velocity: 0.0849',
        'Individual losses: type=0.0003, beat: 0.0401, position: 0.0975, pitch: 0.3156, duration: 0.2457, instrument: 0.1402timesignaturenumerator: 0.0020timesignaturedenominator: 0.0006tempo: 0.0013velocity: 0.0805',
        'Individual losses: type=0.0002, beat: 0.0358, position: 0.0877, pitch: 0.2907, duration: 0.2335, instrument: 0.1231timesignaturenumerator: 0.0016timesignaturedenominator: 0.0005tempo: 0.0011velocity: 0.0773']
    input_string_retnet_single = string2num(input_string_retnet_single)

    input_string_transformer, input_string_retnet, input_string_mamba, input_string_transformer_single, input_string_retnet_single, input_string_mamba_single = array2loss(
        input_string_transformer), array2loss(input_string_retnet), array2loss(input_string_mamba), array2loss(
        input_string_transformer_single), array2loss(input_string_retnet_single), array2loss(input_string_mamba_single)

    data = [[input_string_transformer, input_string_transformer_single],
            [input_string_retnet, input_string_retnet_single], [input_string_mamba, input_string_mamba_single]]
    N = 10
    theta = radar_factory(N, frame='polygon')

    spoke_labels = ['type', 'beat', 'position', 'pitch', 'duration', 'instrument', 'numerator', 'denominator', 'tempo',
                    'velocity']

    fig, axs = plt.subplots(figsize=(9, 9), nrows=1, ncols=3,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    fig.tight_layout()
    colors = ['b', 'r']
    titles = ['transformer-oriented model', 'retnet-oriented model', 'mamba-oriented model']
    # Plot the four cases from the example data on separate axes
    for ax, case_data, title in zip(axs.flat, data, titles):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    labels = ('multi-task learning', 'single')
    legend = axs[0].legend(labels, loc=(0.9, 0),
                              labelspacing=0.1, fontsize='small')

    plt.savefig('radar.png')
