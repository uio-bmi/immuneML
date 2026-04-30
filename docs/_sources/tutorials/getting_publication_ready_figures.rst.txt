How to get publication-ready figures
========================================

When running analyses in immuneML, figures are automatically exported as interactive HTML files.
For many use cases, this is sufficient, but for publications, a static format such as PDF might be needed.
immuneML uses Plotly under the hood and each figure is also exported as a Plotly JSON file. This JSON
file can be loaded and converted to high-quality publication-ready figures.

1. Locate the exported JSON file
------------------------------------

Alognside the HTML figure, immuneML stores the corresponding `.json` file, for example:

.. code-block::

  results/
    my_instruction/
      reports/
        roc_curve_summary/
          ROC_curves_for_label_disease.html
          ROC_curves_for_label_disease.json <---

2. Install necessary packages
------------------------------------------------------------

Make sure you are in the virtual environment where `plotly` and `kaleido` packages are installed.
If not installed, install them using pip:

.. code-block:: bash

  pip install plotly
  pip install kaleido


3. Load the figure in Python and export in desired format
------------------------------------------------------------

Load the figure in python:

.. code-block:: python

  import json
  import plotly.io as pio

  with open("ROC_curves_for_label_disease.json", 'r') as file:
    fig_dict = json.load(file)

  fig = pio.from_json(json.dumps(fig_dict))
  fig.show() # or do any modifications to the figure, eg change colors

  # choose any one of these formats:
  fig.write_image('roc_curves.pdf')
  fig.write_image('roc_curves.svg')
  fig.write_image('roc_curves.png')

  # alternatively, adjust size and resolution
  fig.write_image('roc_curves.pdf', width=800, height=600, scale=2)

