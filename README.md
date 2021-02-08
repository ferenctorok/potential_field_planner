# Single Minimum Gradient Field Based Controller

## Install environment
`conda env create --prefix conda_env -f environment.yml`

`python setup.py develop`


## Description

Prototype implementation of a local minima free gradient field based controller in python. The gradient field combines an attractor and a repulsive field to compose the combined gradient field that by definition only has a single minima at the goal. This gradient field can then be controlled by a sufficiently designed controller, for which we also provide an example implementation.

The attractor field:

<img src="fig/attractor.png" alt="Attractor Gradient Field" width="600"/>
<br/><br/>

The repulsive field:

<img src="fig/repulsive.png" alt="Repulsive Gradient Field" width="600"/>
<br/><br/>

The combined field:

<img src="fig/combined.png" alt="Combined Gradient Field" width="600"/>

## More Descreption is coming soon.
