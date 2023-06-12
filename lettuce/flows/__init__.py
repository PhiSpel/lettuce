"""
Example flows.
"""

from lettuce.flows.taylorgreen import TaylorGreenVortex2D, TaylorGreenVortex3D
from lettuce.flows.couette import CouetteFlow2D
from lettuce.flows.poiseuille import PoiseuilleFlow2D
from lettuce.flows.doublyshear import DoublyPeriodicShear2D
from lettuce.flows.decayingturbulence import DecayingTurbulence
# from lettuce.flows.turbulentmixing import TurbulentMixing2D, TurbulentMixing3D
from lettuce.flows.cavity import Cavity2D
from lettuce.flows.mixinglayer import MixingLayer3D
from lettuce.flows.obstacle import Obstacle, Obstacle2D, Obstacle3D
from lettuce.stencils import D2Q9, D3Q19, D3Q27

flow_by_name = {
    "taylor2D": [TaylorGreenVortex2D, D2Q9],
    "taylor3D": [TaylorGreenVortex3D, D3Q19],
    "poiseuille2D": [PoiseuilleFlow2D, D2Q9],
    "shear2D": [DoublyPeriodicShear2D, D2Q9],
    "couette2D": [CouetteFlow2D, D2Q9],
    "decay": [DecayingTurbulence, D2Q9],
    "mixinglayer3D": [MixingLayer3D, D3Q27],
    "cavity": [Cavity2D, D2Q9]
}
