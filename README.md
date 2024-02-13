# pypt:
### A lightweight python library to compute cosmological phase transition information, from bubble nucleations to their gravitational wave (GW) signatures and the production of primordial black holes (PBH).

Requirements:
* numpy
* scipy
* gmpy2
* [cosmoTransitions](https://github.com/clwainwright/CosmoTransitions/tree/master)


References:
* M. Quiros, "Finite temperature field theory and phase transitions"
 [9901312]
* Lu, Kawana, Xie, "Old Phase Remnants in First-Order Phase Transitions" [2202.03439]
* Hooper, Krnjaic, McDermott, "Dark Radiation and Superheavy Dark Matter from Black Hole Domination" [1905.01301]


## Package overview
Define your own potential function (with 1-loop improved corrections and thermal corrections) that inherits from the ```VFT``` class. The ```VFT``` class has a method ```get_Tc()``` to search for the critical phase transition temperature of your potential.

One can then pass this potential into ```BubbleNucleation``` class via
```
bn = BubbleNucleation(my_eff_potential)
```
which can then compute the nucleation temperature T\* via the ```CosmoTransitions``` package. 


Primordial black hole (PBH) mass spectrum and their Hawking radiation spectra can be generated via the ```PBH``` class.

Gravitational waves (GW) can be generated via the ```Gravitational Wave``` class.


