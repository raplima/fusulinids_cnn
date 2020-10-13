# Transfer learning and convolutional neural networks for the classification of microfossils
This repository contains Python scripts used to make transfer learning analysis of fusulinid images:

- [evaluate_cnn.py](./evaluate_cnn.py): script to evaluate the CNN models trained by fit_models_experiments.py  
- [file_manag.py](./file_manag.py): data organization and data augmentation    
- [fit_models_experiments.py](./fit_models_experiments.py): fit different CNN models using different training techniques  
- [helper.py](./helper.py): helper functions to evaluate memory usage  
- [plots.py](./plots.py): create some figures helpful for analysis 

## References
The scripts in this repo were used for the analysis of the paper publised in [PALAIOS](https://pubs-geoscienceworld-org.ezproxy.lib.ou.edu/palaios) as 
[Convolutional neural networks as an aid to biostratigraphy and micropaleontology: a test on late Paleozoic microfossils](https://pubs.geoscienceworld.org/sepm/palaios/article-abstract/35/9/391/591723/CONVOLUTIONAL-NEURAL-NETWORKS-AS-AN-AID-TO?redirectedFrom=fulltext):
```bibtex
@article{10.2110/palo.2019.102,
    author = {PIRES DE LIMA, RAFAEL and WELCH, KATIE F. and BARRICK, JAMES E. and MARFURT, KURT J. and BURKHALTER, ROGER and CASSEL, MURPHY and SOREGHAN, GERILYN S.},
    title = "{CONVOLUTIONAL NEURAL NETWORKS AS AN AID TO BIOSTRATIGRAPHY AND MICROPALEONTOLOGY: A TEST ON LATE PALEOZOIC MICROFOSSILS}",
    journal = {PALAIOS},
    volume = {35},
    number = {9},
    pages = {391-402},
    year = {2020},
    month = {10},
    issn = {0883-1351},
    doi = {10.2110/palo.2019.102},
    url = {https://doi.org/10.2110/palo.2019.102},
    eprint = {https://pubs.geoscienceworld.org/palaios/article-pdf/35/9/391/5164786/i0883-1351-35-9-391.pdf},
}
```
We would be happy if you reference the paper in case you use something in this repo in your research. 
