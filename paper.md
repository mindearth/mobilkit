---
title: 'Mobilkit: A Python Toolkit for Urban Resilience and Disaster Risk Management Analytics'
tags:
  - mobile phone data 
  - disaster resilience 
  - human mobility
  - geospatial analysis
authors:
  - name: Enrico Ubaldi
    orcid: 0000-0003-1685-9939
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Takahiro Yabe
    orcid: 0000-0001-8967-1967
    equal-contrib: true 
    corresponding: true 
    affiliation: 2
  - name: Nicholas Jones
    affiliation: 3
  - name: Maham Faisal Khan
    affiliation: 3
  - name: Alessandra Feliciotti
    orcid: 0000-0002-1471-5360
    affiliation: 1
  - name: Riccardo Di Clemente
    orcid: 0000-0001-8005-6351
    affiliation: "4, 5"
  - name: Satish V. Ukkusuri
    affiliation: 6
  - name: Emanuele Strano
    orcid: 0000-0002-2339-6824
    affiliation: 1
affiliations:
 - name: MindEarth, Switzerland
   index: 1
 - name: Massachusetts Institute of Technology, USA
   index: 2
 - name: The World Bank, USA
   index: 3
 - name: Complex Connections Lab, Network Science Institute, Northeastern University London, London, E1W 1LP, United Kingdom.
   index: 4
 - name: The Alan Turing Institute, London, NW12DB, United Kingdom.
   index: 5
 - name: Purdue University, USA
   index: 6
date: 12 February 2023
bibliography: paper.bib
---

# Summary

The availability of mobility data is increasing thanks to the widespread adoption 
of mobile phones and location-based services. This data generates powerful insights 
on people's mobility habits, with applications in areas such as health, migration, 
and poverty estimation. Yet despite the growing academic literature on the usage 
and application of mobile phone location data in this field and despite the raising 
awareness of the importance of  disaster preparedness and response and climate change 
resilience, large-scale mobility data remain under-utilized in real-world disaster 
management operations to this date [@barra2020solid].

At present, only few tools allow for an integrated and inclusive analysis of mobility data. 
While several tookits allow users to perform some basic analytics on large mobility datasets
(e.g., [@de2016bandicoot] or [@pappalardo2019scikitmobility]), these cover only some of the
steps in the mobility data pipeline.
These toolkits also do not provide adequate data pre-processing and visualization
functionality which causes users to seek additional external options.
Also, there is a lack of clear documentation 
to enable policymakers and planners to understand the analytics process, outputs, and 
potential questions that mobility data can answer, particularly in the context of post-disaster assessment.


# Statement of need

`Mobilkit` is an open-source Python software toolkit that enables policy makers 
to conduct post-disaster assessment using large-scale mobility data. The toolkit 
allows the user to conduct pre-processing of data, validation of the data 
representativeness, home and office location estimation, post-disaster displacement analysis, 
and point-of-interest visit analysis. The purpose of [`Mobilkit`](https://github.com/mindearth/mobilkit) is to provide urban planners, 
disaster policy makers, and researchers an easy-to-use and practical toolkit to visualize, 
analyze, and monitor post-disaster disruption and recovery. The software is freely-available 
on GitHub along with online documentation and Jupyter Notebooks that provide step-by-step tutorials.

`Mobilkit` allows the user to 1) pre-process the dataset to select users who have sufficient amount of observations,
2) evaluate the representativeness of the mobility data by combining with census population statistics, 
3) conduct post-disaster displacement and recovery analysis, 4) estimate the recovery of businesses 
and social services by using point-of-interest (POI) data, and 5) measure and characterize the spatial structure of cities. 

# Use Case
The usefulness of `Mobilkit` was demonstrated in a recent study carried out in collaboration with the World Bank Global Facility for Disaster Reduction and Recovery [@yabe2021location]. The study focused on assessing the impact of a 7.1 magnitude earthquake that occurred on September 19, 2017 where the epicenter was located around 55 km south of Puebla, Mexico (about 100 km south-east of Mexico City, Mexico). `Mobilkit` was also leveraged to conduct an analysis of the spatial structure of ten cities around the globe using smartphone location data, provided by Quadrant, to generate insights about mobility management options[^1]. Similar analysis could also be explored using `Mobilkit` for planning and recovering activities related to climate, man-made, and other natural disasters.

[^1] See the notebooks covering [Urban Spatial Structure analyses](https://mobilkit.readthedocs.io/en/latest/examples/USS01_Mumbai.html) and an [inter-city comparison of Urban Spatial Structure indicators](https://mobilkit.readthedocs.io/en/latest/examples/USS02_CityComparison.html).

# Acknowledgements

We extend our sincere gratitude to Cuebiq and Quadrant for providing the data to support this effort. 
This work was supported by the Spanish Fund for Latin America and the Caribbean (SFLAC) under the 
Disruptive Technologies for Development Program at the World Bank and by the Global Facility for Disaster 
Reduction and Recovery (GFDRR - USAID Single Donor Trust Fund). The findings, interpretations, and 
conclusions expressed in this paper are entirely those of the authors. They do not necessarily represent 
the views of the International Bank for Reconstruction and Development/World Bank and its affiliated 
organizations, or those of the Executive Directors of the World Bank or the governments they represent.

# References
