---
title: 'Mobilkit: A Python Toolkit for Urban Resilience and Disaster Risk Management Analytics'
tags:
  - mobile phone data 
  - disaster resilience 
  - human mobility
  - geospatial analysis
authors:
  - name: Enrico Ubaldi
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
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
    affiliation: 1
  - name: Riccardo Di Clemente
    affiliation: "4, 5"
  - name: Satish V. Ukkusuri
    affiliation: 6
  - name: Emanuele Strano
    affiliation: 1
affiliations:
 - name: MindEarth, Switzerland
   index: 1
 - name: Massachusetts Institute of Technology, USA
   index: 2
 - name: The World Bank, USA
   index: 3
 - name: University of Exeter, UK
   index: 4
 - name: The Alan Turing Institute, UK
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
and poverty estimation. Yet, despite the growing academic literature on the usage 
and application of mobile phone location data in this field and despite the raising 
awareness of the importance of  disaster preparedness and response and climate change 
resilience, large-scale mobility data remain under-utilized in real-world disaster 
management operations to this date [@barra2020solid].

At present, only few tools allow for an integrated and inclusive analysis of mobility data. 
While toolkits as [@de2016bandicoot] or [@pappalardo2019scikitmobility] allow to perform some basic 
analytics on large mobility datasets, these cover only some of the steps in the mobility data 
analysis pipeline. Very often you have to go fishing for functions/tools from other libraries 
to cover for data pre-processing or visualization. Also, there is a lack of clear documentation 
that enables policymakers and planners to understand the analytics process, outputs, and 
potential questions that mobility data can answer, particularly in the context of post-disaster assessment.


# Statement of need

`Mobilkit` is an open-source Python software toolkit that enables policy makers 
to conduct post-disaster assessment using large-scale mobility data. The toolkit 
allows the user to conduct pre-processing of data, validation of the data 
representativeness, home and office location estimation, post-disaster displacement analysis, 
and point-of-interest visit analysis. The purpose of `Mobilkit` is to provide urban planners, 
disaster policy makers and researchers an easy-to-use and practical toolkit to visualize, 
analyze, and monitor post-disaster disruption and recovery. The software is freely-available 
on GitHub along with online documentation and Jupyter Notebooks that provides step-by-step tutorials.

`Mobilkit` allows the user to 1) pre-process the dataset to select users who have sufficient amount of observations,
2) evaluate the representativeness of the mobility data by combining with census population statistics, 
3) conduct post-disaster displacement and recovery analysis, 4) estimate the recovery of businesses 
and social services by using point-of-interest (POI) data, and 5) measure and characterize the spatial structure of cities. 

The functionality of `Mobilkit` is showcased using the outcomes of a project carried out in collaboration with the World Bank GFDRR aimed at assessing the impact on the population of the 7.1 magnitude earthquake with the epicenter located around 55km south of Puebla (about 100km south-east of Mexico City) occurred on  19 September 2017, using smartphone location data collected for Mexico before and after the earthquake. Methods regarding the spatial structure of cities are demonstrated using smartphone location data provided by Quadrant that cover ten different cities around the globe in March 2022. These use cases showcase the immense potential of using mobile phone location data and `Mobilkit` for planning and recovering from climate-related, man-made, and natural disasters.

# Acknowledgements

We extend our sincere gratitude to Cuebiq and Quadrant for providing the data to support this effort. 
This work was supported by the Spanish Fund for Latin America and the Caribbean (SFLAC) under the 
Disruptive Technologies for Development Program at the World Bank and by the Global Facility for Disaster 
Reduction and Recovery (GFDRR - USAID Single Donor Trust Fund). The findings, interpretations, and 
conclusions expressed in this paper are entirely those of the authors. They do not necessarily represent 
the views of the International Bank for Reconstruction and Development/World Bank and its affiliated 
organizations, or those of the Executive Directors of the World Bank or the governments they represent.

# References