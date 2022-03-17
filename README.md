# MarmosetModel

## Citation request
RANIERI, C. M.; PIMENTEL, J. M. ; ROMANO, M. R. ; ELIAS, L. A. ; ROMERO, R. A. F. ; LONES, M. A. ; ARAUJO, M. F. P. ; VARGAS, PATRICIA A. ; MOIOLI, R. C. . A Data-Driven Biophysical Computational Model of Parkinson’s Disease Based on Marmoset Monkeys. IEEE Access, v. 9, p. 122548-122567, 2021.

## Docker

To build the Docker container:

```
cd docker

bash build-docker.sh
```

To run the Docker container:


```
bash run-docker.sh
```

If running for the first time, compile the ```.mod``` files by running:

```
nrnivmodl
```

To run a demo of the marmoset model:
```
python MarmosetBG.py
```

## Acknowledgement
This work is part of the Neuro4PD project, granted by Royal Society and Newton Fund (NAF\R2\180773), and São Paulo Research Foundation (FAPESP), grants 2017/02377-5 and 2018/25902-0. Moioli and Araujo acknowledge the support from the National Institute of Science and Technology, program Brain Machine Interface (INCT INCEMAQ) of the National Council for Scientific and Technological Development(CNPq/MCTI), the Rio Grande do Norte Research Foundation (FAPERN), the Coordination for the Improvement of Higher Education Personnel (CAPES), the Brazilian Innovation Agency (FINEP), and the Ministry of Education (MEC). Romano was the recipient of a master's scholarship from FAPESP, grant 2018/11075-5. Elias is funded by a CNPq Research Productivity Grant (314231/2020-0). This research was carried out using the computational resources from the Center for Mathematical Sciences Applied to Industry (CeMEAI) funded by FAPESP, grant 2013/07375-0.
Additional resources were provided by the Robotics Lab within the Edinburgh Centre for Robotics, and by the Nvidia Grants program.
