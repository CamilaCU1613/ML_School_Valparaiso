#!/bin/bash

# ###### Zona de ParÃ¡metros de solicitud de recursos a SLURM #################$
#
#SBATCH --job-name=CNN          #Nombre del job
#SBATCH -p medium                      #Cola a usar, Default=short (Ver colas y$
#SBATCH -N 1                            #Nodos requeridos, Default=1
#SBATCH -n 1                            #Tasks paralelos, recomendado para MPI,$
#SBATCH --cpus-per-task=8               #Cores requeridos por task, recomendado$
#SBATCH --mem=262000              #Memoria en Mb por CPU, Default=2048
#SBATCH --time=10-00:00:00                 #Tiempo mÃ¡ximo de corrida, Default=$
#SBATCH --mail-user=c.cardenasu@uniandes.edu.co
#SBATCH --mail-type=ALL
#SBATCH -o CNN.o%j                 #Nombre de archivo de salida
#
###############################################################################$

# ############################### Zona Carga de Modulos #######################$
module load anaconda/python3.9
source activate Tesis3
pip install scikit-learn
pip install astroalign
pip install h2o
#pip install matplotlib
###############################################################################$


# ###### Zona de Ejecucion de codigo y comandos a ejecutar secuencialmente ####$
host=`/bin/hostname`
date=`/bin/hostname`
echo "Soy un JOB de prueba"
echo "Corri en la maquina: "$host
echo "Corri el: "$date

echo -e "Ejecutando Script de python \n"
#python Cluster_b7978.py
conda activate Tesis3
python CNN1.py
echo -e "Finalice la ejecucion del script \n"
###############################################################################$


