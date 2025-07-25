import os
import numpy as np
import pandas as pd
import h2o
from h2o.estimators import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch

# 1. Cargar los datos
try:
    data = np.load("subset_vmic10.npy")
    y = np.load("y_subset_vmic10.npy")
    print("Datos cargados correctamente:")
    print(f"Forma de data: {data.shape}")
    print(f"Forma de y: {y.shape}")
except Exception as e:
    print(f"Error al cargar archivos: {e}")
    raise

# 2. Seleccionar variables objetivo
y_selected = y[:, [0, 1, 7]]  # TEFF, LOGG, LOGMDOT

# 3. Inicializar H2O con configuración para cluster
h2o.init(
    nthreads=8,              # Usar los 8 cores disponibles
    max_mem_size="6G",       # Dejar algo de memoria para el sistema
    ignore_config=True,      # Ignorar configuraciones locales
    start_h2o=True           # Forzar inicio nuevo
)

# 4. Crear DataFrame combinado
df = pd.DataFrame(
    data=np.hstack([data, y_selected]),
    columns=[f"f{i}" for i in range(data.shape[1])] + ["TEFF", "LOGG", "LOGMDOT"]
)
h2o_df = h2o.H2OFrame(df)

# 5. Definir predictores y respuestas
predictors = [f"f{i}" for i in range(data.shape[1])]
responses = ["TEFF", "LOGG", "LOGMDOT"]

# 6. Configuración de hiperparámetros (quitando stopping_metric)
hyper_params = {
    "hidden": [
        [64, 64],
        [128, 64, 32],
        [200, 100, 50]
    ],
    "activation": ["Rectifier", "Tanh"],
    "input_dropout_ratio": [0.1, 0.2],
    "l1": [1e-5, 1e-4],
    "epochs": [50, 100]
}

search_criteria = {
    "strategy": "RandomDiscrete",
    "max_models": 10,
    "seed": 42,
    "stopping_rounds": 5,
    "stopping_tolerance": 0.001
}

# 7. Función para entrenar modelos
def train_model(target):
    print(f"\nEntrenando modelo para {target}...")

    # Configuración base del modelo (con stopping_metric aquí)
    model = H2ODeepLearningEstimator(
        distribution="gaussian",
        nfolds=3,
        stopping_metric="RMSE",  # Definido solo aquí
        stopping_tolerance=0.01,
        stopping_rounds=5,
        seed=42,
        variable_importances=True
    )

    grid = H2OGridSearch(
        model=model,
        grid_id=f"grid_{target}",
        hyper_params=hyper_params,
        search_criteria=search_criteria
    )

    grid.train(x=predictors, y=target, training_frame=h2o_df)
    best_model = grid.get_grid(sort_by="RMSE", decreasing=False).models[0]

    # Guardar modelo
    model_path = h2o.save_model(best_model, path=f"model_{target}", force=True)
    print(f"Modelo para {target} guardado en: {model_path}")

    return best_model

# 8. Entrenar modelos
try:
    models = {target: train_model(target) for target in responses}

    # Mostrar resultados
    print("\nResultados finales:")
    for target, model in models.items():
        perf = model.model_performance()
        print(f"\n{target}:")
        print(f"- RMSE: {perf.rmse():.4f}")
        print(f"- Arquitectura: {model.params['hidden']['actual']}")
        print(f"- Activación: {model.params['activation']['actual']}")
        print(f"- Épocas: {model.params['epochs']['actual']}")

except Exception as e:
    print(f"\nError durante el entrenamiento: {e}")
finally:
    # Cerrar H2O
    h2o.cluster().shutdown()
