# IAG-2024B-GenerativeQA

- https://medium.com/@ajazturki10/simplifying-language-understanding-a-beginners-guide-to-question-answering-with-t5-and-pytorch-253e0d6aac54
- https://www.youtube.com/watch?v=0tT5suZSdkA
- https://www.youtube.com/watch?v=r6XY80Z9eSA
- https://www.youtube.com/watch?v=dRUIGgNBvVk  this video seems the correct
- https://huggingface.co/docs/transformers/tasks/question_answer
- https://huggingface.co/learn/nlp-course/chapter7/7?fw=pt

El siguiente enlace a github es bastante descriptivo sobre el
procedimiento de entrenamiento del question answering

https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/question_answering.ipynb#scrollTo=rXuFTAzDIrJe

## Construcción del dataset con GPU local (ampliación)

Estos archivos amplían el dataset construido en `dataset_build.ipynb` (vía
Hugging Face Inference API + Meta-Llama-3-8B-Instruct) usando inferencia local
en GPU con un modelo distinto, para obtener más muestras de entrenamiento
(hipótesis de ablación: ¿mejora el fine-tuning de QA extractivo con más
datos?). El dataset original (`dataset/dataset_fge_squadM1.json`) se mantiene
como **baseline**; el nuevo dataset es un conjunto adicional para comparar.

### Archivos

- `scripts/local_qa_generation.py`: módulo compartido con la lógica de
  generación (`MODEL_REGISTRY` con 4 configuraciones — `qwen2.5-3b-instruct`,
  `qwen2.5-7b-instruct`, `gemma-3-1b-it`, `gemma-3-4b-it` —, carga de modelo
  4-bit con `transformers` + `bitsandbytes`, generación de JSON con
  `outlines`). Es importado por el notebook piloto, el script de ejecución
  completa y el script de benchmark.
- `dataset_build_local_gpu.ipynb`: notebook piloto para comparar modelos
  candidatos en un subconjunto pequeño antes de lanzar la corrida completa.
  Soporta dos modos vía la variable `PILOT_MODE` en la primera celda:
  `small_1gpu_pair` (Qwen-3B + Gemma-1B en paralelo, una GPU cada uno) o
  `large_2gpu_single` (Qwen-7B o Gemma-4B solo, balanceado entre 2 GPUs).
- `scripts/build_dataset_local_gpu.py`: script de línea de comandos para la
  corrida completa una vez elegido el modelo ganador en el piloto.
- `scripts/benchmark_local_gpu.py`: script de línea de comandos para medir
  tiempo de carga y throughput de generación (contextos/min) de cada
  configuración de modelo sobre una muestra pequeña, antes de lanzar la
  corrida completa.
- `scripts/run_build_parallel.py`: orquestador Python que lanza N workers
  (uno por GPU) para modelos de 1 GPU, espera a que terminen y ejecuta el
  merge de los shards.
- `scripts/run_build_parallel.sh`: wrapper bash equivalente al orquestador
  Python, para lanzar la misma corrida sin escribir comandos largos.
- `scripts/zip_path.py`: utilidad de línea de comandos (solo stdlib) para
  comprimir un archivo o carpeta (p. ej. datasets generados o logs de workers)
  en un `.zip`, sin modificar los originales.

### Requisitos

GPU local (probado para VRAM de 12GB/15GB, cuantización 4-bit por defecto).
Outlines no soporta Python 3.14 (todas sus versiones requieren `<3.14`); usar
un entorno con Python 3.11. El entorno se crea con `environment.yml`, sin
afectar otros entornos usados por los notebooks previos (`dataset_build.ipynb`,
`generative-question-answering-T5/GPT.ipynb`, `question-answering-Bert.ipynb`).

```bash
conda env create -f environment.yml
conda activate pyt-eqa-fge
python -c "import torch; print(torch.cuda.is_available())"  # debe imprimir True
```

Si el driver de la GPU requiere una versión de CUDA distinta a cu124 (revisar
`nvidia-smi`), edita la línea `--extra-index-url` en `environment.yml` antes de
crear el entorno (p. ej. cu121 o cu128).

**Selección de GPUs**: tanto los scripts como el notebook fijan
`CUDA_VISIBLE_DEVICES` a los ids físicos indicados (`--gpu-ids` en los
scripts, `GPU_IDS` en la primera celda del notebook) *antes* de importar
`torch`. Esto solo puede hacerse una vez por proceso: para cambiar de GPU(s) o
de modo de piloto hay que reiniciar el kernel/proceso. Los modelos de 1 GPU
requieren un solo id (p. ej. `--gpu-ids 0`); los de 2 GPUs (`qwen2.5-7b-instruct`,
`gemma-3-4b-it`) requieren exactamente dos (p. ej. `--gpu-ids 4,5`).

Para los modelos de 1 GPU (`qwen2.5-3b-instruct`, `gemma-3-1b-it`) también se
puede lanzar un worker por GPU física con `--worker-id`/`--num-workers`, o usar
`scripts/run_build_parallel.sh`/`.py` que lo hace automáticamente (ver sección
siguiente).

### Uso

1. Abrir `dataset_build_local_gpu.ipynb` en la máquina con GPU, ajustar
   `GPU_IDS`/`PILOT_MODE`/`LARGE_MODEL_KEY` en la primera celda y ejecutar de
   principio a fin. Esto procesa un subconjunto piloto (50 contextos) y
   compara la calidad de las respuestas entre modelos (solo en modo
   `small_1gpu_pair`).
2. Opcionalmente, medir throughput antes de decidir con
   `scripts/benchmark_local_gpu.py` (ver comando abajo).
3. Revisar la sección de comparación y completar la celda de "Decisión" del
   notebook con el modelo elegido.
4. Ejecutar la corrida completa con el script, indicando el modelo elegido:

   ```bash
   python scripts/build_dataset_local_gpu.py \
       --model qwen2.5-7b-instruct \
       --gpu-ids 4,5 \
       --output-file dataset/dataset_squad_v2_localgpu.json
   ```

   Argumentos disponibles: `--model` (`qwen2.5-3b-instruct`,
   `qwen2.5-7b-instruct`, `gemma-3-1b-it` o `gemma-3-4b-it`), `--gpu-ids`
   (ids físicos separados por coma, cantidad debe coincidir con las GPUs que
   requiere el modelo), `--dataset-path`, `--output-file`,
   `--checkpoint-interval`, `--limit` (para pruebas rápidas), `--no-4bit`
   (desactiva la cuantización), `--max-memory-gib` (tope de VRAM por GPU,
   solo aplica a los modelos de 2 GPUs).
5. Por defecto el script procesa el dataset completo; con `--resume` omite
   contextos ya completos y con `--use-dataset-sample` procesa una muestra
   aleatoria reproducible (ver sección siguiente).

### Ejecución en paralelo con réplicas (workers)

Para los modelos de 1 GPU se puede repartir el trabajo entre varias GPUs:
cada worker es un proceso independiente que carga su propia copia del modelo
en una GPU y procesa un subconjunto distinto de contextos. Al final los shards
se unen en un solo JSONL.

El lanzador recomendado es:

```bash
bash scripts/run_build_parallel.sh \
    --model gemma-3-1b-it \
    --gpus 4,5,6,7 \
    --use-dataset-sample --sample-size 17568 --sample-seed 42 \
    --output-file dataset/squadv2_gemma_sample.jsonl
```

Equivalente en Python (misma interfaz):

```bash
python scripts/run_build_parallel.py \
    --model qwen2.5-3b-instruct \
    --gpus 0,1,2,3 \
    --use-dataset-sample --sample-size 10000 --sample-seed 42 \
    --output-file dataset/squadv2_qwen_sample.jsonl
```

Opciones útiles:

- `--dry-run`: muestra los comandos de cada worker y el merge sin ejecutarlos.
- `--no-merge`: lanza los workers pero no ejecuta el merge (para revisar shards).
- `--resume`: reanuda workers interrumpidos.
- `--limit N`: smoke test con solo N contextos.
- `--log-dir DIR`: carpeta de logs de workers (default `logs`).
- `--no-4bit`: desactiva la cuantización 4-bit.

### Nota sobre los logs de workers

Los logs de los workers (`<modelo>_worker_<i>.log` dentro de `--log-dir`) se
**sobrescriben en cada ejecución** (se abren en modo escritura), no se acumulan.
Si quieres conservar las ejecuciones históricas (p. ej. las pruebas de días
distintos) para analizarlas de forma independiente, **respáldalos antes de
volver a lanzar** o usa un `--log-dir` distinto por corrida (p. ej.
`--log-dir logs/qwen_20260829`).

Ejemplo de respaldo antes de repetir una corrida:

```bash
cp -r logs logs_backup_$(date +%Y%m%d_%H%M%S)
```

Recuerda que el `.log` del `nohup` del orquestador (el que rediriges con
`> logs/qwen_dataset.log 2>&1`) sí puede usar cualquier nombre, pero los logs de
los **workers** dependen de `--log-dir` y se sobrescriben.

Para analizar los logs (conteo de `JSON INVALIDO`, `Error crítico`, truncamientos
de JSON, distribución de longitud de respuestas, acuerdo entre Qwen y Gemma,
etc.) usa el script `scripts/logs_reviewer/logs_reviewer.py` (ver su `README.md`).
Apúntalo a la carpeta con los logs de la corrida que quieras revisar
(`--logs-dir`), p. ej.:

```bash
python scripts/logs_reviewer/logs_reviewer.py all \
    --logs-dir logs/qwen_20260829 \
    --qwen-dataset dataset/dataset_squadv2_M1/squadv2_qwen2.5-3b.jsonl \
    --gemma-dataset dataset/dataset_squadv2_M1/squadv2_gemma-3-1b.jsonl
```

Smoke test de 8 contextos con 2 GPUs:

```bash
python scripts/run_build_parallel.py \
    --model gemma-3-1b-it \
    --gpus 4,5 \
    --limit 8 \
    --use-dataset-sample --sample-size 8 --sample-seed 42 \
    --output-file /tmp/e2e_gemma.jsonl
```

También puedes lanzar los workers manualmente con
`build_dataset_local_gpu.py --worker-id i --num-workers N` (uno por GPU) y
luego `build_dataset_local_gpu.py --merge --output-file ...`. El orquestador
espera a que terminen todos los workers, valida que cada `context_id` tenga
5 filas y escribe el archivo final en `--output-file`.

### Reanudar una corrida cambiando el número de GPUs

Si una corrida de `run_build_parallel.sh` se interrumpe (o quieres sumar/quitar
GPUs a mitad de camino), **no** relances simplemente con `--gpus` distinto: el
reparto de contextos por worker depende del número de GPUs (`--num-workers`),
así que cambiarlo reindexa todo y los shards viejos (`.worker-0`, `.worker-1`,
...) dejan de corresponder a lo que le tocaría a cada nuevo worker. Usa en su
lugar `scripts/plan_resume_shards.py`, que reparte sólo lo que falta entre las
GPUs libres, sin reprocesar lo ya generado. Esto no modifica el resume normal
(`--resume`/`--num-workers`) ni el orquestador; es un flujo aparte y opcional.

1. **Revisar el progreso de cada worker antes de matar nada.** Cada worker
   tiene su propio avance y su propio checkpoint (`--checkpoint-interval`,
   default 100). Mira en su log la última línea
   `Checkpoint guardado en contexto N`; si el próximo checkpoint está cerca,
   conviene esperar unos minutos a que aparezca antes de matar ese proceso
   (minimiza cuánto se pierde por buffering no flusheado).

2. **Detener los workers del modelo a reasignar, uno por uno:**

   ```bash
   ps aux | grep "build_dataset_local_gpu.py --model gemma-3-1b-it"
   kill <pid>          # SIGTERM; no uses -9 salvo que no responda en ~10s
   nvidia-smi          # confirmar que la VRAM de esa GPU se liberó
   ```

3. **Generar el manifiesto** con las GPUs realmente libres en ese momento
   (nunca asumas un número fijo; pásalo explícito en `--new-gpus`):

   ```bash
   python scripts/plan_resume_shards.py \
       --model gemma-3-1b-it \
       --output-file ../../data/dataset_squadv2_M2/squadv2_gemma-3-1b.jsonl \
       --sample-size 20000 --sample-seed 42 \
       --new-gpus 0,1,2,3,4,5,6,7 \
       --max-new-tokens 512
   ```

   `--dataset-path`, `--sample-size` y `--sample-seed` deben coincidir
   exactamente con los de la corrida original para que los `context_id` sigan
   alineados al mismo texto. El comando hace backup (`.bak_<timestamp>`) de
   cada shard existente antes de limpiar contextos parciales, imprime cuántos
   contextos ya están completos y cuántos faltan, y escribe en
   `resume_manifests/<modelo>_<timestamp>/` un manifiesto JSON por GPU
   nueva más un `launch_resume.sh` listo para ejecutar.

4. **Lanzar el resume:**

   ```bash
   bash resume_manifests/gemma-3-1b-it_<timestamp>/launch_resume.sh
   ```

   Cada worker corre con `build_dataset_local_gpu.py --resume-manifest ...`,
   que procesa exactamente los índices listados en su manifiesto (no
   recalcula ningún reparto por GPU) y escribe logs en una carpeta **nueva**
   con timestamp (nunca pisa los logs de la corrida anterior).

5. **Merge final**, igual que siempre:

   ```bash
   python scripts/build_dataset_local_gpu.py --merge \
       --output-file ../../data/dataset_squadv2_M2/squadv2_gemma-3-1b.jsonl
   ```

### Benchmark de desempeño

```bash
python scripts/benchmark_local_gpu.py --model gemma-3-4b-it --gpu-ids 6,7 --num-samples 5
```

Mide tiempo de carga del modelo, tiempo promedio de generación por contexto,
contextos/min y uso de VRAM antes/después de cargar el modelo. Cada corrida
agrega una línea JSON a `benchmark_results.jsonl` para comparar configuraciones.

### Empaquetar datasets y logs (zip)

Para comprimir un archivo o carpeta generada (dataset, logs de workers, etc.)
sin borrar los originales:

```bash
# Comprimir una carpeta (dentro del zip se mantiene el nombre de la carpeta):
python scripts/zip_path.py --input dataset

# Comprimir un archivo concreto:
python scripts/zip_path.py --input logs/worker_0.log

# Destino explícito (crea los directorios intermedios y añade .zip si falta):
python scripts/zip_path.py --input logs --output /tmp/logs_backup
```

Opciones:

- `--input PATH` (obligatorio): archivo o carpeta a comprimir.
- `--output PATH` (opcional): ruta del `.zip` de salida. Si se omite, crea
  `<nombre>_<YYYYmmdd_HHMMSS>.zip` junto al origen (nombre de la carpeta, o
  nombre base del archivo sin extensión).

Se incluyen los archivos ocultos y las carpetas vacías; los originales nunca se
modifican.

## Preparación del dataset final y experimentos de QA extractivo (zero-shot vs fine-tuning)

Una vez aplicado el QC (`qc_squadv2_datasets.ipynb` → `out_qc_M2/`), los JSONL de
`out_qc_M2/` ya son el dataset final en formato SQuAD v2 (no requieren más
filtrado). Los siguientes scripts permiten preparar/subir ese dataset y ejecutar
las ablaciones del artículo extendido: comparación zero-shot vs fine-tuning,
ablación por variante de dataset (`merged`, `strict_gemma`, `strict_qwen`),
ablación de modelos (BETO, MMG BETO, XLM-R base, distilled BETO), métricas por
tipo de pregunta, métricas SQuAD v2 HasAns/NoAns y evaluación sobre el gold
audit de 200 filas etiquetadas.

### Preparar y subir el dataset final a Hugging Face

`scripts/prepare_final_dataset.py` no aplica filtros adicionales de calidad:
valida el esquema SQuAD v2, deriva `context_id` del campo `id`, hace un split por
contexto (train/dev/test, default 80/10/10, seed 42), **excluye por defecto los
pares `(context, question)` del gold audit** (`out_qc_M2/audit_stratified_sample_labeled_v1.csv`)
para evitar fuga de datos, escribe `gold_test.jsonl` con las 200 filas del gold
audit y opcionalmente sube `train`/`validation`/`test`/`gold_test` a Hugging Face.

```bash
# Preparar localmente (gold audit excluido por defecto):
python scripts/prepare_final_dataset.py --output-dir dataset/prepared_m2

# Smoke test con solo 20 contextos:
python scripts/prepare_final_dataset.py --limit-contexts 20 --output-dir /tmp/prepared_m2

# Preparar y subir a Hugging Face (usa HUGGINGFACE_TOKEN del .env):
python scripts/prepare_final_dataset.py \
    --input out_qc_M2/squadv2_final_merged.jsonl \
    --output-dir dataset/prepared_m2 \
    --repo-id LeninGF/question-answering-robbery-m2 \
    --push

# Desactivar la exclusión del gold audit (no recomendado):
python scripts/prepare_final_dataset.py --no-exclude-gold
```

También sigue disponible el modo `--push-only` de
`scripts/build_dataset_local_gpu.py` para subir un JSONL sin hacer split:

```bash
python scripts/build_dataset_local_gpu.py --push-only \
    --input-file out_qc_M2/squadv2_final_merged.jsonl \
    --repo-id LeninGF/question-answering-robbery-m2
```

### Ejecutar una ablación (un experimento por proceso)

`scripts/run_qa_ablation.py` está diseñado para ejecutarse de forma
**independiente por experimento**: una invocación = un `(modelo, dataset, modo)`
en una GPU. Esto permite lanzar procesos en paralelo, uno por GPU.

```bash
# Un experimento (fine-tuning) en la GPU 0:
python scripts/run_qa_ablation.py \
    --model mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es \
    --dataset merged --mode ft --gpu 0 \
    --output-dir out_experiments/run1

# Smoke test rápido (100 contextos, 1 época):
python scripts/run_qa_ablation.py \
    --model mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es \
    --dataset merged --mode both --gpu 0 \
    --limit-contexts 100 --epochs 1 \
    --output-dir /tmp/qa_smoke

# Imprimir la matriz completa (4 modelos x 3 datasets x zsl/ft) con GPUs round-robin:
python scripts/run_qa_ablation.py --plan-only --plan-gpus 8 \
    --output-dir out_experiments/run1

# Ejecutar toda la matriz secuencialmente en una GPU:
python scripts/run_qa_ablation.py --all --gpu 0 --output-dir out_experiments/run1

# Usar el dataset final preparado localmente (gold audit ya excluido):
python scripts/run_qa_ablation.py \
    --model mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es \
    --dataset merged --mode both --gpu 0 \
    --data-dir dataset/prepared_m2 --output-dir out_experiments/run1

# Usar el dataset final directamente desde Hugging Face:
python scripts/run_qa_ablation.py \
    --model mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es \
    --dataset merged --mode both --gpu 0 \
    --hf-dataset LeninGF/question-answering-robbery-m2 --output-dir out_experiments/run1
```

Cada experimento escribe sus resultados en
`out_experiments/<run_id>/<modelo>/<dataset>/<modo>/`: `metrics_summary.csv/json`,
`metrics_by_question_type.csv`, `gold_audit_metrics.json`, `predictions_test.jsonl`,
`config.json` y, para fine-tuning, `training_history.csv` +
`training_curves.png/pdf` (pérdida y EM/F1 por época).

### Lanzador paralelo: Opción A vs Opción B

`scripts/run_ablation_parallel.sh` lanza la matriz en hasta 8 GPUs. Hay dos modos:

- **Opción A (por defecto) — ablación de variantes del dataset**: usa los JSONL
  de QC en disco (`out_qc_M2/`): `merged`, `strict_gemma` y `strict_qwen`.
  Matriz 4 modelos × 3 datasets × {zsl, ft} = 24 experimentos. Nota: este camino
  **no** excluye las filas del gold audit del entrenamiento; usa `test` para las
  comparaciones principales y trata `gold_audit_metrics` como sanity check con
  posible fuga.
- **Opción B — dataset final merged (gold-safe)**: pasa `--data-dir` o
  `--hf-dataset`; usa los splits preparados (gold audit ya excluido) y solo el
  dataset `merged` (8 experimentos).

```bash
# Opción A (24 experimentos, QC local):
bash scripts/run_ablation_parallel.sh \
    --output-dir out_experiments/run1 \
    --gpus "0 1 2 3 4 5 6 7"

# Opción B desde Hugging Face (8 experimentos, epochs y early stopping):
bash scripts/run_ablation_parallel.sh \
    --output-dir out_experiments/run1 \
    --gpus "0 1 2 3 4 5 6 7" \
    --hf-dataset LeninGF/question-answering-robbery-m2 \
    --epochs 20 \
    --early-stopping-patience 3

# Opción B desde splits locales preparados:
bash scripts/run_ablation_parallel.sh \
    --output-dir out_experiments/run1 \
    --gpus "0 1 2 3 4 5 6 7" \
    --data-dir dataset/prepared_m2 \
    --epochs 20

# Ver qué comandos se lanzarían sin ejecutarlos:
bash scripts/run_ablation_parallel.sh --dry-run --gpus "0 1 2"
```

Opciones del lanzador:

- `--output-dir DIR`, `--gpus "0 1 ..."`, `--dry-run`
- `--epochs N` (por defecto 10 en `run_qa_ablation.py`)
- `--early-stopping-patience N` (detiene si `eval_f1` no mejora en N evaluaciones)
- `--data-dir DIR` / `--hf-dataset REPO` (Opción B)
- `--limit-contexts N` (smoke tests)

Para una matriz personalizada, genera los comandos con `--plan-only` y edítalos:

```bash
python scripts/run_qa_ablation.py --plan-only --plan-gpus 8 \
    --hf-dataset LeninGF/question-answering-robbery-m2 \
    --epochs 20 --early-stopping-patience 3 \
    --output-dir out_experiments/run1
```

### Test de meseta de F1 (¿cuántas épocas usar?)

Antes de la corrida completa conviene ver en qué época el F1 de validación se
estabiliza con un solo modelo (el baseline mrm8488):

```bash
# Por defecto: 20 épocas, GPU 0, dataset HF LeninGF/question-answering-robbery-m2
bash scripts/test_f1_plateau.sh

# Con splits locales:
bash scripts/test_f1_plateau.sh --data-dir dataset/prepared_m2

# Ajustar duración y criterio:
bash scripts/test_f1_plateau.sh --epochs 30 --gpu 1 --tolerance 0.2 --min-epochs 5
```

El script entrena un solo modelo (fine-tuning, sin early stopping) y luego llama
a `scripts/find_f1_plateau.py`, que imprime por época el `eval_f1`, la mejor
época y la primera época de meseta. Con eso eliges `--epochs` (o
`--early-stopping-patience 2-3`) para la corrida completa.

### Utilidades compartidas

`scripts/qa_dataset_utils.py` contiene la lógica común usada por los scripts
anteriores: carga/validación de JSONL SQuAD v2, derivación de `context_id`,
split por contexto, mapeo de tipo de pregunta, lectura del gold audit y métricas
SQuAD v2 locales (EM/F1, HasAns/NoAns). No requiere GPU.

### Reporte de resultados

`experiments_report.ipynb` lee el directorio de salida
(`out_experiments/<run_id>/`) y genera las tablas/figuras del artículo:
zero-shot vs fine-tuning, heatmap modelo×dataset, F1 por tipo de pregunta,
métricas HasAns/NoAns, gold audit y curvas de entrenamiento. Los CSV, LaTeX y
PNG/PDF se exportan a `out_experiments/<run_id>/report/`. Solo hay que editar la
variable `OUT_DIR` en la primera celda del notebook.

### Dependencias

`environment.yml` incluye ahora `evaluate` para compatibilidad con el notebook
original; los scripts nuevos usan una implementación local de las métricas
SQuAD v2 y las dependencias pesadas (torch, transformers, datasets) se importan
solo cuando se ejecuta un experimento.

