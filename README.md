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
5. El script reprocesa siempre el dataset completo (no hay deduplicación ni
   reanudación automática).

### Benchmark de desempeño

```bash
python scripts/benchmark_local_gpu.py --model gemma-3-4b-it --gpu-ids 6,7 --num-samples 5
```

Mide tiempo de carga del modelo, tiempo promedio de generación por contexto,
contextos/min y uso de VRAM antes/después de cargar el modelo. Cada corrida
agrega una línea JSON a `benchmark_results.jsonl` para comparar configuraciones.

