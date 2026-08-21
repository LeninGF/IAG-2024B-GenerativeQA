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
  generación (carga de modelo 4-bit con `transformers` + `bitsandbytes`,
  generación de JSON con `outlines`). Es importado tanto por el notebook
  piloto como por el script de ejecución completa.
- `dataset_build_local_gpu.ipynb`: notebook piloto para comparar modelos
  candidatos (Qwen2.5-7B-Instruct y Gemma-3-4B-IT) en un subconjunto pequeño
  antes de lanzar la corrida completa.
- `scripts/build_dataset_local_gpu.py`: script de línea de comandos para la
  corrida completa una vez elegido el modelo ganador en el piloto.

### Requisitos

GPU local (probado para VRAM de 12GB/15GB, cuantización 4-bit obligatoria).
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

### Uso

1. Abrir y ejecutar `dataset_build_local_gpu.ipynb` de principio a fin en la
   máquina con GPU. Esto carga ambos modelos (uno por GPU), procesa un
   subconjunto piloto (50 contextos) y compara la calidad de las respuestas
   entre modelos.
2. Revisar la sección de comparación y completar la celda de "Decisión" del
   notebook con el modelo elegido.
3. Ejecutar la corrida completa con el script, indicando el modelo elegido:

   ```bash
   python scripts/build_dataset_local_gpu.py \
       --model qwen2.5-7b-instruct \
       --device cuda:0 \
       --output-file dataset/dataset_squad_v2_localgpu.json
   ```

   Argumentos disponibles: `--model` (`qwen2.5-7b-instruct` o
   `gemma-3-4b-it`), `--device`, `--dataset-path`,
   `--output-file`, `--checkpoint-interval`, `--limit` (para pruebas rápidas),
   `--no-4bit` (desactiva la cuantización).
4. El script reprocesa siempre el dataset completo (no hay deduplicación ni
   reanudación automática).

