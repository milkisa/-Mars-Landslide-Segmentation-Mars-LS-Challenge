FROM nvcr.io/nvidia/pytorch:25.01-py3
RUN pip install timm==1.0.19
RUN pip install segmentation-models-pytorch==0.5.0
# --- Make numpy+opencv consistent ---

COPY requirements_2.txt .
RUN pip install -r requirements_2.txt
