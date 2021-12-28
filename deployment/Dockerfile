FROM pytorch/torchserve:latest-cpu

# install dependencies
#RUN pip3 install transformers

# copy model artifacts, custom handler and other dependencies
COPY zin_vert_handler.py /home/model-server/
COPY ./itos_de.json.json /home/model-server/
COPY ./itos_en.json.json.json /home/model-server/
COPY ./stoi_de.json.json.json /home/model-server/
COPY ./stoi_en.json.json.json /home/model-server/
COPY constants.pkl /home/model-server/
COPY JIT/model.pt /home/model-server/

# create torchserve configuration file
USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties
USER model-server

# expose health and prediction listener ports from the image
EXPOSE 7080
EXPOSE 7081

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f \
  --model-name=zin_vert_without_att \
  --version=1.0 \
  --serialized-file=/home/model-server/model.pt \
  --handler=/home/model-server/custom_text_handler.py \
  --extra-files "./constants.pkl,./itos_de.json,./itos_en.json,./stoi_de.json,./stoi_en.json" \
  --export-path=/home/model-server/model-store

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "zin_vert_without_att=zin_vert_without_att.mar", \
     "--model-store", \
     "/home/model-server/model-store"]