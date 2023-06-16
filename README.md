# DL-Image-Colorization
This project delves into the applications of Image Colorization utilizing state-of-the-art Deep Learning techniques. The results of our analysis can be seen in the form of a [blog post](https://www.notion.so/Computer-Vision-Project-8bf621b364b446c6bfee0218c7187ae3?pvs=4).
![Alt text](image-1.png)
## Running the Code
In order to run the model, we advise you build a separate anaconda environment for each modesl. The models can be found in the [src/models](src/models) folder. The requirements for each model can be found in the same folder. You also have detailed instructions on how to run the models in the same folder.

## Metrics
All the output images can be found in the [data/imgs_out](data/imgs_out) folder.
To quantitatively evaluate all output images, run the following command:
```
python src/metrics/evaluate.py
```
To qualitatively evaluate all output images, run the following command:
```
python src/analysis/vector_scope.py
```