### ICC - Trabajo Final: MP-SENet para la Mejora de Voz

**Nota:** El código y los resultados de MP-SENet están siendo implementados como parte del trabajo final del curso de  investigación en Ciencias de la Computación en la UNSA.

**Resumen:**
El trabajo presenta MP-SENet, una red de mejora de voz que realiza denoising de espectros de magnitud y fase en paralelo. MP-SENet utiliza una arquitectura de códec con transformers aumentados por convolución para conectar el codificador y el decodificador. El codificador convierte las representaciones tiempo-frecuencia de los espectros ruidosos en una forma compacta, mientras que el decodificador está compuesto por un decodificador de máscara de magnitud y un decodificador de fase que recuperan los espectros limpios de magnitud y fase. Las pérdidas a múltiples niveles en los espectros de magnitud, fase, complejos y en el dominio del tiempo se utilizan para entrenar el modelo. Los resultados experimentales muestran que MP-SENet alcanza un PESQ de 3.50 en el conjunto de datos VoiceBank+DEMAND y supera los métodos avanzados existentes.

**Implementación y Requisitos:**
El código de MP-SENet está disponible en [este repositorio](https://github.com/yxlu-0102/MP-SENet). Para utilizarlo, se requiere:
1. Python >= 3.6.
2. Clonar el repositorio.
3. Instalar los requisitos de Python especificados en `requirements.txt`.
4. Descargar y extraer el conjunto de datos VoiceBank+DEMAND, y asegurarse de que todos los archivos wav estén a 16 kHz.

Para el entrenamiento y la inferencia, se deben seguir las instrucciones proporcionadas en el repositorio.

**Comparación con Otros Modelos:**
Se proporciona una comparación visual con otros modelos de mejora de voz en el repositorio. MP-SENet demuestra un rendimiento superior en la mejora de la calidad del habla en comparación con los enfoques existentes.

**Agradecimientos:**
El desarrollo de MP-SENet se basó en trabajos previos como [HiFiGAN](https://github.com/jik876/hifi-gan), [NSPP](https://github.com/YangAi520/NSPP) y [CMGAN](https://github.com/ruizhecao96/CMGAN).

**Citación:**
```
@inproceedings{lu2023mp,
  title={{MP-SENet}: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra},
  author={Lu, Ye-Xin and Ai, Yang and Ling, Zhen-Hua},
  booktitle={Proc. Interspeech},
  pages={3834--3838},
  year={2023}
}
```