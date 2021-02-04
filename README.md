# Deep Q Snake

Aplicando DQN en el juego Sanke con Python.

<p align="center">
  <img width="400" height="400" src="/Images/animation.gif">
</p>

snake_env.py contiene el environment utilizado para entrenar el modelo.

deep_q_learning.py contiene el código de entrenamiento del modelo utilizando DQN, para esto crea una y entrena una red utilizando la ecuación de Bellman.

Entrenar el modelo llevó mucho tiempo de entrenamiento en plataformas como Google Colab. El modelo jugó en el simulador 16000 veces, lo que le tomó más de 20 horas de entrenamiento con TPUs prestadas por la plataforma Google Colab.

El modelo utilizado es muy simple, y se puede ver en la siguiente imágen

<p align="center">
  <img width="400" height="400" src="/Images/model.png">
</p>

## Análisis

Llegados a este punto, a medida que pasan las iteraciones, el modelo no presenta grandes mejoras, esto se puede ver claramente en el siguiente gráfico

<p align="center">
  <img width="680" height="480" src="/Images/agerage_scores.jpg">
</p>

## Testing

Si desea probar en funcionamiento el modelo, deberá clonar el repositorio y ejecutar el archivo bot_playing.py

