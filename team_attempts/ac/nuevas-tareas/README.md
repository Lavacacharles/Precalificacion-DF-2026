# Nuevas tareas

Para plantear estos nuevos experimentos, considera todo lo aprendido durante el EDA y los análisis posteriores: características de los datos, particularidades, inconsistencias y otros hallazgos. Ten en cuenta también los resultados de los modelos *baseline* —como la importancia de las variables y los algoritmos con mejor desempeño— y el entrenamiento de los modelos finales.

Como sabes, las métricas de AUC están alrededor de 0.63, por lo que todavía hay bastante margen de mejora. La idea es desafiar los datos: no asumir que cuentan toda la verdad, explorar cómo pudieron haberse generado y evaluar distintas hipótesis.

## Ideas

- Como prueba inicial, entrenar un modelo que prediga si un cliente se convertirá o no, independientemente de cuándo lo haga. Es decir, ignorar la variable temporal y evaluar la conversión en general. Esto puede darnos una idea de qué tan útiles son los datos para predecir conversiones.

- Crear una variable que represente, de alguna manera, el «éxito» de una persona. Por ejemplo, podría asignar un puntaje alto a alguien joven y con ingresos elevados.

- Suponer que los datos son una «foto de entrada arrastrada hacia adelante» y hacer *feature engineering* a partir de esa hipótesis. Esto implicaría recalcular variables que podrían verse afectadas, como `dias_ultima_transaccion`.

- Evaluar técnicas de *oversampling* o la creación de datos sintéticos. También se podría completar «hacia atrás» el historial de los clientes que se convirtieron. Si aparecen nuevas cohortes cada mes y un cliente comienza a figurar en el mes 3, ¿sería posible crear un historial para los meses 1 y 2?

- Replantear el tipo de problema. Por ejemplo, predecir cuándo se convertirá un cliente en lugar de predecir únicamente si se convertirá. También podría probarse una combinación de ambos enfoques o varios modelos, uno por cada cantidad de meses hasta la conversión.

- Explorar la posibilidad de que cada «mes» represente en realidad un año. Esto explicaría por qué `dias_ultima_interaccion` y `dias_ultima_transaccion` solo llegan hasta 364: podrían indicar la última interacción o transacción de un año, no de un mes. Si `dias_ultima_transaccion` se repite, quizá sus valores eran nulos y se completaron con un dato conocido de otro año. Esta hipótesis podría explicar la regularidad de los datos y la poca variación de las variables.

- Revisar si tiene sentido que un usuario posea una tarjeta de crédito y no la use. Por ejemplo, si `tiene_tarjeta_credito` es verdadero, pero la última transacción ocurrió hace mucho tiempo, podría existir una inconsistencia: tal vez la variable booleana debería ser falsa o la última transacción debería ser más reciente.
