# Hipótesis sobre el origen del panel

**Conclusión:** la evidencia favorece un panel construido con perfiles fijos por cliente y una recencia alterada progresivamente por calendario. Puede ser un dataset sintético o una extracción que combina una tabla estática con observaciones mensuales. Los CSV no permiten identificar cuándo se midió originalmente cada perfil.

Se revisaron train y test: **120,000 filas, 26,467 clientes y 93,533 pares de meses consecutivos**. Las pruebas no usan etiquetas desconocidas ni modifican los modelos. Las hipótesis no son necesariamente excluyentes.

## Escenarios contrastados

| Hipótesis | Comprobación y resultado | Evaluación |
|---|---|---|
| **1. Foto final propagada hacia atrás.** Los atributos corresponden al último registro del cliente. | Si la antigüedad describe la misma cuenta, al restar el seguimiento aparecen **416 clientes con antigüedad inicial negativa** entre 20,695 clientes repetidos. Incluso permitiendo un mes de margen por fechas y redondeo, persisten **313**, incluidos **125 con conversión observada**. | **Contradice una explicación universal con antigüedades correctas y la misma cuenta.** No descarta que algunos campos se hayan actualizado retrospectivamente, que cambie la cuenta de referencia o que existan errores. |
| **2. Foto de entrada arrastrada hacia adelante.** El perfil refleja la primera observación y no vuelve a actualizarse. | Los 21 predictores estáticos conservan exactamente su valor, incluidos ingresos y saldo con decimales. Una antigüedad inicial positiva no genera la contradicción del escenario anterior. | **Compatible, pero no demostrable.** Que el primer registro sea el primero disponible no prueba que allí se midiera el perfil. |
| **3. Unión incorrecta con una tabla maestra sin histórico.** Se unió por cliente, omitiendo fecha/vigencia. | Un único perfil por ID repetido en todos los meses es exactamente la firma esperada. La recencia dinámica podría venir de otra tabla o transformación. | **Compatible e indistinguible de una foto fija con estos CSV.** Si la tabla maestra contiene información posterior al evento, habría fuga de origen. |
| **4. Variables realmente mensuales, pero poca actividad o redondeo.** | No explica conjuntamente saldo e ingresos idénticos, antigüedades congeladas y días desde transacción constantes. Además, la interacción aumenta **más de 62 días entre meses consecutivos en 21,212 pares**; el máximo es **+362 días**. | **Muy poco compatible con mediciones mensuales correctas.** Esos saltos exigirían revisiones, cambios de definición o errores; nuevas interacciones normales no los explican. |
| **5. Sistema que no actualiza el perfil durante el periodo.** | Hay **1,963 clientes presentes de enero a diciembre** sin cambios en ninguno de los 21 atributos estáticos. | **Compatible con valores almacenados desactualizados**, no con antigüedades/recencias calculadas realmente a cada mes. No se distingue de una foto de entrada o tabla maestra fija. |
| **6. Cambio de ID cuando cambia el perfil.** Cada ID representaría una versión y no una persona estable. | Los **26,467 perfiles estáticos completos son distintos**; no se encontraron clones exactos con IDs diferentes. Las salidas anteriores a diciembre coinciden exactamente con conversiones, sin salidas adicionales atribuibles a renovaciones de ID. | **Sin apoyo directo.** Se descarta el simple reciclado de perfiles idénticos, pero no se puede enlazar personas que cambien tanto ID como atributos sin otra llave. |
| **7. Panel generado/sintético con perfiles muestreados una vez.** | En cada mes se conserva exactamente a los sobrevivientes y se agregan IDs nuevos en bloques consecutivos. No hay huecos. Hay masas exactas en límites: **1,754 clientes con saldo 300** y **432 con ingresos 18,000**. Se suma la regularidad de la recencia descrita abajo. | **La explicación conjunta más respaldada**, aunque ninguna de estas propiedades por separado prueba un origen sintético. También puede ser una extracción deliberadamente preparada para supervivencia. |
| **8. Interacción derivada de transacción y degradada mediante reemplazos/reordenamientos.** | Igualdad completa en enero; caída casi lineal mes a mes; en diciembre los dos vectores tienen **exactamente los mismos valores y frecuencias**, pero correlación **−0.0028** y solo **26 coincidencias entre 9,900 clientes**. | **Muy fuertemente respaldada como transformación de datos.** En diciembre el vector de interacción es una permutación del vector de transacción en términos de valores; no se demuestra qué algoritmo lo produjo ni si fue intencional o un error de alineación. |

## Dos hallazgos que distinguen escenarios

**La foto final tiene contraejemplos concretos.** El cliente **156** aparece desde enero hasta su conversión en septiembre. Su antigüedad de cuenta permanece en **6 meses** durante los nueve registros. Si seis meses fuera la edad de esa misma cuenta en septiembre, no podría corresponder a una cuenta ya existente en enero. Esta contradicción persiste con un mes de margen. No es necesario conocer su comportamiento después de convertir.

**La recencia parece tener un reloj de calendario, no de usuario.** El porcentaje de igualdad entre interacción y transacción es:

| Mes | Igualdad exacta |
|---|---:|
| Enero | 100.00% |
| Abril | 72.82% |
| Julio | 45.66% |
| Octubre | 18.43% |
| Noviembre | 9.40% |
| Diciembre | 0.26% |

Una explicación candidata conserva el valor original con proporción `1 − p` y lo reemplaza con proporción `p`, donde `p = (mes − 1) / 11`. Al considerar coincidencias accidentales entre valores, reproduce las proporciones observadas con un error máximo de **0.114 puntos porcentuales**. La relación lineal descriptiva tiene **R² = 0.999998**.

No es una prueba estadística confirmatoria: la hipótesis se planteó después de observar el patrón y distintos algoritmos podrían generarlo. En meses intermedios los conjuntos de valores no son idénticos, de modo que no puede afirmarse que cada mes sea una permutación íntegra. Una mezcla de valores originales y valores de un vector reordenado es compatible.

La coincidencia depende también del calendario entre clientes que acaban de entrar: **89.79%** entre las primeras observaciones de febrero frente a **8.93%** entre las de noviembre. Esto debilita la explicación de que la diferencia se acumule simplemente por el tiempo que cada cliente lleva observado.

## Implicaciones

1. **No corregir automáticamente antigüedades o recencias.** Sumar/restar tiempo adopta una fecha de referencia que todavía no conocemos. Las variantes deben etiquetarse como escenarios, no como recuperación de la verdad.
2. **Distinguir fuga de código de fuga de origen.** Nuestras pruebas anteriores verifican cálculos causales sobre el CSV. Si sus atributos se actualizaron retrospectivamente después del evento, un split temporal no elimina esa contaminación.
3. **Pedir cuatro datos de procedencia:** fecha de medición/vigencia y disponibilidad de cada perfil; clave y condición temporal del join; estabilidad de la cuenta asociada al cliente; lógica que produce `dias_ultima_interaccion`.
4. **Prioridad analítica:** tratar los atributos congelados como perfiles de fecha desconocida y la interacción como señal posiblemente alterada. El escenario de generación artificial merece más peso que una evolución real reconstruible con simples sumas de meses.

La comprobación es reproducible con `auditar_origen_panel.py`. Las cifras, contraejemplos y tablas mensuales están en `auditoria_origen_panel/`, especialmente `evidencia.json`, `snapshot_final.csv` y `ruido_recencias_por_mes.csv`.
