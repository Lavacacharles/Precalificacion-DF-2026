# Reporte de la sesión experimental

## Resultado principal

Se completaron **66 ajustes**: 14 alternativas mensuales en cuatro cortes (13 CatBoost y un Cox),
cuatro diagnósticos de conversión observada y seis modelos de horizonte a tres meses.
**Ninguna propuesta demuestra una mejora estable del hazard mensual.** Sí aparecen resultados útiles
para distinguir información del perfil, oportunidad de seguimiento y ruido del dataset.

El candidato elegido con agosto–octubre fue **Edad_ingreso** (AUC media 0.6314 frente a 0.6308 del
control). En noviembre pierde: **0.6198 frente a 0.6209**. ΔAUC = **−0.0011**, con IC bootstrap
pareado 95% **[−0.0047, +0.0025]**. No hay evidencia de superioridad ni de una caída concluyente.

El control usa 33 features de historia y parámetros fijos; no es el baseline anterior con 30 features
ni aquel que seleccionaba iteraciones usando noviembre. Comparar directamente sus números sería engañoso.

## 1. Conversión «sin importar cuándo»

Solo puede observarse **conversión hasta noviembre**, no «convertirá alguna vez». Los ceros restantes
están censurados. Una fila de entrada por cliente y split aleatorio por cliente, sin identidad compartida:

| Diagnóstico | AUC | Qué indica |
|---|---:|---|
| Perfil, sin tiempo explícito | 0.7055 | Hay señal para conversión acumulada observada. |
| Solo cohorte de entrada | 0.6981 | Buena parte de la discriminación procede de la oportunidad de seguimiento. |
| Perfil + cohorte | 0.7864 | Combina perfil y horizonte disponible; no mejora comparable del problema mensual. |
| Solo cohorte enero, horizonte común de 11 meses | 0.7081 | La señal del perfil persiste al igualar seguimiento. |

La prevalencia es 67.28% en el holdout general y 81.12% en el de enero. Cambian etiqueta, población y
horizonte respecto al hazard mensual. Son diagnósticos retrospectivos: no una validación fuera de tiempo.
**Conclusión:** la data contiene señal de propensión acumulada; la AUC 0.7864 no permite afirmar que
podamos alcanzar ese ranking para un único mes.

## 2. Proxies de ingreso/edad y dispositivo

Se añadieron ingreso/edad, ingreso/años desde los 18 y diferencia de logaritmos. Para dispositivo,
se usaron cruces con bandas de ingresos, préstamo y ratio de deuda. Se conservaron las variables originales.
No se asignaron etiquetas personales de «éxito», apariencia o ahorro, que no están observadas.

| Modelo | AUC media ago–oct | AUC noviembre |
|---|---:|---:|
| Historia control | 0.6308 | **0.6209** |
| Ingreso/edad | **0.6314** | 0.6198 |
| Dispositivo/ingreso | 0.6308 | 0.6179 |
| Ambos grupos | 0.6302 | 0.6195 |

**Conclusión:** diferencias pequeñas e inestables. No hay apoyo predictivo consistente para estas
representaciones ni evidencia que valide las interpretaciones psicológicas sugeridas.

## 3. Foto de entrada y reloj mensual o anual

Se agregaron cinco valores reconstruidos desde la primera observación: antigüedad de cuenta/dirección,
edad, recencia de transacción e interacción inicial. El reloj mensual usa meses observados y días reales
del calendario; el anual usa 12 meses y 365.25 días por paso. El crecimiento de recencias presupone,
adicionalmente, que no hubo nuevos eventos. Son proxies hipotéticos, no correcciones de la verdad.

| Escenario | AUC media ago–oct | AUC noviembre |
|---|---:|---:|
| Control | 0.6308 | **0.6209** |
| Foto de entrada + reloj mensual | 0.6299 | 0.6183 |
| Foto de entrada + reloj anual | 0.6297 | 0.6198 |
| Eliminar interacción y sus derivados | 0.6309 | 0.6167 |

Ninguno mejora consistentemente. Que una recencia llegue hasta 364 no demuestra que cada periodo sea
un año: también encaja con recorte de rango o generación artificial. Interpretar pasos como años no
explica que edad y antigüedades sigan congeladas. Bajo la hipótesis alternativa de **foto final** y la
misma cuenta, en train hay 284 incompatibilidades con reloj mensual (margen de un mes) y 4,380 con
reloj anual (margen de doce meses). No son los conteos del reporte train+test previo: cambia el universo.

**Conclusión:** no encontramos motivo para adoptar el reloj anual ni para reemplazar automáticamente
las variables. La ausencia de mejora tampoco demuestra qué fecha de referencia es correcta.

## 4. Oversampling y datos sintéticos

Se balanceó únicamente el entrenamiento de cada corte. SMOTENC conserva categorías y booleanos,
escala numéricas con train y redondea contadores. No se incluyeron IDs ni meses en su espacio de síntesis.
La corrección de prior utiliza solo la tasa de entrenamiento; no recalibra contra noviembre.

| Modelo en noviembre | AUC | Brier ↓ | Probabilidad media |
|---|---:|---:|---:|
| Perfil control | **0.6179** | **0.1248** | 13.87% |
| Duplicación aleatoria, bruto | 0.6155 | 0.2179 | 45.04% |
| Duplicación, prior corregido | 0.6155 | 0.1249 | 13.55% |
| SMOTENC, bruto | 0.5860 | 0.2060 | 38.42% |
| SMOTENC, prior corregido | 0.5860 | 0.1312 | 12.07% |

La tasa real es **15.15%**. Corregir odds recupera gran parte de la calibración de la duplicación,
pero no su ranking. SMOTENC empeora AUC en las cuatro ventanas (0.5885, 0.5792, 0.5743, 0.5860).
La corrección de prior no repara la geometría sintética ni garantiza calibración.
**Conclusión:** no recomendar estas formas de sobremuestreo para este dataset y configuración.
La degradación sugiere que aquí la interpolación introduce distorsión; no identifica por sí sola su causa.

## 5. Inventar meses anteriores a la entrada

Completar solo futuros conversores añadiría **37,593 ceros** y bajaría artificialmente la tasa
fila–mes de **15.05% a 11.22%**. Completar a todos los entrantes añadiría **77,279 filas**.
Hay **257 entrantes** cuya antigüedad ni siquiera permite retroceder hasta enero con una misma cuenta,
incluso dando un mes de margen.

**No se entrenó con esas filas como si fueran datos reales.** Elegir conversores usa el desenlace para
seleccionar el pasado; incluso completar a todos requiere suponer población en riesgo, covariables y
etiquetas no observadas antes de la entrada. Se hizo el diagnóstico cuantitativo y se optó por censura/
entrada al panel explícitas. Para probarlo como una simulación contrafactual separada haría falta definir
qué proceso de observación produjo las altas tardías; los CSV no lo identifican.

## 6. Predecir cuándo convierte

**Cox con censura:** AUC media de selección **0.6198**, y **0.6013** en noviembre. No mejora CatBoost.
El ajuste convergió numéricamente. Usa riesgos proporcionales y Breslow con meses discretizados;
en noviembre 2,259 clientes necesitan extrapolar el hazard basal más allá del seguimiento de train.
Por tanto, este resultado no invalida toda la familia de supervivencia.

**Horizonte de tres meses:** se entrenaron un binario y un multiclase (sin evento / mes 1 / mes 2 / mes 3).
Las cohortes de entrenamiento tienen todos sus labels maduros antes del origen de evaluación.

| Cohorte evaluada | Última cohorte de train | AUC binario 3m | AUC multiclase, sumando eventos |
|---|---|---:|---:|
| Julio | Abril | 0.6737 | 0.6784 |
| Agosto | Mayo | 0.6561 | 0.6588 |
| Septiembre | Junio | 0.6686 | 0.6600 |

La clasificación multiclase no gana consistentemente. Entre quienes convierten dentro de tres meses,
el error absoluto de su mediana predicha es **0.680 / 0.713 / 0.668 meses**, frente a
**0.683 / 0.717 / 0.661** del baseline constante. Las diferencias son mínimas y no se sostiene la mejora.
La evaluación del tiempo está condicionada a eventos dentro del horizonte; no incluye tiempos futuros
de los censurados.

**Conclusión:** podemos ordenar propensión a convertir durante un horizonte, pero estos modelos aportan
poca precisión adicional para ubicar el mes exacto. No se ha identificado una clase de «nunca convertirá».

## 7. Tarjeta y recencia

Entre clientes únicos de train, el **50.58%** de quienes tienen tarjeta tiene recencia general mayor a
180 días; entre quienes no tienen tarjeta es **50.00%**. No se documenta que esa transacción sea de tarjeta.

| Tratamiento | AUC media ago–oct | AUC noviembre |
|---|---:|---:|
| Conservar datos: historia control | **0.6308** | **0.6209** |
| Agregar cruces tarjeta–recencia | 0.6284 | 0.6193 |
| Forzar flag a falso si recencia >180 | 0.6270 | 0.6165 |
| Limitar recencia a 90 si tiene tarjeta | 0.6263 | 0.6147 |

**Conclusión:** tener tarjeta sin usarla es compatible con su vigencia, y la transacción del cliente
puede pertenecer a otro producto. Las alteraciones ensayadas empeoran el modelo y no tienen fundamento
semántico suficiente para considerarlas limpieza.

## Decisión práctica y límites

- Mantener el hazard mensual como formulación alineada con diciembre y conservar los valores originales.
- No promover proxies sociales, reloj anual, correcciones de tarjeta ni SMOTENC sobre la base de estos resultados.
- Para decisiones de campaña a tres meses, usar un problema separado con horizonte maduro; no comparar
  su AUC con la del mes siguiente ni con conversión acumulada de horizontes heterogéneos.
- La falta de mejora no prueba que 0.63 sea el límite: estas son familias, parámetros y semillas concretas.
  Noviembre ya ha sido inspeccionado varias veces y no es una prueba externa independiente.
- No se leyó test ni se reemplazó la entrega de diciembre. La historia inventada se diagnosticó, no se
  presentó como entrenamiento sin fuga. Las covariables del CSV aún carecen de timestamps de procedencia.

Notebook ejecutado: `Sesion_experimental.ipynb`. Código: `experimentos.py` y `sesion.py`.
Reproducción desde la raíz: `.\.venv\Scripts\python.exe nuevas-tareas\ejecutar.py`.
Métricas, predicciones y manifiesto de selección: `resultados/`.
